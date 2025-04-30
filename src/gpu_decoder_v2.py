"""
gpu_decoder_v2.py
=================
GPUMemoryManagerV2 を用いて

  1. pass‑1 (len/null 収集) - GPUで実行
  2. prefix‑sum で offsets / total_bytes - GPU(CuPy)で実行
  3. pass‑2 (scatter‑copy, NUMERIC→UTF8 文字列化) - GPUカーネルで実行
  4. Arrow RecordBatch を生成 - CPUで実行

までを 1 関数 `decode_chunk()` で実行するラッパ。

前提
-----
* PostgreSQL COPY BINARY 解析済みの
    - `field_offsets` (rows, ncols) int32 DeviceNDArray
    - `field_lengths` (rows, ncols) int32 DeviceNDArray
  を受け取る。 (現在は `gpu_parse_wrapper.py` の CPU 実装の結果)

* ColumnMeta 配列は `meta_fetch.fetch_column_meta()` で取得済みとする。
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import cupy as cp
import pyarrow as pa
import pyarrow.compute as pc 
try:
    import pyarrow.cuda as pacuda
except ImportError:
    pacuda = None

from numba import cuda

from .type_map import *
from .gpu_memory_manager_v2 import GPUMemoryManagerV2

from .cuda_kernels.arrow_gpu_pass1 import pass1_len_null # Use Pass 1 GPU Kernel
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.arrow_gpu_pass2_fixed import pass2_scatter_fixed
from .cuda_kernels.arrow_gpu_pass2_decimal128 import pass2_scatter_decimal128 # Import the new kernel
from .cuda_kernels.numeric_utils import int64_to_decimal_ascii  # noqa: F401  (import for Numba registration)

def build_validity_bitmap(valid_bool: np.ndarray) -> pa.Buffer:
    """Arrow の仕様 (LSB が行0、1=valid) に従ってビットマップを組み立てる"""
    bits_le = np.packbits(valid_bool.astype(np.uint8), bitorder="little")
    return pa.py_buffer(bits_le.tobytes())


# ----------------------------------------------------------------------
def _build_var_indices(columns: List[ColumnMeta]) -> np.ndarray:
    """
    可変長列 → インデックス配列 (-1 = fixed)
    """
    var_idx = -1
    idxs = np.full(len(columns), -1, dtype=np.int32)
    for i, m in enumerate(columns):
        if m.is_variable:
            var_idx += 1
            idxs[i] = var_idx
    return idxs


# ----------------------------------------------------------------------
def decode_chunk(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # uint8[:]
    field_offsets_dev,  # int32[:, :]
    field_lengths_dev,  # int32[:, :]
    columns: List[ColumnMeta],
) -> pa.RecordBatch:
    """
    GPU メモリ上の COPY バイナリ解析結果を 2‑pass で Arrow RecordBatch へ変換
    """
    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    # ----------------------------------
    # 1. GPU バッファ確保 (Arrow出力用)
    # ----------------------------------
    gmm = GPUMemoryManagerV2()
    bufs: Dict[str, Any] = gmm.initialize_device_buffers(columns, rows)

    # varlen_meta の準備 (Pass 2 で使用) - NUMERIC(DECIMAL128)は固定長なので除外
    varlen_meta = []  # (col_idx, var_idx, name)
    fixedlen_meta = [] # (col_idx, name)
    for cidx, col in enumerate(columns):
        # Check arrow_id for variable length (UTF8, BINARY)
        if col.arrow_id == UTF8 or col.arrow_id == BINARY:
             varlen_meta.append((cidx, len(varlen_meta), col.name))
        else: # Fixed length including DECIMAL128
             fixedlen_meta.append((cidx, col.name))


    # ----------------------------------
    # 2. pass‑1 len/null (GPU Kernel)
    # ----------------------------------
    print("\n--- Running Pass 1 (len/null collection) on GPU ---")
    var_indices_host = _build_var_indices(columns) # Still need this mapping
    var_indices_dev = cuda.to_device(var_indices_host)
    n_var = len(varlen_meta)

    # Allocate output arrays on GPU
    # Note: pass1_len_null expects d_nulls shape (rows, ncols)
    # and d_var_lens shape (n_var, rows)
    d_nulls_all = cuda.device_array((rows, ncols), dtype=np.uint8)
    d_var_lens = cuda.device_array((n_var, rows), dtype=np.int32)

    # Calculate grid/block dimensions for Pass 1 kernel
    threads_pass1 = 256 # Or use a configurable value
    blocks_pass1 = (rows + threads_pass1 - 1) // threads_pass1

    # Launch Pass 1 kernel
    pass1_len_null[blocks_pass1, threads_pass1](
        field_lengths_dev, # Input: lengths calculated by Pass 0
        var_indices_dev,   # Input: mapping from col index to varlen index
        d_var_lens,        # Output: lengths for varlen columns
        d_nulls_all        # Output: null bitmap (Arrow format: 0=NULL, 1=Valid)
    )
    cuda.synchronize()
    print("--- Finished Pass 1 (GPU) ---")

    # --- DEBUG: Check Pass 1 Output (copy from GPU) ---
    print("\n--- After Pass 1 (GPU) ---")
    host_nulls_all = d_nulls_all.copy_to_host() # Copy result for printing
    print(f"d_nulls_all (first 3 rows, 5 cols):\n{host_nulls_all[:min(3, rows), :min(5, ncols)]}")
    if n_var > 0:
        host_var_lens = d_var_lens.copy_to_host() # Copy result for printing
        # d_var_lens is (n_var, rows), print first 3 columns (rows) for first 5 var columns
        print(f"d_var_lens (first 5 var cols, 3 rows):\n{host_var_lens[:min(5, n_var), :min(3, rows)]}")
    # --- END DEBUG ---

    # ----------------------------------
    # 3. prefix‑sum offsets (GPU - CuPy)
    # ----------------------------------
    print("--- Running Prefix Sum (GPU - CuPy) ---")
    d_offsets = cuda.device_array((n_var, rows + 1), dtype=np.int32)
    total_bytes = []

    # デバッグ: 可変長レングスを表示 (CPU計算結果)
    for v in range(n_var):
        print(f"Var column {v} lens = {host_var_lens[v, :5]}") # Print first 5 lengths

    for v in range(n_var):
        cp_len = cp.asarray(d_var_lens[v]) 
        cp_off = cp.concatenate([cp.zeros(1, dtype=np.int32), cp.cumsum(cp_len, dtype=np.int32)])
        d_offsets[v] = cp_off
        total_bytes.append(int(cp_off[-1].get()))
    print("--- Finished Prefix Sum ---")


    # 可変長 values バッファを調整 (max_len ではなく実サイズ)
    values_dev = []
    for (cidx, v_idx, name), tbytes in zip(varlen_meta, total_bytes):
        # NUMERIC(1700) は ASCII 文字列化で最大 24B 程度になるため余裕を持たせる
        col = columns[cidx]
        if col.pg_oid == 1700:
            tbytes = max(tbytes, rows * 24)
        # 再確保して置換
        new_buf = cuda.device_array(max(1, tbytes), dtype=np.uint8)
        old_tuple = bufs[name]
        bufs[name] = (new_buf, old_tuple[1], old_tuple[2] if len(old_tuple) > 2 else 0)
        values_dev.append(new_buf)

    # ----------------------------------
    # 4. pass-2 scatter-copy per var-col (GPU Kernel)
    # ----------------------------------
    print("--- Running Pass 2 VarLen (GPU Kernel) ---")
    threads = 256
    blocks = (rows + threads - 1) // threads

    for i, (cidx, v_idx, name) in enumerate(varlen_meta):
        # Ensure the column is actually variable length (UTF8/BINARY) before calling varlen kernel
        col_meta = columns[cidx]
        if col_meta.arrow_id == UTF8 or col_meta.arrow_id == BINARY:
            offsets_v = d_offsets[v_idx]
            field_off_v = field_offsets_dev[:, cidx]
            field_len_v = field_lengths_dev[:, cidx]
            # numeric_mode is no longer needed as NUMERIC is handled as fixed DECIMAL128
            pass2_scatter_varlen[blocks, threads](
                raw_dev,
                field_off_v,
                field_len_v,
                offsets_v,
                values_dev[v_idx], # Use the reallocated buffer
                0 # numeric_mode = 0 (normal copy)
            )
        else:
            # This case should not happen if varlen_meta is built correctly
            print(f"Warning: Column {name} in varlen_meta but is not UTF8/BINARY (arrow_id={col_meta.arrow_id}). Skipping varlen pass.")

    cuda.synchronize()
    print("--- Finished Pass 2 VarLen ---")


    # ----------------------------------
    # 4.5 pass-2 scatter-copy for fixed-length cols (GPU Kernel)
    # ----------------------------------
    print("--- Running Pass 2 FixedLen (GPU Kernel) ---")
    # Iterate through fixedlen_meta instead of all columns
    for cidx, name in fixedlen_meta:
        col = columns[cidx] # Get the full ColumnMeta
        # fixed-length: includes INTs, FLOATs, BOOL, DATE, TS, and now DECIMAL128
        d_vals, d_nulls_col, stride = bufs[name]

        # Check for DECIMAL128 and call the specific kernel
        if col.arrow_id == DECIMAL128:
            print(f"Running Pass 2 kernel for DECIMAL128 column {name}")
            pass2_scatter_decimal128[blocks, threads](
                raw_dev,
                field_offsets_dev[:, cidx], # Offsets for this column
                field_lengths_dev[:, cidx], # Lengths (needed for signature, maybe useful for validation inside kernel)
                d_vals,                     # Output buffer for this column
                stride                      # Should be 16 for Decimal128
            )
        else:
            # Call the existing generic fixed-length kernel for other types
            pass2_scatter_fixed[blocks, threads](
                raw_dev,
                field_offsets_dev[:, cidx],
                col.elem_size,
                d_vals,
                stride
            )
    cuda.synchronize()
    print("--- Finished Pass 2 FixedLen ---")

    # --- DEBUG: Check fixed-length buffer content after kernel ---
    try:
        target_col_name = "lo_linenumber" # Example: Check this column
        if target_col_name in bufs:
            d_vals_check, _, stride_check = bufs[target_col_name]
            print(f"\n--- DEBUG: Content of '{target_col_name}' buffer after pass2_fixed (first {min(rows, 5)} rows) ---")
            # Copy the relevant part of the buffer to host
            bytes_to_copy = min(rows, 5) * stride_check
            host_vals_check = d_vals_check[:bytes_to_copy].copy_to_host()
            # Print raw bytes and interpreted values (assuming int32 for lo_linenumber)
            for r in range(min(rows, 5)):
                offset = r * stride_check
                raw_bytes = host_vals_check[offset : offset + stride_check]
                # Interpret as little-endian int32 (adjust type if needed)
                try:
                    # Read only the first 4 bytes as int32 (elem_size for lo_linenumber is 4)
                    interpreted_val = np.frombuffer(raw_bytes[:4], dtype=np.int32)[0]
                except IndexError:
                    interpreted_val = "Error reading bytes"
                hex_str = " ".join(f"{b:02x}" for b in raw_bytes)
                print(f"Row {r}: Bytes=[{hex_str}] Interpreted(int32)={interpreted_val}")
        else:
            print(f"\n--- DEBUG: Column '{target_col_name}' not found in bufs for checking ---")
    except Exception as e:
        print(f"\n--- DEBUG: Error during buffer check: {e} ---")
    # --- END DEBUG ---


    # ----------------------------------
    # 5. Arrow RecordBatch 組立 (CPU)
    # ----------------------------------
    print("--- Assembling Arrow RecordBatch (CPU) ---")
    arrays = []
    for cidx, col in enumerate(columns):
        # Get the null mask (boolean array: True=valid, False=null)
        null_np_col = np.ascontiguousarray(host_nulls_all[:, cidx])
        boolean_mask = (null_np_col == 1)

        # --- Get data from GPU buffers to host numpy arrays ---
        values_np = None
        offsets_np = None # Only for varlen

        if col.is_variable:
            v_idx = var_indices_host[cidx]
            # Ensure values_dev has data for this index
            if v_idx < len(values_dev):
                values_np = values_dev[v_idx].copy_to_host()
                offsets_np = d_offsets[v_idx].copy_to_host()
            else:
                print(f"Warning: No data buffer found for varlen column {col.name} (v_idx={v_idx})")
                # Handle error or create empty array? For now, skip.
                continue
        else: # Fixed-width
            if col.name in bufs:
                triple = bufs[col.name]
                d_values = triple[0]
                values_np = d_values.copy_to_host()
            else:
                print(f"Warning: No data buffer found for fixed column {col.name}")
                # Handle error or create empty array? For now, skip.
                continue

        # --- Determine Arrow type for ALL columns based on ColumnMeta.arrow_id ---
        pa_type = None
        if col.arrow_id == DECIMAL128:
            precision, scale = col.arrow_param or (38, 0) # Get precision/scale from meta_fetch result
            # --- DEBUG PRINT START ---
            print(f"DEBUG: Column='{col.name}', arrow_param={col.arrow_param}, Calculated Precision={precision}, Calculated Scale={scale}")
            # --- DEBUG PRINT END ---
            # Validate precision before calling pa.decimal128
            if not (1 <= precision <= 38):
                 print(f"ERROR: Invalid precision {precision} for column {col.name}. Defaulting to (38, 0) for type creation, but data might be incorrect.")
                 # Fallback or raise error? For now, try default for type creation
                 precision, scale = 38, 0 # Use valid default for type creation
            pa_type = pa.decimal128(precision, scale)
        elif col.arrow_id == UTF8:
             pa_type = pa.string()
        elif col.arrow_id == BINARY:
             pa_type = pa.binary()
        elif col.arrow_id == INT16: pa_type = pa.int16()
        elif col.arrow_id == INT32: pa_type = pa.int32()
        elif col.arrow_id == INT64: pa_type = pa.int64()
        elif col.arrow_id == FLOAT32: pa_type = pa.float32()
        elif col.arrow_id == FLOAT64: pa_type = pa.float64()
        elif col.arrow_id == BOOL: pa_type = pa.bool_()
        elif col.arrow_id == DATE32: pa_type = pa.date32()
        elif col.arrow_id == TS64_US: pa_type = pa.timestamp('us')
        else: # Includes UNKNOWN
            print(f"Warning: Unhandled arrow_id {col.arrow_id} for column {col.name}. Falling back to binary.")
            pa_type = pa.binary() # Safer fallback

        # --- Create Arrow Array using pa.array() ---
        # Initialize arr to None before the try block, ensure it's always assigned
        arr = None
        try:
            if values_np is None:
                print(f"Skipping array creation for {col.name} due to missing data.")
                # Create a null array of the determined type if data is missing
                if pa_type:
                    arr = pa.nulls(rows, type=pa_type)
                else:
                    # Fallback if pa_type couldn't be determined (should not happen with current logic)
                    arr = pa.nulls(rows)

            elif pa.types.is_binary(pa_type) or pa.types.is_string(pa_type):
                if offsets_np is None:
                    print(f"Skipping array creation for {col.name} due to missing offsets.")
                    continue
                # Reconstruct list of Python bytes/strings from buffers
                list_data = []
                for r in range(rows):
                    if boolean_mask[r]: # If valid
                        start = offsets_np[r]
                        end = offsets_np[r+1]
                        byte_slice = values_np[start:end]
                        if pa.types.is_string(pa_type):
                            try:
                                list_data.append(bytes(byte_slice).decode('utf-8'))
                            except UnicodeDecodeError:
                                print(f"Warning: UTF-8 decode error in {col.name} row {r}. Appending None.")
                                list_data.append(None)
                                boolean_mask[r] = False # Mark as null if decode fails
                        else:
                            list_data.append(bytes(byte_slice))
                    else: # If null
                        list_data.append(None)
                 # Create array from list
                arr = pa.array(list_data, type=pa_type, mask=~boolean_mask) # Mask is True for NULL
            elif pa.types.is_fixed_size_binary(pa_type):
                # Need item size for fixed binary
                item_size = pa_type.byte_width
                # Reshape or view needed? values_np is currently uint8 flat buffer
                # Example: Assuming data is tightly packed
                list_data = [bytes(values_np[r*item_size:(r+1)*item_size]) if boolean_mask[r] else None for r in range(rows)]
                arr = pa.array(list_data, type=pa_type, mask=~boolean_mask)
            elif pa.types.is_boolean(pa_type):
                # Assuming boolean is stored as 1 byte (0 or 1)
                # Need to view the uint8 buffer as bool
                arr = pa.array(values_np.view(np.bool_), type=pa_type, mask=~boolean_mask)
            elif pa.types.is_decimal(pa_type):
                # Handle Decimal128 - needs buffer view/cast
                item_size = 16 # bytes
                if values_np.size != rows * item_size:
                    raise ValueError(f"Buffer size {values_np.size} incorrect for Decimal128 column {col.name} ({rows} * {item_size})")
                # Create buffer for Arrow - assumes values_np contains raw 16-byte LE integers
                arrow_data_buf = pa.py_buffer(values_np)

                # どのビルドでも確実に動く自前実装を使う
                validity_buf = build_validity_bitmap(boolean_mask) 

                arr = pa.Decimal128Array.from_buffers(pa_type, rows, [validity_buf, arrow_data_buf], null_count=rows - np.count_nonzero(boolean_mask))

            elif pa.types.is_date(pa_type) or pa.types.is_timestamp(pa_type) or pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
                # For primitive types, view the buffer with the correct numpy dtype
                try:
                    # Determine numpy dtype directly from pyarrow type without pandas
                    if pa.types.is_int16(pa_type): np_dtype = np.int16
                    elif pa.types.is_int32(pa_type): np_dtype = np.int32
                    elif pa.types.is_int64(pa_type): np_dtype = np.int64
                    elif pa.types.is_float32(pa_type): np_dtype = np.float32
                    elif pa.types.is_float64(pa_type): np_dtype = np.float64
                    elif pa.types.is_date32(pa_type): np_dtype = np.int32 # date32 is days since epoch (int32)
                    elif pa.types.is_timestamp(pa_type): np_dtype = np.int64 # timestamp is typically int64
                    else:
                        raise TypeError(f"Cannot determine numpy dtype for Arrow type {pa_type}")

                    # Ensure buffer size is compatible with dtype itemsize
                    item_size = np.dtype(np_dtype).itemsize
                    if values_np.size % item_size != 0:
                        raise ValueError(f"Buffer size {values_np.size} not divisible by itemsize {item_size} for {col.name}")
                    # Create array using view and mask
                    # Check if stride was used during allocation
                    _d_vals, _d_nulls, stride_alloc = bufs[col.name] # Get stride from bufs
                    if stride_alloc == item_size: # Tightly packed
                        arr = pa.array(values_np.view(np_dtype), type=pa_type, mask=~boolean_mask)
                    else: # Need to gather elements
                        print(f"Warning: Gathering elements for {col.name} due to stride {stride_alloc} != itemsize {item_size}")
                        gathered_data = np.empty(rows, dtype=np_dtype)
                        for r in range(rows):
                            if boolean_mask[r]:
                                start_byte = r * stride_alloc
                                # Ensure we don't read past the end of values_np
                                if start_byte + item_size <= values_np.size:
                                     gathered_data[r] = np.frombuffer(values_np[start_byte:start_byte+item_size], dtype=np_dtype)[0]
                                else:
                                    print(f"Error: Out-of-bounds read attempted for {col.name} row {r}")
                                    # Handle error, maybe set to null?
                                    boolean_mask[r] = False # Mark as null
                            # else: let numpy handle default for masked elements
                        arr = pa.array(gathered_data, type=pa_type, mask=~boolean_mask)
                except (TypeError, ValueError, AttributeError) as e_view:
                    print(f"Error creating primitive array for {col.name} (type {pa_type}): {e_view}. Buffer content: {values_np[:20]}")
                    arr = pa.nulls(rows, type=pa_type) # Fallback to null array
            else:
                print(f"Unhandled Arrow type {pa_type} for column {col.name}. Creating null array.")
                arr = pa.nulls(rows, type=pa_type) # Fallback for unhandled types

        except Exception as e_create:
            print(f"Error creating Arrow array for column {col.name}: {e_create}")
            # Assign a null array as fallback if arr wasn't created successfully
            if arr is None:
                try:
                    arr = pa.nulls(rows, type=pa_type if pa_type else pa.null())
                except Exception as e_fallback: # Failsafe if pa_type itself is bad
                    print(f"Error creating fallback null array for {col.name}: {e_fallback}")
                    arr = pa.nulls(rows) # Absolute fallback

        # Append exactly **once** and skip the duplicated builder below
        if arr is None:
            arr = pa.nulls(rows, type=pa_type)  
        arrays.append(arr)
        continue     # ← これで以降の重複ブロックを飛ばす

    batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    print("--- Finished Arrow Assembly ---")
    return batch
__all__ = ["decode_chunk"]
