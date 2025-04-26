"""
gpu_decoder_v2.py
=================
GPUMemoryManagerV2 を用いて

  1. pass‑1 (len/null 収集) - CPUで実行
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
try:
    import pyarrow.cuda as pacuda
except ImportError:
    pacuda = None

from numba import cuda

from .type_map import (
    ColumnMeta,
    UTF8,
    BINARY,
)
from .gpu_memory_manager_v2 import GPUMemoryManagerV2

# Pass1 Kernel is removed as it's done on CPU now
# from .cuda_kernels.arrow_gpu_pass1 import pass1_len_null
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.arrow_gpu_pass2_fixed import pass2_scatter_fixed
from .cuda_kernels.numeric_utils import int64_to_decimal_ascii  # noqa: F401  (import for Numba registration)


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
    (Pass 1 は CPU で実行)
    """
    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    # ----------------------------------
    # 1. GPU バッファ確保 (Arrow出力用)
    # ----------------------------------
    gmm = GPUMemoryManagerV2()
    bufs: Dict[str, Any] = gmm.initialize_device_buffers(columns, rows)

    # varlen_meta の準備 (Pass 2 で使用)
    varlen_meta = []  # (col_idx, var_idx, name)
    for cidx, col in enumerate(columns):
        if col.is_variable:
            varlen_meta.append((cidx, len(varlen_meta), col.name))

    # ----------------------------------
    # 2. pass‑1 len/null (CPUで実行)
    # ----------------------------------
    print("\n--- Running Pass 1 (len/null collection) on CPU ---")
    field_lengths_host = field_lengths_dev.copy_to_host() # Get lengths from device
    var_indices_host = _build_var_indices(columns)
    n_var = len(varlen_meta)

    # Allocate host arrays for results
    host_nulls_all = np.zeros((rows, ncols), dtype=np.uint8)
    host_var_lens = np.zeros((n_var, rows), dtype=np.int32)

    for r in range(rows):
        for c in range(ncols):
            flen = field_lengths_host[r, c]
            is_null = (flen == -1)

            # Populate null bitmap (Arrow standard: 1=valid, 0=null)
            host_nulls_all[r, c] = 0 if is_null else 1

            # Populate variable lengths array
            v_idx = var_indices_host[c]
            if v_idx != -1:
                host_var_lens[v_idx, r] = 0 if is_null else flen

    # Copy results to device for subsequent steps
    d_nulls_all = cuda.to_device(host_nulls_all)
    d_var_lens = cuda.to_device(host_var_lens)
    print("--- Finished Pass 1 (CPU) ---")

    # --- DEBUG: Check Pass 1 Output ---
    print("\n--- After Pass 1 (CPU) ---")
    # Access with [row, col] indexing for printing
    print(f"d_nulls_all (first 3 rows, 5 cols):\n{host_nulls_all[:min(3, rows), :min(5, ncols)]}")
    if n_var > 0:
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
        cp_len = cp.asarray(d_var_lens[v]) # Use device array directly with CuPy
        # Check if all lengths are zero (potentially due to all NULLs or empty strings)
        # Note: The test data filling logic might need adjustment if truly empty strings are possible
        # if cp.all(cp_len == 0):
        #     print(f"Fixing empty var column {v} with test data") # Keep this for now
        #     test_data = cp.ones(rows, dtype=cp.int32) * 8
        #     cp_off = cp.concatenate([cp.zeros(1, dtype=cp.int32), cp.cumsum(test_data, dtype=cp.int32)])
        #     d_offsets[v] = cp_off
        #     total_bytes.append(int(cp_off[-1].get()))
        # else:
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
        # Ensure buffer size is at least 1 byte even if tbytes is 0
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
    # use_test_data = [] # Test data logic removed for now
    # for v in range(n_var):
    #     cp_len = cp.asarray(d_var_lens[v])
    #     use_test_data.append(cp.all(cp_len == 0))

    for i, (cidx, v_idx, name) in enumerate(varlen_meta):
        offsets_v = d_offsets[v_idx]
        field_off_v = field_offsets_dev[:, cidx]
        field_len_v = field_lengths_dev[:, cidx]
        numeric_mode = 1 if columns[cidx].pg_oid == 1700 else 0

        # if use_test_data[v_idx]: # Test data logic removed
        #     # ...
        # else:
        # 通常のスキャッタカーネル
        pass2_scatter_varlen[blocks, threads](
            raw_dev,
            field_off_v,
            field_len_v,
            offsets_v,
            values_dev[v_idx], # Use the reallocated buffer
            numeric_mode,
        )
    cuda.synchronize()
    print("--- Finished Pass 2 VarLen ---")


    # ----------------------------------
    # 4.5 pass-2 scatter-copy for fixed-length cols (GPU Kernel)
    # ----------------------------------
    print("--- Running Pass 2 FixedLen (GPU Kernel) ---")
    for cidx, col in enumerate(columns):
        # fixed-length: non-variable or numeric
        if not col.is_variable:
            d_vals, d_nulls_col, stride = bufs[col.name]  # Use d_nulls_col from bufs for fixed
            pass2_scatter_fixed[blocks, threads](
                raw_dev,
                field_offsets_dev[:, cidx],
                col.elem_size,
                d_vals,
                stride  # バッファ確保時のstrideを使用
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

        # --- Determine Arrow type ---
        pa_type = None
        if col.pg_oid == 1700: # NUMERIC or DECIMAL128
            # Handle NUMERIC - currently treated as UTF8 string in pass2 kernel
            # or as DECIMAL128 if fixed
            if col.is_variable: # Treated as UTF8 string
                 pa_type = pa.string()
                 # Need to reconstruct strings from values_np and offsets_np
                 # This requires careful handling of the raw bytes potentially produced by pass2
                 # For simplicity in debugging, let's treat as binary for now
                 pa_type = pa.binary()
            else: # Treated as DECIMAL128
                 precision, scale = col.arrow_param or (38, 0)
                 pa_type = pa.decimal128(precision, scale)
                 # Need to ensure values_np can be interpreted as decimal128
                 # The buffer contains raw bytes, need conversion or view
                 # Example: values_np = values_np.view(np.int64).reshape(-1, 2) # If stored as pairs of int64
                 # This part needs alignment with how pass2_scatter_fixed writes decimals
                 # For now, let's skip decimal assembly to simplify
                 print(f"Skipping assembly for DECIMAL column {col.name}")
                 continue # Skip this column for now
        elif col.is_variable: # Other varlen (UTF8/BINARY)
             pa_type = pa.binary()
             if col.arrow_id == UTF8:
                 pa_type = pa.string()
        else: # Other fixed-width
             if col.pg_oid == 21: pa_type = pa.int16()
             elif col.pg_oid == 23: pa_type = pa.int32()
             elif col.pg_oid == 20: pa_type = pa.int64()
             elif col.pg_oid == 700: pa_type = pa.float32()
             elif col.pg_oid == 701: pa_type = pa.float64()
             elif col.pg_oid == 16: pa_type = pa.bool_()
             elif col.pg_oid == 1082: pa_type = pa.date32()
             elif col.pg_oid in (1114, 1184): pa_type = pa.timestamp('us')
             else:
                 print(f"Warning: Unhandled fixed-width pg_oid {col.pg_oid} for column {col.name}. Falling back to binary.")
                 pa_type = pa.binary() # Safer fallback

        # --- Create Arrow Array using pa.array() ---
        try:
            if values_np is None:
                 print(f"Skipping array creation for {col.name} due to missing data.")
                 continue

            if pa.types.is_binary(pa_type) or pa.types.is_string(pa_type):
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
                 arr = pa.array(values_np.view(np.bool_), type=pa_type, mask=~boolean_mask)
            elif pa.types.is_date(pa_type) or pa.types.is_timestamp(pa_type) or pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
                 # For primitive types, view the buffer with the correct numpy dtype
                 try:
                      np_dtype = pa_type.to_pandas_dtype()
                      # Ensure buffer size is compatible with dtype itemsize
                      item_size = np.dtype(np_dtype).itemsize
                      if values_np.size % item_size != 0:
                           raise ValueError(f"Buffer size {values_np.size} not divisible by itemsize {item_size} for {col.name}")
                      # Create array using view and mask
                      # Need to handle potential size mismatch if stride > item_size
                      if values_np.size == rows * item_size: # Tightly packed
                           arr = pa.array(values_np.view(np_dtype), type=pa_type, mask=~boolean_mask)
                      else: # Need to extract data based on stride (assuming stride was used in allocation)
                           # This case indicates GPUMemoryManagerV2 allocated with stride > itemsize
                           # We need the stride value used during allocation
                           _d_vals, _d_nulls, stride_alloc = bufs[col.name] # Get stride from bufs
                           if stride_alloc == item_size: # Should be tightly packed
                                arr = pa.array(values_np.view(np_dtype), type=pa_type, mask=~boolean_mask)
                           else: # Need to gather elements
                                print(f"Warning: Gathering elements for {col.name} due to stride {stride_alloc} != itemsize {item_size}")
                                gathered_data = np.empty(rows, dtype=np_dtype)
                                for r in range(rows):
                                     if boolean_mask[r]:
                                          start_byte = r * stride_alloc
                                          gathered_data[r] = np.frombuffer(values_np[start_byte:start_byte+item_size], dtype=np_dtype)[0]
                                     # else: let numpy handle default for masked elements
                                arr = pa.array(gathered_data, type=pa_type, mask=~boolean_mask)

                 except (TypeError, ValueError, AttributeError) as e_view:
                      print(f"Error creating primitive array for {col.name} (type {pa_type}): {e_view}. Buffer content: {values_np[:20]}")
                      arr = pa.nulls(rows, type=pa_type) # Fallback to null array
            else:
                 print(f"Unhandled Arrow type {pa_type} for column {col.name}. Creating null array.")
                 arr = pa.nulls(rows, type=pa_type) # Fallback for unhandled types

            arrays.append(arr)
        except Exception as e_create:
            print(f"Error creating Arrow array for column {col.name}: {e_create}")
            # Append a null array as fallback
            try:
                 arrays.append(pa.nulls(rows, type=pa_type if pa_type else pa.null()))
            except: # Failsafe if pa_type itself is bad
                 arrays.append(pa.nulls(rows))


    # --- Original from_buffers logic (commented out) ---
    # for cidx, col in enumerate(columns):
    #     # Get the correct null buffer for this column (shape=(rows,))
    #     # Use the host copy calculated in CPU Pass 1
    #     # Ensure the sliced array is contiguous
    #     null_np_col = np.ascontiguousarray(host_nulls_all[:, cidx])
    #
    #     if pacuda:
    #          # This path might need review if pacuda is enabled, as null_buf creation differs
    #          print("WARNING: pacuda path in assembly needs review for null buffer handling.")
    #          null_buf = pacuda.CudaBuffer.from_data(null_np_col) # Create from host copy
    #
    #          if col.is_variable:
    #          # ... (original pacuda varlen logic) ...
    #          pass
    #     else:
    #         # pacuda がなければホストへコピーして通常の Buffer 使用
    #         null_buf = pa.py_buffer(null_np_col) # Use host copy
    #
    #         if col.is_variable:
    #             v_idx = var_indices_host[cidx]
    #             # デバイスから転送
    #             values_np = values_dev[v_idx].copy_to_host()
    #             offsets_np = d_offsets[v_idx].copy_to_host()
    #
    #             # PyArrow バッファへ
    #             values_buf = pa.py_buffer(values_np)
    #             offsets_buf = pa.py_buffer(offsets_np)
    #
    #             if col.pg_oid == 1700:  # NUMERIC
    #                 # ... (original numeric handling) ...
    #                 pass
    #             else:
    #                 print(f"Host Column {col.name}: NULL bitmap = {null_np_col[:5]}")
    #                 print(f"Host Column {col.name}: offsets = {offsets_np[:6]}")
    #                 null_idx = np.where(null_np_col == 0)[0]
    #                 if len(null_idx) > 0: print(f"Host Column {col.name}: NULL rows = {null_idx[:5]}")
    #
    #                 pa_arr = pa.Array.from_buffers(
    #                     pa.binary(), rows, [null_buf, offsets_buf, values_buf]
    #                 )
    #                 if col.arrow_id == UTF8:
    #                     pa_arr = pa_arr.cast(pa.string())
    #                 arrays.append(pa_arr)
    #         else: # Fixed-width
    #             triple = bufs[col.name]
    #             d_values = triple[0]
    #             values_np = d_values.copy_to_host()
    #             values_buf = pa.py_buffer(values_np)
    #
    #             if col.pg_oid == 1700:  # DECIMAL128
    #                 # ... (original decimal handling) ...
    #                 pass
    #             else: # Other fixed-width types based on pg_oid
    #                 if col.pg_oid == 21: # int2
    #                     pa_type = pa.int16()
    #                 elif col.pg_oid == 23: # int4
    #                     pa_type = pa.int32()
    #                 elif col.pg_oid == 20: # int8
    #                     pa_type = pa.int64()
    #                 elif col.pg_oid == 700: # float4
    #                     pa_type = pa.float32()
    #                 elif col.pg_oid == 701: # float8
    #                     pa_type = pa.float64()
    #                 elif col.pg_oid == 16: # bool
    #                     pa_type = pa.bool_()
    #                 elif col.pg_oid == 1082: # date
    #                     pa_type = pa.date32()
    #                 elif col.pg_oid in (1114, 1184): # timestamp, timestamptz
    #                     pa_type = pa.timestamp('us')
    #                 else:
    #                     # Fallback or raise error for unhandled fixed-width types
    #                     print(f"Warning: Unhandled fixed-width pg_oid {col.pg_oid} for column {col.name}. Falling back to int64.")
    #                     pa_type = pa.int64() # Default fallback, might be incorrect
    #
    #             # This append should be outside the inner if/else for pg_oid, but inside the outer fixed-width else block
    #             arrays.append(
    #                 pa.Array.from_buffers(pa_type, rows, [null_buf, values_buf])
    #             )

    batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    print("--- Finished Arrow Assembly ---")
    return batch


__all__ = ["decode_chunk"]
