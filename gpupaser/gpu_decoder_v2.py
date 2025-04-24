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
    # d_nulls_all = cuda.to_device(host_nulls_all) # Keep nulls on host for now
    d_var_lens = cuda.to_device(host_var_lens)
    print("--- Finished Pass 1 (CPU) ---")

    # --- DEBUG: Check Pass 1 Output ---
    print("\n--- After Pass 1 (CPU) ---")
    # Access with [row, col] indexing for printing
    print(f"host_nulls_all (first 3 rows, 5 cols):\n{host_nulls_all[:min(3, rows), :min(5, ncols)]}")
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
        cp_off = cp.concatenate([cp.zeros(1, dtype=np.int32), cp.cumsum(cp_len, dtype=np.int32)])
        d_offsets[v] = cp_off
        total_bytes.append(int(cp_off[-1].get()))
    print("--- Finished Prefix Sum ---")


    # 可変長 values バッファを調整 (max_len ではなく実サイズ)
    values_dev = []
    for (cidx, v_idx, name), tbytes in zip(varlen_meta, total_bytes):
        col = columns[cidx]
        if col.pg_oid == 1700:
            tbytes = max(tbytes, rows * 24)
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
        offsets_v = d_offsets[v_idx]
        field_off_v = field_offsets_dev[:, cidx]
        field_len_v = field_lengths_dev[:, cidx]
        numeric_mode = 1 if columns[cidx].pg_oid == 1700 else 0

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
        if not col.is_variable:
            d_vals, d_nulls_col, stride = bufs[col.name]
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
        target_col_name = "lo_linenumber"
        if target_col_name in bufs:
            d_vals_check, _, stride_check = bufs[target_col_name]
            print(f"\n--- DEBUG: Content of '{target_col_name}' buffer after pass2_fixed (first {min(rows, 5)} rows) ---")
            bytes_to_copy = min(rows, 5) * stride_check
            host_vals_check = d_vals_check[:bytes_to_copy].copy_to_host()
            for r in range(min(rows, 5)):
                offset = r * stride_check
                raw_bytes = host_vals_check[offset : offset + stride_check]
                try:
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
    # 5. Arrow RecordBatch 組立 (CPU) - Revised using pa.array()
    # ----------------------------------
    print("--- Assembling Arrow RecordBatch (CPU) ---")
    arrays = []
    for cidx, col in enumerate(columns):
        # Get the null mask (boolean array: True=valid, False=null)
        null_np_col = np.ascontiguousarray(host_nulls_all[:, cidx])
        boolean_mask = (null_np_col == 1) # True where valid

        # --- Get data from GPU buffers to host numpy arrays ---
        values_np = None
        offsets_np = None # Only for varlen

        if col.is_variable:
            v_idx = var_indices_host[cidx]
            if v_idx != -1 and v_idx < len(values_dev):
                 values_np = values_dev[v_idx].copy_to_host()
                 offsets_np = d_offsets[v_idx].copy_to_host()
            else:
                 print(f"Warning: No data buffer found for varlen column {col.name} (v_idx={v_idx})")
                 pass
        else: # Fixed-width
            if col.name in bufs:
                 triple = bufs[col.name]
                 d_values = triple[0]
                 values_np = d_values.copy_to_host()
            else:
                 print(f"Warning: No data buffer found for fixed column {col.name}")
                 pass

        # --- Determine Arrow type ---
        pa_type = None
        skip_assembly = False
        if col.pg_oid == 1700: # NUMERIC or DECIMAL128
            if col.is_variable:
                 print(f"Warning: NUMERIC column {col.name} treated as varlen, assembling as binary.")
                 pa_type = pa.binary()
            else:
                 precision, scale = col.arrow_param or (38, 0)
                 pa_type = pa.decimal128(precision, scale)
                 print(f"Skipping assembly for fixed DECIMAL column {col.name}")
                 skip_assembly = True
        elif col.is_variable:
             pa_type = pa.binary()
             if col.arrow_id == UTF8:
                 pa_type = pa.string()
        else:
             if col.pg_oid == 21: pa_type = pa.int16()
             elif col.pg_oid == 23: pa_type = pa.int32()
             elif col.pg_oid == 20: pa_type = pa.int64()
             elif col.pg_oid == 700: pa_type = pa.float32()
             elif col.pg_oid == 701: pa_type = pa.float64()
             elif col.pg_oid == 16: pa_type = pa.bool_()
             elif col.pg_oid == 1082: pa_type = pa.date32()
             elif col.pg_oid in (1114, 1184): pa_type = pa.timestamp('us')
             else:
                 print(f"Warning: Unhandled fixed-width pg_oid {col.pg_oid} for column {col.name}. Falling back to binary({col.elem_size}).")
                 try:
                     pa_type = pa.binary(col.elem_size)
                 except:
                     print(f"Error creating binary({col.elem_size}) type for {col.name}. Skipping.")
                     skip_assembly = True

        # --- Create Arrow Array using pa.array() ---
        if skip_assembly:
             try:
                 arrays.append(pa.nulls(rows, type=pa_type if pa_type else pa.null()))
             except:
                 arrays.append(pa.nulls(rows))
             continue

        arr = None
        try:
            if values_np is None:
                 print(f"Creating null array for {col.name} due to missing data.")
                 arr = pa.nulls(rows, type=pa_type if pa_type else pa.null())
            elif pa_type is None:
                 print(f"Cannot determine Arrow type for {col.name}. Creating null array.")
                 arr = pa.nulls(rows)
            elif pa.types.is_binary(pa_type) or pa.types.is_string(pa_type):
                 if offsets_np is None:
                      print(f"Skipping array creation for varlen {col.name} due to missing offsets.")
                      arr = pa.nulls(rows, type=pa_type)
                 else:
                     # Build list of Python objects (bytes or str)
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
                                      boolean_mask[r] = False # Mark as null
                              else: # Binary type
                                  list_data.append(bytes(byte_slice))
                          else: # If null
                              list_data.append(None)
                     # Create Arrow array from the list
                     arr = pa.array(list_data, type=pa_type, mask=~boolean_mask)
            elif pa.types.is_fixed_size_binary(pa_type):
                 item_size = pa_type.byte_width
                 if values_np.size != rows * item_size:
                      print(f"Warning: Buffer size mismatch for fixed_size_binary {col.name}. Expected {rows*item_size}, got {values_np.size}. Creating null array.")
                      arr = pa.nulls(rows, type=pa_type)
                 else:
                      # Build list of Python bytes objects
                      list_data = [bytes(values_np[r*item_size:(r+1)*item_size]) if boolean_mask[r] else None for r in range(rows)]
                      arr = pa.array(list_data, type=pa_type, mask=~boolean_mask)
            elif pa.types.is_boolean(pa_type):
                 if values_np.size != rows:
                      print(f"Warning: Buffer size mismatch for boolean {col.name}. Expected {rows}, got {values_np.size}. Creating null array.")
                      arr = pa.nulls(rows, type=pa_type)
                 else:
                      # Create boolean array directly from numpy view
                      arr = pa.array(values_np.view(np.bool_), type=pa_type, mask=~boolean_mask)
            elif pa.types.is_primitive(pa_type): # Covers integers, floats, dates, timestamps
                 try:
                      np_dtype = pa_type.to_pandas_dtype()
                      item_size = np.dtype(np_dtype).itemsize
                      _d_vals, _d_nulls, stride_alloc = bufs[col.name]

                      if values_np.size == rows * stride_alloc:
                           if stride_alloc == item_size: # Tightly packed
                                # Create array directly from numpy view
                                arr = pa.array(values_np.view(np_dtype), type=pa_type, mask=~boolean_mask)
                           else: # Need to gather elements due to stride
                                print(f"Warning: Gathering elements for {col.name} due to stride {stride_alloc} != itemsize {item_size}")
                                gathered_data = np.empty(rows, dtype=np_dtype)
                                for r in range(rows):
                                     if boolean_mask[r]:
                                          start_byte = r * stride_alloc
                                          if start_byte + item_size <= values_np.size:
                                               gathered_data[r] = np.frombuffer(values_np[start_byte:start_byte+item_size], dtype=np_dtype)[0]
                                          else:
                                               print(f"Error: Out-of-bounds read attempted for {col.name} row {r}")
                                               boolean_mask[r] = False # Mark as null
                                # Create array from the gathered data
                                arr = pa.array(gathered_data, type=pa_type, mask=~boolean_mask)
                      else:
                           raise ValueError(f"Buffer size {values_np.size} does not match expected allocated size {rows * stride_alloc} for {col.name}")

                 except (TypeError, ValueError, AttributeError) as e_view:
                      print(f"Error creating primitive array for {col.name} (type {pa_type}): {e_view}. Buffer content: {values_np[:20]}")
                      arr = pa.nulls(rows, type=pa_type) # Fallback to null array
            else:
                 print(f"Unhandled Arrow type {pa_type} for column {col.name}. Creating null array.")
                 arr = pa.nulls(rows, type=pa_type) # Fallback for unhandled types

            arrays.append(arr)

        except Exception as e_create:
            print(f"Error creating Arrow array for column {col.name}: {e_create}")
            try:
                 arrays.append(pa.nulls(rows, type=pa_type if pa_type else pa.null()))
            except:
                 arrays.append(pa.nulls(rows))

    # --- Create the final RecordBatch ---
    try:
        # Filter out None arrays that might have been added during error handling
        valid_arrays = [a for a in arrays if a is not None]
        valid_names = [columns[i].name for i, a in enumerate(arrays) if a is not None]
        if len(valid_arrays) != len(columns):
             print(f"Warning: Number of valid arrays ({len(valid_arrays)}) does not match number of columns ({len(columns)}).")
             # Attempt to build with valid arrays only, or handle error differently
             # For now, let's try building with what we have, but this indicates issues.

        # Ensure all arrays have the same length
        if valid_arrays:
            expected_length = valid_arrays[0].length
            if not all(a.length == expected_length for a in valid_arrays):
                 print("Error: Not all arrays have the same length for RecordBatch creation.")
                 # Handle error: maybe return empty batch or raise
                 schema = pa.schema([(name, pa.null()) for name in valid_names])
                 batch = pa.RecordBatch.from_arrays([pa.nulls(rows) for _ in valid_arrays], schema=schema)
                 return batch # Return potentially empty/null batch

        batch = pa.RecordBatch.from_arrays(valid_arrays, names=valid_names)
        print("--- Finished Arrow Assembly ---")
    except Exception as e_batch:
        print(f"Error creating final RecordBatch: {e_batch}")
        # Optionally return an empty batch or raise
        schema = pa.schema([(c.name, pa.null()) for c in columns]) # Create dummy schema
        batch = pa.RecordBatch.from_arrays([pa.nulls(rows) for _ in columns], schema=schema)

    return batch


__all__ = ["decode_chunk"]
