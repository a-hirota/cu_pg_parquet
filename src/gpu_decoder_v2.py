"""GPU COPY BINARY → Arrow RecordBatch 2パス変換"""

from __future__ import annotations

from typing import List, Dict, Any

import os
import warnings
import numpy as np
import cupy as cp
import pyarrow as pa
import pyarrow.compute as pc
try:
    import pyarrow.cuda as pa_cuda
    PYARROW_CUDA_AVAILABLE = True
except ImportError:
    pa_cuda = None
    PYARROW_CUDA_AVAILABLE = False


from numba import cuda

from .type_map import *
from .gpu_memory_manager_v2 import GPUMemoryManagerV2

from .cuda_kernels.arrow_gpu_pass1 import pass1_len_null # Use Pass 1 GPU Kernel
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.arrow_gpu_pass2_fixed import pass2_scatter_fixed
from .cuda_kernels.arrow_gpu_pass2_decimal128 import (
    pass2_scatter_decimal128, pass2_scatter_decimal128_optimized, pass2_scatter_decimal64_optimized,
    POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST # ホスト側のテーブル定義をインポート
)
from .cuda_kernels.numeric_utils import int64_to_decimal_ascii  # noqa: F401  (import for Numba registration)

def build_validity_bitmap(valid_bool: np.ndarray) -> pa.Buffer:
    """Arrow validity bitmap (LSB=行0, 1=valid)"""
    if isinstance(valid_bool, cp.ndarray):
        valid_bool = valid_bool.get()
    elif not isinstance(valid_bool, np.ndarray):
        raise TypeError("Input must be a NumPy or CuPy array")

    valid_bool = np.ascontiguousarray(valid_bool, dtype=np.bool_)
    bits_le = np.packbits(valid_bool, bitorder='little')
    return pa.py_buffer(bits_le)


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

    # --- 10のべき乗テーブルをGPUに転送 ---
    d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
    d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)

    # ----------------------------------
    # 1. GPU バッファ確保 (Arrow出力用) - 初期確保
    # ----------------------------------
    gmm = GPUMemoryManagerV2()
    # bufs now contains offset buffers for varlen columns as well
    # varlen: (d_values, d_nulls, d_offsets, max_len)
    # fixed: (d_values, d_nulls, stride)
    bufs: Dict[str, Any] = gmm.initialize_device_buffers(columns, rows)

    # varlen_meta の準備 (Pass 2 で使用) - NUMERIC(DECIMAL128)は固定長なので除外
    varlen_meta = []  # (col_idx, var_idx, name) # var_idx is the index within varlen columns
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

    host_nulls_all = d_nulls_all.copy_to_host()

    # ----------------------------------
    # 3. prefix‑sum offsets (GPU - CuPy) & データバッファ再確保
    # ----------------------------------
    print("--- Running Prefix Sum (GPU - CuPy) & Reallocating Varlen Buffers ---")
    total_bytes_list = [] # Store total bytes for each varlen column
    values_dev_reallocated = [] # Store reallocated data buffers

    # Get the initially allocated offset buffers from gmm
    # Assuming varlen tuple is (d_values, d_nulls, d_offsets, max_len)
    initial_offset_buffers = [bufs[name][2] for _, _, name in varlen_meta]

    for v_idx, (cidx, _, name) in enumerate(varlen_meta):
        print(f"\n--- DEBUG: Processing varlen column '{name}' (v_idx={v_idx}, cidx={cidx}) ---")
        
        # Calculate prefix sum using the lengths from Pass 1
        cp_len = cp.asarray(d_var_lens[v_idx]) # Lengths for this varlen column
        print(f"Lengths from Pass 1 (first 10): {cp_len[:min(10, len(cp_len))].get()}")
        print(f"Lengths stats: min={cp_len.min().get()}, max={cp_len.max().get()}, mean={cp_len.mean().get():.2f}")
        
        # Calculate offsets (including the initial 0)
        cp_off = cp.cumsum(cp_len, dtype=np.int32)
        total_bytes = int(cp_off[-1].get()) if rows > 0 else 0
        total_bytes_list.append(total_bytes)
        
        print(f"Prefix sum offsets (first 10): {cp_off[:min(10, len(cp_off))].get()}")
        print(f"Total bytes calculated: {total_bytes}")

        # Get the pre-allocated offset buffer for this column
        d_offset_col = initial_offset_buffers[v_idx]
        print(f"Offset buffer shape: {d_offset_col.shape}, dtype: {d_offset_col.dtype}")
        
        # Write the calculated offsets into the buffer (need to handle the initial 0)
        print("Writing offsets to GPU buffer...")
        try:
            d_offset_col[0] = 0
            print(f"Set offset[0] = 0")
            
            # Check type compatibility before assignment
            print(f"cp_off type: {type(cp_off)}, dtype: {cp_off.dtype}")
            print(f"d_offset_col[1:] type: {type(d_offset_col[1:])}, dtype: {d_offset_col[1:].dtype}")
            
            # Convert to ensure compatible assignment
            cp_off_as_cupy = cp.asarray(cp_off, dtype=np.int32)
            d_offset_col[1:rows+1] = cp_off_as_cupy  # Ensure we don't write beyond buffer
            
            print(f"Successfully wrote {len(cp_off)} offsets to buffer")
            
            # Verify the written offsets
            written_offsets = d_offset_col[:min(10, rows+1)].copy_to_host()
            print(f"Verified offsets in buffer (first 10): {written_offsets}")
            
        except Exception as e_offset_write:
            print(f"ERROR writing offsets to buffer: {e_offset_write}")
            print(f"Exception type: {type(e_offset_write)}")
            import traceback
            traceback.print_exc()

        # Reallocate the data buffer using the calculated total_bytes
        new_data_buf = gmm.replace_varlen_data_buffer(name, total_bytes)
        values_dev_reallocated.append(new_data_buf)

        # Debug print
        print(f"VarCol '{name}' (v_idx={v_idx}): Total Bytes={total_bytes}, Reallocated Buffer: {new_data_buf.device_ctypes_pointer.value if new_data_buf else 'None'}")
        print(f"--- End processing '{name}' ---")

    print("--- Finished Prefix Sum & Reallocation ---")


    # ----------------------------------
    # 4. pass-2 scatter-copy per var-col (GPU Kernel)
    # ----------------------------------
    print("--- Running Pass 2 VarLen (GPU Kernel) ---")
    threads = 256
    blocks = (rows + threads - 1) // threads

    for v_idx, (cidx, _, name) in enumerate(varlen_meta):
        col_meta = columns[cidx]
        print(f"\n--- DEBUG: Pass 2 VarLen for '{name}' (v_idx={v_idx}, cidx={cidx}) ---")
        
        # Only run for actual variable length types
        if col_meta.arrow_id == UTF8 or col_meta.arrow_id == BINARY:
            # Get the offset buffer (already filled by prefix sum)
            d_offset_v = bufs[name][2] # Get from the updated bufs dict
            # Get the reallocated data buffer
            d_values_v = bufs[name][0] # Get from the updated bufs dict

            # Get field offsets and lengths for this column from the Pass 0 result
            field_off_v = field_offsets_dev[:, cidx]
            field_len_v = field_lengths_dev[:, cidx]

            # DEBUG: Check field information before kernel
            host_field_off = field_off_v[:min(5, rows)].copy_to_host()
            host_field_len = field_len_v[:min(5, rows)].copy_to_host()
            print(f"Field offsets (first 5): {host_field_off}")
            print(f"Field lengths (first 5): {host_field_len}")

            # Call the simplified kernel
            print(f"Launching pass2_scatter_varlen kernel with blocks={blocks}, threads={threads}")
            pass2_scatter_varlen[blocks, threads](
                raw_dev,
                field_off_v,
                field_len_v,
                d_offset_v,    # Pass the offset buffer for this column
                d_values_v     # Pass the reallocated data buffer
            )
            cuda.synchronize()
            print(f"Kernel completed for '{name}'")
            
            # DEBUG: Check data buffer content after kernel
            if d_values_v.size > 0:
                sample_size = min(100, d_values_v.size)
                sample_data = d_values_v[:sample_size].copy_to_host()
                print(f"Data buffer sample (first {sample_size} bytes): {sample_data[:50]}...")
                
                # Try to decode first string using offsets
                host_offsets_post = d_offset_v[:min(5, rows+1)].copy_to_host()
                print(f"Offsets after kernel (first 5): {host_offsets_post}")
                
                if len(host_offsets_post) > 1:
                    try:
                        start = host_offsets_post[0]
                        end = host_offsets_post[1]
                        if 0 <= start < end <= d_values_v.size:
                            first_string_bytes = d_values_v[start:end].copy_to_host()
                            decoded = first_string_bytes.tobytes().decode('utf-8', errors='replace')
                            print(f"First string decoded: '{decoded}'")
                        else:
                            print(f"Invalid first string range: [{start}:{end}] (buffer size: {d_values_v.size})")
                    except Exception as e_decode:
                        print(f"Error decoding first string: {e_decode}")
            else:
                print(f"Data buffer is empty for '{name}'")
            
        else:
            # This case should not happen if varlen_meta is built correctly
             warnings.warn(f"Column {name} in varlen_meta but is not UTF8/BINARY (arrow_id={col_meta.arrow_id}). Skipping varlen pass.")

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

        # Check for DECIMAL128 and call the appropriate kernel
        if col.arrow_id == DECIMAL128:
            # Check if optimization is enabled via environment variable
            use_optimization = os.environ.get("USE_DECIMAL_OPTIMIZATION", "1") == "1"
            
            if use_optimization:
                print(f"Running Pass 2 OPTIMIZED kernel for DECIMAL128 column {name}")
                
                # Extract scale from ColumnMeta.arrow_param
                precision, scale = col.arrow_param or (38, 0)
                target_scale = scale  # Use the column's defined scale
                
                print(f"DECIMAL128 {name}: precision={precision}, scale={scale}")
                
                # Choose optimal kernel based on precision
                if precision <= 18 and stride == 8:
                    # Use Decimal64 optimization for high precision cases
                    print(f"Using Decimal64 optimized kernel for {name} (precision={precision})")
                    pass2_scatter_decimal64_optimized[blocks, threads](
                        raw_dev,
                        field_offsets_dev[:, cidx],
                        field_lengths_dev[:, cidx],
                        d_vals,
                        stride,
                        target_scale,
                        d_pow10_table_lo,
                        d_pow10_table_hi
                    )
                else:
                    # Use Decimal128 optimized kernel
                    print(f"Using Decimal128 optimized kernel for {name} (precision={precision})")
                    pass2_scatter_decimal128_optimized[blocks, threads](
                        raw_dev,
                        field_offsets_dev[:, cidx],
                        field_lengths_dev[:, cidx],
                        d_vals,
                        stride,
                        target_scale,
                        d_pow10_table_lo,
                        d_pow10_table_hi
                    )
            else:
                # Use original kernel for comparison
                print(f"Running Pass 2 ORIGINAL kernel for DECIMAL128 column {name} (optimization disabled)")
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
    # 5. Arrow RecordBatch 組立 (Zero-Copy where possible)
    # ----------------------------------
    print("--- Assembling Arrow RecordBatch (Zero-Copy Attempt) ---")
    arrays = []
    # Get the full null bitmap from GPU once
    # host_nulls_all was already copied for debugging, reuse it.
    # If not debugging, copy here: host_nulls_all = d_nulls_all.copy_to_host()

    for cidx, col in enumerate(columns):
        print(f"Assembling column: {col.name} (Arrow ID: {col.arrow_id}, IsVar: {col.is_variable})")
        # --- 1. Get Validity Buffer ---
        # Get the boolean mask (True=valid) for this column from the host copy
        boolean_mask_np = (host_nulls_all[:, cidx] == 1)
        null_count = rows - np.count_nonzero(boolean_mask_np)
        # Build the Arrow validity bitmap buffer (on host for now)
        # build_validity_bitmap handles contiguous array conversion
        try:
            validity_buffer = build_validity_bitmap(boolean_mask_np)
        except Exception as e_vb:
            print(f"Error building validity bitmap for {col.name}: {e_vb}")
            arrays.append(pa.nulls(rows)) # Fallback
            continue

        # --- 2. Determine Arrow Type ---
        pa_type = None
        if col.arrow_id == DECIMAL128:
            # Precision/scale should be in col.arrow_param from meta_fetch
            precision, scale = col.arrow_param or (38, 0)
            if not (1 <= precision <= 38):
                 warnings.warn(f"Invalid precision {precision} for DECIMAL column {col.name}. Using (38, 0).")
                 precision, scale = 38, 0
            pa_type = pa.decimal128(precision, scale)
        elif col.arrow_id == UTF8: pa_type = pa.string()
        elif col.arrow_id == BINARY: pa_type = pa.binary()
        elif col.arrow_id == INT16: pa_type = pa.int16()
        elif col.arrow_id == INT32: pa_type = pa.int32()
        elif col.arrow_id == INT64: pa_type = pa.int64()
        elif col.arrow_id == FLOAT32: pa_type = pa.float32()
        elif col.arrow_id == FLOAT64: pa_type = pa.float64()
        elif col.arrow_id == BOOL: pa_type = pa.bool_() # Assuming stored as 1 byte in GPU buffer
        elif col.arrow_id == DATE32: pa_type = pa.date32()
        elif col.arrow_id == TS64_US:
            # Handle timezone explicitly using arrow_param if available
            tz_info = col.arrow_param # Expects None or a timezone string
            if tz_info is not None and not isinstance(tz_info, str):
                 warnings.warn(f"Invalid timezone info in arrow_param for {col.name}: {tz_info}. Ignoring.")
                 tz_info = None
            # NOTE: meta_fetch.py currently does not populate arrow_param with timezone.
            # This code assumes it might in the future.
            print(f"Timestamp column {col.name}: using timezone '{tz_info}' from arrow_param.")
            pa_type = pa.timestamp('us', tz=tz_info)
        else: # Includes UNKNOWN
            warnings.warn(f"Unhandled arrow_id {col.arrow_id} for column {col.name}. Falling back to binary.")
            pa_type = pa.binary() # Fallback to binary

        # --- 3. Get Data/Offset Buffers (GPU Pointers) ---
        arr = None
        try:
            if col.is_variable:
                # Get buffers from the potentially updated bufs dict
                # Tuple: (d_values, d_nulls, d_offsets, max_len)
                if col.name not in bufs or len(bufs[col.name]) != 4:
                     raise ValueError(f"Variable length buffer tuple not found or invalid for {col.name}")
                d_values_col = bufs[col.name][0] # Reallocated data buffer
                d_offsets_col = bufs[col.name][2] # Offset buffer

                if d_values_col is None or d_offsets_col is None:
                     raise ValueError(f"Missing data or offset buffer for varlen column {col.name}")

                # DEBUG: Check buffer content before PyArrow assembly
                print(f"\n--- DEBUG: {col.name} varlen buffer analysis ---")
                host_offsets = d_offsets_col.copy_to_host()
                host_values = d_values_col.copy_to_host() if d_values_col.size > 0 else np.array([], dtype=np.uint8)
                print(f"Offsets shape: {host_offsets.shape}, dtype: {host_offsets.dtype}")
                print(f"Values shape: {host_values.shape}, dtype: {host_values.dtype}")
                print(f"First 10 offsets: {host_offsets[:min(10, len(host_offsets))]}")
                print(f"Last 5 offsets: {host_offsets[-5:] if len(host_offsets) >= 5 else host_offsets}")
                print(f"Values size: {len(host_values)} bytes")
                
                # Check if offsets are monotonic and valid
                if len(host_offsets) > 1:
                    offsets_diff = np.diff(host_offsets)
                    print(f"Offset differences (first 10): {offsets_diff[:min(10, len(offsets_diff))]}")
                    print(f"Min offset diff: {np.min(offsets_diff)}, Max: {np.max(offsets_diff)}")
                    if np.any(offsets_diff < 0):
                        print("WARNING: Non-monotonic offsets detected!")
                
                # Check sample string data
                if len(host_values) > 0 and len(host_offsets) > 1:
                    try:
                        # Try to decode first few strings
                        for i in range(min(3, len(host_offsets)-1)):
                            start = host_offsets[i]
                            end = host_offsets[i+1]
                            if 0 <= start <= end <= len(host_values):
                                string_bytes = host_values[start:end]
                                try:
                                    decoded_str = string_bytes.tobytes().decode('utf-8', errors='replace')
                                    print(f"String {i}: [{start}:{end}] = '{decoded_str[:50]}{'...' if len(decoded_str) > 50 else ''}'")
                                except Exception as e_decode:
                                    print(f"String {i}: [{start}:{end}] decode error: {e_decode}")
                            else:
                                print(f"String {i}: Invalid range [{start}:{end}] (values size: {len(host_values)})")
                    except Exception as e_sample:
                        print(f"Error sampling strings: {e_sample}")

                # Wrap GPU buffers for PyArrow
                if PYARROW_CUDA_AVAILABLE:
                    pa_offset_buf = pa_cuda.as_cuda_buffer(d_offsets_col)
                    pa_data_buf = pa_cuda.as_cuda_buffer(d_values_col)
                    print(f"Using CUDA buffers for {col.name}")
                else:
                    # Fallback: Copy to host if pyarrow.cuda is not available
                    warnings.warn("pyarrow.cuda not available. Copying varlen data/offsets to host for Arrow assembly.")
                    pa_offset_buf = pa.py_buffer(host_offsets)
                    pa_data_buf = pa.py_buffer(host_values)
                    print(f"Using host buffers for {col.name}")

                # DEBUG: Check PyArrow buffer properties
                print(f"PyArrow offset buffer: size={pa_offset_buf.size}, address={pa_offset_buf.address}")
                print(f"PyArrow data buffer: size={pa_data_buf.size}, address={pa_data_buf.address}")
                
                # Create array using from_buffers
                # Ensure parameters are correct Python types
                rows_int = int(rows)
                null_count_int = int(null_count)
                print(f"Attempting from_buffers for {col.name}: type={pa_type}, rows={rows_int}, null_count={null_count_int}")
                print(f"Parameter types: rows={type(rows_int)}, null_count={type(null_count_int)}")
                print(f"Buffer types: validity={type(validity_buffer)}, offset={type(pa_offset_buf)}, data={type(pa_data_buf)}")
                
                # DETAILED DEBUGGING: Check PyArrow version and signature
                print(f"PyArrow version: {pa.__version__}")
                import inspect
                try:
                    sig = inspect.signature(pa.StringArray.from_buffers)
                    print(f"StringArray.from_buffers signature: {sig}")
                except:
                    print("Could not inspect StringArray.from_buffers signature")
                
                # DETAILED DEBUGGING: Check buffer contents
                try:
                    offset_bytes = pa_offset_buf.to_pybytes()
                    offset_as_int32 = np.frombuffer(offset_bytes, dtype=np.int32)
                    print(f"Offset buffer raw bytes (first 40): {offset_bytes[:40].hex()}")
                    print(f"Offset as int32 array (first 10): {offset_as_int32[:10]}")
                    print(f"Offset buffer size: {len(offset_bytes)} bytes, expected: {(rows_int + 1) * 4}")
                    
                    data_bytes = pa_data_buf.to_pybytes()
                    print(f"Data buffer raw bytes (first 40): {data_bytes[:40].hex()}")
                    print(f"Data buffer size: {len(data_bytes)} bytes")
                    
                    # System info
                    import sys
                    print(f"System byte order: {sys.byteorder}")
                    print(f"NumPy int32 dtype: {np.dtype(np.int32)}")
                    print(f"Host offsets dtype: {host_offsets.dtype}")
                    print(f"Host offsets flags: {host_offsets.flags}")
                    
                except Exception as e_debug:
                    print(f"Error during buffer inspection: {e_debug}")
                
                # DETAILED DEBUGGING: Test minimal case first
                print("\n--- Testing minimal StringArray.from_buffers ---")
                try:
                    test_data = b"helloworldtest"
                    test_offsets = np.array([0, 5, 10, 14], dtype=np.int32)
                    test_offset_buf = pa.py_buffer(test_offsets)
                    test_data_buf = pa.py_buffer(test_data)
                    test_arr = pa.StringArray.from_buffers(
                        length=3,
                        value_offsets=test_offset_buf,
                        data=test_data_buf,
                        null_bitmap=None
                    )
                    print("Minimal test: SUCCESS")
                except Exception as e_minimal:
                    print(f"Minimal test failed: {e_minimal}")
                    import traceback
                    traceback.print_exc()
                
                # The issue might be that PyArrow expects specific buffer order or None for validity
                # Let's try without the null_count parameter first, and with None for validity if no nulls
                if null_count_int == 0:
                    validity_arg = None
                else:
                    validity_arg = validity_buffer
                
                # DETAILED DEBUGGING: Check each argument separately
                print(f"\n--- Checking from_buffers arguments ---")
                print(f"pa_type: {pa_type}")
                print(f"rows_int: {rows_int} (type: {type(rows_int)})")
                print(f"validity_arg: {validity_arg}")
                print(f"pa_offset_buf: {pa_offset_buf} (size: {pa_offset_buf.size})")
                print(f"pa_data_buf: {pa_data_buf} (size: {pa_data_buf.size})")
                
                try:
                    if pa.types.is_string(pa_type):
                        # Try the standard Arrow StringArray.from_buffers signature                        
                        arr = pa.StringArray.from_buffers(
                            length=rows_int,
                            value_offsets=pa_offset_buf,
                            data=pa_data_buf,
                            null_bitmap=validity_arg,
                            null_count=null_count_int
                        )
                        print(f"SUCCESS: StringArray created for {col.name}")
                    elif pa.types.is_binary(pa_type):
                        arr = pa.BinaryArray.from_buffers(
                            length=rows_int,
                            value_offsets=pa_offset_buf,
                            data=pa_data_buf,
                            null_bitmap=validity_arg,
                            null_count=null_count_int
                        )
                        print(f"SUCCESS: BinaryArray created for {col.name}")
                    else: # Fallback if type mismatch
                        warnings.warn(f"Type mismatch for varlen column {col.name}. Expected String/Binary, got {pa_type}. Creating null array.")
                        arr = pa.nulls(rows_int, type=pa_type)
                except Exception as e_from_buffers:
                    print(f"ERROR in from_buffers for {col.name}: {e_from_buffers}")
                    print(f"Exception type: {type(e_from_buffers)}")
                    import traceback
                    print("Full traceback:")
                    traceback.print_exc()
                    # Create fallback null array
                    arr = pa.nulls(rows, type=pa_type)
                    print(f"Created fallback null array for {col.name}")

            else: # Fixed-width
                # Get buffer from bufs dict
                # Tuple: (d_values, d_nulls, stride)
                if col.name not in bufs or len(bufs[col.name]) != 3:
                     raise ValueError(f"Fixed length buffer tuple not found or invalid for {col.name}")
                d_values_col = bufs[col.name][0]
                stride = bufs[col.name][2]
                expected_item_size = pa_type.byte_width if hasattr(pa_type, 'byte_width') else stride # Use stride if byte_width not available (e.g., bool)

                if d_values_col is None:
                     raise ValueError(f"Missing data buffer for fixed column {col.name}")

                # Check if data is contiguous or needs gathering due to stride
                is_contiguous = (stride == expected_item_size)

                # Wrap GPU buffer or copy if needed
                if PYARROW_CUDA_AVAILABLE and is_contiguous:
                    pa_data_buf = pa_cuda.as_cuda_buffer(d_values_col)
                else:
                    if not is_contiguous:
                        warnings.warn(f"Copying fixed-length column {col.name} to host due to stride ({stride} != {expected_item_size}).")
                        # Gather data on host
                        host_vals_np = d_values_col.copy_to_host()
                        np_dtype = pa_type.to_pandas_dtype() # Get numpy dtype
                        gathered_data = np.empty(rows, dtype=np_dtype)
                        item_size = np.dtype(np_dtype).itemsize
                        for r in range(rows):
                             start_byte = r * stride
                             if start_byte + item_size <= host_vals_np.size:
                                 gathered_data[r] = np.frombuffer(host_vals_np[start_byte:start_byte+item_size], dtype=np_dtype)[0]
                             else: # Should not happen if allocation was correct
                                 warnings.warn(f"Potential out-of-bounds read during gather for {col.name} row {r}")
                                 # Set to default value or handle as null? For now, let numpy decide default.
                        pa_data_buf = pa.py_buffer(gathered_data)
                    else:
                        warnings.warn(f"pyarrow.cuda not available. Copying fixed column {col.name} to host.")
                        pa_data_buf = pa.py_buffer(d_values_col.copy_to_host())


                # Create array using from_buffers
                # Note: For boolean, from_buffers expects a bit-packed buffer.
                if pa.types.is_boolean(pa_type):
                     # Strategy: Copy byte-per-bool data from GPU, pack on CPU, then use from_buffers.
                     # This avoids needing a GPU packing kernel for now.
                     warnings.warn("Packing boolean data on CPU before using from_buffers.")
                     host_byte_bools = d_values_col.copy_to_host()
                     # Ensure stride is handled if necessary (though bool stride is likely 1)
                     if not is_contiguous:
                         # This path should ideally not be hit for bool (stride=1)
                         # but handle defensively.
                         warnings.warn(f"Gathering boolean bytes due to stride {stride} != 1.")
                         np_byte_type = np.uint8
                         gathered_bytes = np.empty(rows, dtype=np_byte_type)
                         item_size = 1 # bool is 1 byte here
                         for r in range(rows):
                             start_byte = r * stride
                             if start_byte + item_size <= host_byte_bools.size:
                                 gathered_bytes[r] = np.frombuffer(host_byte_bools[start_byte:start_byte+item_size], dtype=np_byte_type)[0]
                             else:
                                 warnings.warn(f"Potential out-of-bounds read during bool gather for {col.name} row {r}")
                         host_byte_bools = gathered_bytes

                     # Pack the bytes into bits (LSB order for Arrow)
                     packed_bits = np.packbits(host_byte_bools.view(np.bool_), bitorder='little')
                     pa_data_buf = pa.py_buffer(packed_bits)
                     # Now use from_buffers with the packed data buffer
                     arr = pa.BooleanArray.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
                elif pa.types.is_decimal(pa_type):
                     arr = pa.Decimal128Array.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
                elif pa.types.is_fixed_size_list(pa_type) or pa.types.is_fixed_size_binary(pa_type) or \
                     pa.types.is_primitive(pa_type): # Catches numeric, date, timestamp etc.
                     arr = pa.Array.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
                else:
                     warnings.warn(f"Cannot use from_buffers for fixed type {pa_type} of column {col.name}. Falling back to host copy and pa.array().")
                     # Fallback to host copy for unsupported types
                     host_vals_np = d_values_col.copy_to_host()
                     np_dtype = pa_type.to_pandas_dtype()
                     arr = pa.array(host_vals_np.view(np_dtype), type=pa_type, mask=~boolean_mask_np)


        except Exception as e_assembly:
            print(f"Error assembling Arrow array for column {col.name} (type {pa_type}): {e_assembly}")
            arr = pa.nulls(rows, type=pa_type if pa_type else pa.null()) # Fallback

        if arr is None: # Should not happen with fallbacks, but as a safeguard
            print(f"Array creation failed unexpectedly for {col.name}. Creating null array.")
            arr = pa.nulls(rows, type=pa_type if pa_type else pa.null())

        arrays.append(arr)

    batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    print("--- Finished Arrow Assembly ---")
    return batch
def convert_gpu_buffers_to_cudf_dataframe(bufs, columns, rows, d_nulls_all):
    """GPU buffers を cuDF DataFrame に変換（PyArrow経由で確実性を保証）"""
    import cudf
    import pyarrow as pa
    import numpy as np
    from .type_map import (
        DECIMAL128, UTF8, BINARY, INT16, INT32, INT64,
        FLOAT32, FLOAT64, BOOL, DATE32, TS64_US
    )
    
    arrays = []
    host_nulls_all = d_nulls_all.copy_to_host()
    
    for cidx, col in enumerate(columns):
        # PyArrowと同じロジックでArrow Arrayを作成
        boolean_mask_np = (host_nulls_all[:, cidx] == 1)
        null_count = rows - np.count_nonzero(boolean_mask_np)
        validity_buffer = build_validity_bitmap(boolean_mask_np)
        
        # Arrow type確定
        pa_type = None
        if col.arrow_id == DECIMAL128:
            precision, scale = col.arrow_param or (38, 0)
            if precision is None or precision <= 0 or precision > 38:
                precision = 38
            pa_type = pa.decimal128(precision, scale)
        elif col.arrow_id == UTF8:
            pa_type = pa.string()
        elif col.arrow_id == BINARY:
            pa_type = pa.binary()
        elif col.arrow_id == INT16:
            pa_type = pa.int16()
        elif col.arrow_id == INT32:
            pa_type = pa.int32()
        elif col.arrow_id == INT64:
            pa_type = pa.int64()
        elif col.arrow_id == FLOAT32:
            pa_type = pa.float32()
        elif col.arrow_id == FLOAT64:
            pa_type = pa.float64()
        elif col.arrow_id == BOOL:
            pa_type = pa.bool_()
        elif col.arrow_id == DATE32:
            pa_type = pa.date32()
        elif col.arrow_id == TS64_US:
            pa_type = pa.timestamp('us')
        else:
            pa_type = pa.binary()
        
        # GPU bufferからPyArrow arrayを作成（簡易版）
        if col.is_variable:
            d_values, d_nulls, d_offsets, _ = bufs[col.name]
            host_offsets = d_offsets.copy_to_host()
            host_values = d_values.copy_to_host()
            
            pa_offset_buf = pa.py_buffer(host_offsets)
            pa_data_buf = pa.py_buffer(host_values)
            
            if pa.types.is_string(pa_type):
                arr = pa.StringArray.from_buffers(
                    length=rows,
                    value_offsets=pa_offset_buf,
                    data=pa_data_buf,
                    null_bitmap=validity_buffer,
                    null_count=null_count
                )
            else:  # binary
                arr = pa.BinaryArray.from_buffers(
                    length=rows,
                    value_offsets=pa_offset_buf,
                    data=pa_data_buf,
                    null_bitmap=validity_buffer,
                    null_count=null_count
                )
        else:
            d_values, d_nulls, stride = bufs[col.name]
            host_vals = d_values.copy_to_host()
            
            # stride処理
            elem_size = col.elem_size
            if stride != elem_size:
                gathered_data = np.empty(rows * elem_size, dtype=np.uint8)
                for r in range(rows):
                    start_byte = r * stride
                    gathered_data[r*elem_size:(r+1)*elem_size] = host_vals[start_byte:start_byte+elem_size]
                host_vals = gathered_data
            
            pa_data_buf = pa.py_buffer(host_vals)
            
            if pa.types.is_decimal(pa_type):
                arr = pa.Decimal128Array.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
            else:
                arr = pa.Array.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
        
        arrays.append(arr)
    
    # PyArrow RecordBatchを作成
    batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    
    # PyArrow TableからcuDF DataFrameに変換
    table = pa.Table.from_batches([batch])
    return cudf.DataFrame.from_arrow(table)


def _build_cudf_mask(valid_bool_cupy):
    """cuDF用のビットマスク構築（GPU上で実行）"""
    import cupy as cp
    # 現在の実装では使用されていないが、互換性のために残す
    # 標準的なcuDFシリーズ作成で十分
    return None


def decode_chunk_cudf_optimized(
    raw_dev, field_offsets_dev, field_lengths_dev, columns, output_path=None
):
    """cuDF最適化版のデコード関数（GPU上でParquet出力まで完結）"""
    import cudf
    
    gmm = GPUMemoryManagerV2()
    rows, ncols = field_lengths_dev.shape

    # --- 10のべき乗テーブルをGPUに転送 ---
    d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
    d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)

    # ----------------------------------
    # 1. GPU バッファ確保 (初期確保)
    # ----------------------------------
    bufs: Dict[str, Any] = gmm.initialize_device_buffers(columns, rows)

    # varlen_meta の準備 (Pass 2 で使用)
    varlen_meta = []  # (col_idx, var_idx, name)
    fixedlen_meta = []  # (col_idx, name)
    for cidx, col in enumerate(columns):
        if col.arrow_id == UTF8 or col.arrow_id == BINARY:
            varlen_meta.append((cidx, len(varlen_meta), col.name))
        else:  # Fixed length including DECIMAL128
            fixedlen_meta.append((cidx, col.name))

    # ----------------------------------
    # 2. Pass-1 len/null (GPU Kernel)
    # ----------------------------------
    print("\n--- Running Pass 1 (len/null collection) on GPU ---")
    var_indices_host = _build_var_indices(columns)
    var_indices_dev = cuda.to_device(var_indices_host)
    n_var = len(varlen_meta)

    d_nulls_all = cuda.device_array((rows, ncols), dtype=np.uint8)
    d_var_lens = cuda.device_array((n_var, rows), dtype=np.int32)

    threads_pass1 = 256
    blocks_pass1 = (rows + threads_pass1 - 1) // threads_pass1

    pass1_len_null[blocks_pass1, threads_pass1](
        field_lengths_dev,
        var_indices_dev,
        d_var_lens,
        d_nulls_all
    )
    cuda.synchronize()
    print("--- Finished Pass 1 (GPU) ---")

    # ----------------------------------
    # 3. Prefix-sum offsets (GPU - CuPy) & データバッファ再確保
    # ----------------------------------
    print("--- Running Prefix Sum (GPU - CuPy) & Reallocating Varlen Buffers ---")
    
    initial_offset_buffers = [bufs[name][2] for _, _, name in varlen_meta]

    for v_idx, (cidx, _, name) in enumerate(varlen_meta):
        print(f"\n--- Processing varlen column '{name}' (v_idx={v_idx}, cidx={cidx}) ---")
        
        cp_len = cp.asarray(d_var_lens[v_idx])
        cp_off = cp.cumsum(cp_len, dtype=np.int32)
        total_bytes = int(cp_off[-1].get()) if rows > 0 else 0

        d_offset_col = initial_offset_buffers[v_idx]
        d_offset_col[0] = 0
        cp_off_as_cupy = cp.asarray(cp_off, dtype=np.int32)
        d_offset_col[1 : rows + 1] = cp_off_as_cupy

        # Reallocate the data buffer
        gmm.replace_varlen_data_buffer(name, total_bytes)

    print("--- Finished Prefix Sum & Reallocation ---")

    # ----------------------------------
    # 4. Pass-2 scatter-copy per var-col (GPU Kernel)
    # ----------------------------------
    print("--- Running Pass 2 VarLen (GPU Kernel) ---")
    threads = 256
    blocks = (rows + threads - 1) // threads

    for v_idx, (cidx, _, name) in enumerate(varlen_meta):
        col_meta = columns[cidx]
        if col_meta.arrow_id == UTF8 or col_meta.arrow_id == BINARY:
            d_offset_v = bufs[name][2]
            d_values_v = bufs[name][0]
            field_off_v = field_offsets_dev[:, cidx]
            field_len_v = field_lengths_dev[:, cidx]

            pass2_scatter_varlen[blocks, threads](
                raw_dev,
                field_off_v,
                field_len_v,
                d_offset_v,
                d_values_v
            )
            cuda.synchronize()

    print("--- Finished Pass 2 VarLen ---")

    # ----------------------------------
    # 4.5 Pass-2 scatter-copy for fixed-length cols (GPU Kernel)
    # ----------------------------------
    print("--- Running Pass 2 FixedLen (GPU Kernel) ---")
    
    for cidx, name in fixedlen_meta:
        col = columns[cidx]
        d_vals, d_nulls_col, stride = bufs[name]

        if col.arrow_id == DECIMAL128:
            use_optimization = os.environ.get("USE_DECIMAL_OPTIMIZATION", "1") == "1"
            
            if use_optimization:
                precision, scale = col.arrow_param or (38, 0)
                target_scale = scale

                if precision <= 18 and stride == 8:
                    pass2_scatter_decimal64_optimized[blocks, threads](
                        raw_dev,
                        field_offsets_dev[:, cidx],
                        field_lengths_dev[:, cidx],
                        d_vals,
                        stride,
                        target_scale,
                        d_pow10_table_lo,
                        d_pow10_table_hi
                    )
                else:
                    pass2_scatter_decimal128_optimized[blocks, threads](
                        raw_dev,
                        field_offsets_dev[:, cidx],
                        field_lengths_dev[:, cidx],
                        d_vals,
                        stride,
                        target_scale,
                        d_pow10_table_lo,
                        d_pow10_table_hi
                    )
            else:
                pass2_scatter_decimal128[blocks, threads](
                    raw_dev,
                    field_offsets_dev[:, cidx],
                    field_lengths_dev[:, cidx],
                    d_vals,
                    stride
                )
        else:
            pass2_scatter_fixed[blocks, threads](
                raw_dev,
                field_offsets_dev[:, cidx],
                col.elem_size,
                d_vals,
                stride
            )
    
    cuda.synchronize()
    print("--- Finished Pass 2 FixedLen ---")

    # ----------------------------------
    # 5. cuDF DataFrame構築（GPU上で完結）
    # ----------------------------------
    print("--- Building cuDF DataFrame (GPU-direct) ---")
    df = convert_gpu_buffers_to_cudf_dataframe(bufs, columns, rows, d_nulls_all)

    # GPU上で直接Parquet書き出し
    if output_path:
        print(f"--- Writing Parquet to {output_path} (GPU-direct) ---")
        df.to_parquet(output_path)
        print("--- Finished cuDF Parquet Write ---")
        return None  # メモリ節約
    else:
        print("--- Returning cuDF DataFrame ---")
        return df  # 後続処理用


__all__ = ["decode_chunk", "convert_gpu_buffers_to_cudf_dataframe", "decode_chunk_cudf_optimized"]
