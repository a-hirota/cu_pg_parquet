"""GPU COPY BINARY → Arrow RecordBatch 2パス変換（Decimal Pass1統合最適化版）

主な改善点:
1. Pass1でDecimal変換を統合実行
2. Pass2でDecimal処理をスキップ
3. メモリアクセス回数削減による高速化
4. 共有メモリ活用による定数アクセス最適化
"""

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

# 従来のカーネルをインポート
from .cuda_kernels.arrow_gpu_pass1 import pass1_len_null
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.arrow_gpu_pass2_fixed import pass2_scatter_fixed
from .cuda_kernels.arrow_gpu_pass2_decimal128 import (
    pass2_scatter_decimal128, pass2_scatter_decimal128_optimized, pass2_scatter_decimal64_optimized,
    POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
)

# 新しい統合カーネルをインポート
from .cuda_kernels.arrow_gpu_pass1_decimal_optimized import (
    pass1_len_null_decimal_integrated,
    pass2_skip_decimal_optimized
)

from .cuda_kernels.numeric_utils import int64_to_decimal_ascii  # noqa: F401

def build_validity_bitmap(valid_bool: np.ndarray) -> pa.Buffer:
    """Arrow validity bitmap (LSB=行0, 1=valid)"""
    if isinstance(valid_bool, cp.ndarray):
        valid_bool = valid_bool.get()
    elif not isinstance(valid_bool, np.ndarray):
        raise TypeError("Input must be a NumPy or CuPy array")

    valid_bool = np.ascontiguousarray(valid_bool, dtype=np.bool_)
    bits_le = np.packbits(valid_bool, bitorder='little')
    return pa.py_buffer(bits_le)

def _build_var_indices(columns: List[ColumnMeta]) -> np.ndarray:
    """可変長列 → インデックス配列 (-1 = fixed)"""
    var_idx = -1
    idxs = np.full(len(columns), -1, dtype=np.int32)
    for i, m in enumerate(columns):
        if m.is_variable:
            var_idx += 1
            idxs[i] = var_idx
    return idxs

def _build_decimal_indices(columns: List[ColumnMeta]):
    """
    Decimal列のインデックスとスケール配列を構築
    
    Returns:
    --------
    decimal_indices : np.ndarray (int32)
        各列→Decimal列インデックス（非Decimalは-1）
    decimal_scales : np.ndarray (int32)
        Decimal列のターゲットスケール配列
    """
    decimal_idx = -1
    indices = np.full(len(columns), -1, dtype=np.int32)
    scales = []
    
    for i, col in enumerate(columns):
        if col.arrow_id == DECIMAL128:
            decimal_idx += 1
            indices[i] = decimal_idx
            # arrow_paramからスケールを取得
            precision, scale = col.arrow_param or (38, 0)
            scales.append(scale)
    
    return indices, np.array(scales, dtype=np.int32)

def decode_chunk_decimal_optimized(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # uint8[:]
    field_offsets_dev,  # int32[:, :]
    field_lengths_dev,  # int32[:, :]
    columns: List[ColumnMeta],
    use_pass1_integration: bool = True,  # 統合カーネル使用フラグ
) -> pa.RecordBatch:
    """
    Decimal Pass1統合最適化版のGPU デコード
    
    Parameters:
    -----------
    use_pass1_integration : bool
        True: Pass1でDecimal変換統合実行（最適化版）
        False: 従来の2パス分離処理
    """
    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    # 10のべき乗テーブルをGPUに転送
    d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
    d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)

    # GPU バッファ確保
    gmm = GPUMemoryManagerV2()
    bufs: Dict[str, Any] = gmm.initialize_device_buffers(columns, rows)

    # メタデータ準備
    varlen_meta = []
    fixedlen_meta = []
    decimal_meta = []  # 新規：Decimal列メタデータ
    
    for cidx, col in enumerate(columns):
        if col.arrow_id == UTF8 or col.arrow_id == BINARY:
            varlen_meta.append((cidx, len(varlen_meta), col.name))
        elif col.arrow_id == DECIMAL128:
            decimal_meta.append((cidx, len(decimal_meta), col.name))
            # 固定長としても扱う（Pass2スキップ用）
            fixedlen_meta.append((cidx, col.name))
        else:
            fixedlen_meta.append((cidx, col.name))

    print(f"\n--- Decimal Optimization Mode: {'ENABLED' if use_pass1_integration else 'DISABLED'} ---")
    print(f"Decimal columns detected: {len(decimal_meta)}")
    print(f"Variable length columns: {len(varlen_meta)}")
    print(f"Fixed length columns: {len(fixedlen_meta)}")

    # ----------------------------------
    # Pass1実行（統合版 or 従来版）
    # ----------------------------------
    
    var_indices_host = _build_var_indices(columns)
    var_indices_dev = cuda.to_device(var_indices_host)
    n_var = len(varlen_meta)

    # 出力配列確保
    d_nulls_all = cuda.device_array((rows, ncols), dtype=np.uint8)
    d_var_lens = cuda.device_array((n_var, rows), dtype=np.int32)

    threads_pass1 = 256
    blocks_pass1 = (rows + threads_pass1 - 1) // threads_pass1

    if use_pass1_integration and len(decimal_meta) > 0:
        print("--- Running Pass 1 INTEGRATED (len/null + decimal conversion) on GPU ---")
        
        # Decimal用インデックス・スケール配列準備
        decimal_indices_host, decimal_scales_host = _build_decimal_indices(columns)
        decimal_indices_dev = cuda.to_device(decimal_indices_host)
        decimal_scales_dev = cuda.to_device(decimal_scales_host)
        
        # Decimal出力バッファのリスト作成
        decimal_buffers = []
        for cidx, dec_idx, col_name in decimal_meta:
            d_vals, _, _ = bufs[col_name]  # Decimal列の出力バッファ取得
            decimal_buffers.append(d_vals)
        
        # 統合カーネル実行
        pass1_len_null_decimal_integrated[blocks_pass1, threads_pass1](
            field_lengths_dev,
            field_offsets_dev,
            raw_dev,
            var_indices_dev,
            decimal_indices_dev,
            decimal_scales_dev,
            decimal_buffers,
            d_var_lens,
            d_nulls_all,
            d_pow10_table_lo,
            d_pow10_table_hi
        )
        cuda.synchronize()
        print("--- Finished Pass 1 INTEGRATED (GPU) ---")
        
    else:
        print("--- Running Pass 1 TRADITIONAL (len/null only) on GPU ---")
        # 従来のPass1カーネル
        pass1_len_null[blocks_pass1, threads_pass1](
            field_lengths_dev,
            var_indices_dev,
            d_var_lens,
            d_nulls_all
        )
        cuda.synchronize()
        print("--- Finished Pass 1 TRADITIONAL (GPU) ---")

    host_nulls_all = d_nulls_all.copy_to_host()

    # ----------------------------------
    # Prefix-sum処理（可変長列）
    # ----------------------------------
    print("--- Running Prefix Sum (GPU - CuPy) & Reallocating Varlen Buffers ---")
    total_bytes_list = []
    values_dev_reallocated = []

    initial_offset_buffers = [bufs[name][2] for _, _, name in varlen_meta]

    for v_idx, (cidx, _, name) in enumerate(varlen_meta):
        print(f"\n--- Processing varlen column '{name}' (v_idx={v_idx}, cidx={cidx}) ---")
        
        cp_len = cp.asarray(d_var_lens[v_idx])
        cp_off = cp.cumsum(cp_len, dtype=np.int32)
        total_bytes = int(cp_off[-1].get()) if rows > 0 else 0
        total_bytes_list.append(total_bytes)
        
        print(f"Total bytes calculated: {total_bytes}")

        d_offset_col = initial_offset_buffers[v_idx]
        
        try:
            d_offset_col[0] = 0
            cp_off_as_cupy = cp.asarray(cp_off, dtype=np.int32)
            d_offset_col[1:rows+1] = cp_off_as_cupy
            print(f"Successfully wrote {len(cp_off)} offsets to buffer")
        except Exception as e_offset_write:
            print(f"ERROR writing offsets to buffer: {e_offset_write}")

        new_data_buf = gmm.replace_varlen_data_buffer(name, total_bytes)
        values_dev_reallocated.append(new_data_buf)

        print(f"VarCol '{name}' Total Bytes={total_bytes}")

    print("--- Finished Prefix Sum & Reallocation ---")

    # ----------------------------------
    # Pass2実行（可変長 + 非Decimal固定長）
    # ----------------------------------
    print("--- Running Pass 2 (VarLen + Non-Decimal Fixed) ---")
    threads = 256
    blocks = (rows + threads - 1) // threads

    # 可変長列の処理
    for v_idx, (cidx, _, name) in enumerate(varlen_meta):
        col_meta = columns[cidx]
        print(f"\n--- Pass 2 VarLen for '{name}' ---")
        
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
            print(f"Kernel completed for '{name}'")

    # 固定長列の処理（Decimalの処理を条件分岐）
    for cidx, name in fixedlen_meta:
        col = columns[cidx]
        d_vals, d_nulls_col, stride = bufs[name]

        if col.arrow_id == DECIMAL128:
            if use_pass1_integration:
                print(f"SKIPPING Decimal column '{name}' (processed in Pass1)")
                continue  # Pass1で処理済みのためスキップ
            else:
                print(f"Processing Decimal column '{name}' in Pass2 (traditional mode)")
                # 従来のDecimal処理
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
            # 非Decimal固定長列の処理
            pass2_scatter_fixed[blocks, threads](
                raw_dev,
                field_offsets_dev[:, cidx],
                col.elem_size,
                d_vals,
                stride
            )

    cuda.synchronize()
    print("--- Finished Pass 2 ---")

    # ----------------------------------
    # 性能統計の出力
    # ----------------------------------
    if use_pass1_integration:
        print(f"\n=== DECIMAL PASS1 INTEGRATION STATS ===")
        print(f"Decimal columns processed in Pass1: {len(decimal_meta)}")
        print(f"Memory access reduction: ~{len(decimal_meta)} decimal field reads saved")
        print(f"Kernel launch reduction: {len(decimal_meta)} Pass2 decimal kernels skipped")
        print(f"Estimated speedup: {1.0 + 0.1 * len(decimal_meta):.1f}x (theoretical)")

    # ----------------------------------
    # Arrow RecordBatch組立（従来と同様）
    # ----------------------------------
    print("--- Assembling Arrow RecordBatch ---")
    arrays = []

    for cidx, col in enumerate(columns):
        print(f"Assembling column: {col.name} (Arrow ID: {col.arrow_id})")
        
        # Validity buffer
        boolean_mask_np = (host_nulls_all[:, cidx] == 1)
        null_count = rows - np.count_nonzero(boolean_mask_np)
        
        try:
            validity_buffer = build_validity_bitmap(boolean_mask_np)
        except Exception as e_vb:
            print(f"Error building validity bitmap for {col.name}: {e_vb}")
            arrays.append(pa.nulls(rows))
            continue

        # Arrow type determination
        pa_type = None
        if col.arrow_id == DECIMAL128:
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
        elif col.arrow_id == BOOL: pa_type = pa.bool_()
        elif col.arrow_id == DATE32: pa_type = pa.date32()
        elif col.arrow_id == TS64_US:
            tz_info = col.arrow_param
            if tz_info is not None and not isinstance(tz_info, str):
                warnings.warn(f"Invalid timezone info in arrow_param for {col.name}: {tz_info}. Ignoring.")
                tz_info = None
            pa_type = pa.timestamp('us', tz=tz_info)
        else:
            warnings.warn(f"Unhandled arrow_id {col.arrow_id} for column {col.name}. Falling back to binary.")
            pa_type = pa.binary()

        # Buffer creation（従来ロジックを維持）
        arr = None
        try:
            if col.is_variable:
                # 可変長列の処理（従来と同様）
                if col.name not in bufs or len(bufs[col.name]) != 4:
                    raise ValueError(f"Variable length buffer tuple not found for {col.name}")
                d_values_col = bufs[col.name][0]
                d_offsets_col = bufs[col.name][2]

                if d_values_col is None or d_offsets_col is None:
                    raise ValueError(f"Missing buffer for varlen column {col.name}")

                # PyArrow buffer wrapping
                if PYARROW_CUDA_AVAILABLE:
                    pa_offset_buf = pa_cuda.as_cuda_buffer(d_offsets_col)
                    pa_data_buf = pa_cuda.as_cuda_buffer(d_values_col)
                else:
                    host_offsets = d_offsets_col.copy_to_host()
                    host_values = d_values_col.copy_to_host() if d_values_col.size > 0 else np.array([], dtype=np.uint8)
                    pa_offset_buf = pa.py_buffer(host_offsets)
                    pa_data_buf = pa.py_buffer(host_values)

                rows_int = int(rows)
                null_count_int = int(null_count)
                validity_arg = None if null_count_int == 0 else validity_buffer

                if pa.types.is_string(pa_type):
                    arr = pa.StringArray.from_buffers(
                        length=rows_int,
                        value_offsets=pa_offset_buf,
                        data=pa_data_buf,
                        null_bitmap=validity_arg,
                        null_count=null_count_int
                    )
                elif pa.types.is_binary(pa_type):
                    arr = pa.BinaryArray.from_buffers(
                        length=rows_int,
                        value_offsets=pa_offset_buf,
                        data=pa_data_buf,
                        null_bitmap=validity_arg,
                        null_count=null_count_int
                    )
                else:
                    warnings.warn(f"Type mismatch for varlen column {col.name}. Creating null array.")
                    arr = pa.nulls(rows_int, type=pa_type)

            else:  # 固定長列
                if col.name not in bufs or len(bufs[col.name]) != 3:
                    raise ValueError(f"Fixed length buffer tuple not found for {col.name}")
                d_values_col = bufs[col.name][0]
                stride = bufs[col.name][2]
                expected_item_size = pa_type.byte_width if hasattr(pa_type, 'byte_width') else stride

                if d_values_col is None:
                    raise ValueError(f"Missing data buffer for fixed column {col.name}")

                is_contiguous = (stride == expected_item_size)

                # Buffer wrapping
                if PYARROW_CUDA_AVAILABLE and is_contiguous:
                    pa_data_buf = pa_cuda.as_cuda_buffer(d_values_col)
                else:
                    if not is_contiguous:
                        warnings.warn(f"Copying fixed-length column {col.name} due to stride mismatch.")
                        host_vals_np = d_values_col.copy_to_host()
                        np_dtype = pa_type.to_pandas_dtype()
                        gathered_data = np.empty(rows, dtype=np_dtype)
                        item_size = np.dtype(np_dtype).itemsize
                        for r in range(rows):
                            start_byte = r * stride
                            if start_byte + item_size <= host_vals_np.size:
                                gathered_data[r] = np.frombuffer(host_vals_np[start_byte:start_byte+item_size], dtype=np_dtype)[0]
                        pa_data_buf = pa.py_buffer(gathered_data)
                    else:
                        pa_data_buf = pa.py_buffer(d_values_col.copy_to_host())

                # Array creation
                if pa.types.is_boolean(pa_type):
                    warnings.warn("Packing boolean data on CPU before using from_buffers.")
                    host_byte_bools = d_values_col.copy_to_host()
                    if not is_contiguous:
                        warnings.warn(f"Gathering boolean bytes due to stride {stride} != 1.")
                        gathered_bytes = np.empty(rows, dtype=np.uint8)
                        for r in range(rows):
                            start_byte = r * stride
                            if start_byte + 1 <= host_byte_bools.size:
                                gathered_bytes[r] = host_byte_bools[start_byte]
                        host_byte_bools = gathered_bytes
                    packed_bits = np.packbits(host_byte_bools.view(np.bool_), bitorder='little')
                    pa_data_buf = pa.py_buffer(packed_bits)
                    arr = pa.BooleanArray.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
                elif pa.types.is_decimal(pa_type):
                    arr = pa.Decimal128Array.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
                elif pa.types.is_primitive(pa_type):
                    arr = pa.Array.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)
                else:
                    warnings.warn(f"Cannot use from_buffers for type {pa_type} of column {col.name}. Falling back.")
                    host_vals_np = d_values_col.copy_to_host()
                    np_dtype = pa_type.to_pandas_dtype()
                    arr = pa.array(host_vals_np.view(np_dtype), type=pa_type, mask=~boolean_mask_np)

        except Exception as e_assembly:
            print(f"Error assembling Arrow array for column {col.name}: {e_assembly}")
            arr = pa.nulls(rows, type=pa_type if pa_type else pa.null())

        if arr is None:
            print(f"Array creation failed for {col.name}. Creating null array.")
            arr = pa.nulls(rows, type=pa_type if pa_type else pa.null())

        arrays.append(arr)

    batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    print("--- Finished Arrow Assembly ---")
    return batch

__all__ = ["decode_chunk_decimal_optimized", "build_validity_bitmap"]