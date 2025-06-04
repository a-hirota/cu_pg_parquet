"""GPU COPY BINARY → Arrow RecordBatch 列ごとDecimal最適化版

主な改善点:
1. Pass1段階でDecimal処理を統合（列ごとに実行）
2. メモリアクセス削減（Decimal列の二重読み込み回避）
3. Numba互換性確保（List[DeviceNDArray]を使わない）
4. 従来構造との整合性維持
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

# カーネルのインポート
from .cuda_kernels.arrow_gpu_pass1_decimal_column_wise import (
    pass1_len_null_decimal_column_wise,
    pass1_len_null_non_decimal
)
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.arrow_gpu_pass2_fixed import pass2_scatter_fixed
from .cuda_kernels.arrow_gpu_pass2_decimal128 import (
    pass2_scatter_decimal128, POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
)

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

def decode_chunk_decimal_column_wise(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # uint8[:]
    field_offsets_dev,  # int32[:, :]
    field_lengths_dev,  # int32[:, :]
    columns: List[ColumnMeta],
    use_pass1_integration: bool = True,  # Pass1統合フラグ
) -> pa.RecordBatch:
    """
    列ごとDecimal最適化版のGPU デコード
    
    Parameters:
    -----------
    use_pass1_integration : bool
        True: Pass1でDecimal処理を統合
        False: 従来のPass1/Pass2分離
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
    decimal_meta = []  # Decimal列メタデータ
    
    for cidx, col in enumerate(columns):
        if col.arrow_id == UTF8 or col.arrow_id == BINARY:
            varlen_meta.append((cidx, len(varlen_meta), col.name))
        elif col.arrow_id == DECIMAL128:
            decimal_meta.append((cidx, len(decimal_meta), col.name))
            # 固定長としても扱う（バッファ管理のため）
            fixedlen_meta.append((cidx, col.name))
        else:
            fixedlen_meta.append((cidx, col.name))

    print(f"\n--- Column-wise Decimal Optimization: {'ENABLED' if use_pass1_integration else 'DISABLED'} ---")
    print(f"Decimal columns detected: {len(decimal_meta)}")
    print(f"Variable length columns: {len(varlen_meta)}")
    print(f"Fixed length columns: {len(fixedlen_meta)}")

    # ----------------------------------
    # Pass1実行: 統合版 vs 従来版
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
        print("--- Running Pass 1 INTEGRATED (column-wise decimal processing) ---")
        
        # Decimal列マスクを作成
        decimal_cols_mask = np.zeros(ncols, dtype=np.uint8)
        for cidx, _, _ in decimal_meta:
            decimal_cols_mask[cidx] = 1
        d_decimal_cols_mask = cuda.to_device(decimal_cols_mask)
        
        # 1. 非Decimal列の処理（従来Pass1）
        print("Processing non-decimal columns...")
        pass1_len_null_non_decimal[blocks_pass1, threads_pass1](
            field_lengths_dev,
            var_indices_dev,
            d_var_lens,
            d_nulls_all,
            d_decimal_cols_mask
        )
        cuda.synchronize()
        
        # 2. Decimal列の統合処理（列ごと）
        print("Processing decimal columns with integration...")
        for cidx, d_idx, name in decimal_meta:
            col = columns[cidx]
            
            # バッファ取得
            d_vals, d_nulls_col_unused, stride = bufs[name]
            
            # スケール取得とデバッグ情報
            print(f"  Debug: {name} - arrow_param={col.arrow_param}, type={type(col.arrow_param)}")
            precision, scale = col.arrow_param or (38, 0)
            print(f"  Debug: {name} - precision={precision}, scale={scale}, types=({type(precision)}, {type(scale)})")
            
            # Handle PostgreSQL NUMERIC default (precision=0, scale=0)
            if precision == 0:
                print(f"Info: {name} has PostgreSQL NUMERIC default (precision=0), using (38, 0)")
                precision = 38
                scale = 0
            elif not isinstance(precision, int) or not (1 <= precision <= 38):
                print(f"Warning: Invalid precision {precision} for {name}, using (38, 0)")
                precision = 38
                scale = 0
            
            if not isinstance(scale, int) or scale < 0 or scale > precision:
                print(f"Warning: Invalid scale {scale} for {name}, using 0")
                scale = 0
            target_scale = scale
            
            # 可変長インデックス取得
            var_idx = var_indices_host[cidx]  # -1 if not variable
            
            print(f"  Processing {name}: precision={precision}, scale={scale}, var_idx={var_idx}")
            
            # 列ごとPass1統合カーネル実行
            pass1_len_null_decimal_column_wise[blocks_pass1, threads_pass1](
                raw_dev,
                field_offsets_dev[:, cidx],    # この列のオフセット
                field_lengths_dev[:, cidx],    # この列の長さ
                d_vals,                        # この列の出力バッファ
                stride,                        # 16バイト
                target_scale,                  # スケール
                d_pow10_table_lo,
                d_pow10_table_hi,
                d_nulls_all[:, cidx],          # この列のNULLフラグ
                var_idx,                       # 可変長インデックス（通常-1）
                d_var_lens                     # 可変長長さ配列
            )
            cuda.synchronize()
            
        print("--- Finished Pass 1 INTEGRATED ---")
        
    else:
        print("--- Running Pass 1 TRADITIONAL (compatibility mode) ---")
        # 従来のPass1を使用
        from .cuda_kernels.arrow_gpu_pass1 import pass1_len_null
        pass1_len_null[blocks_pass1, threads_pass1](
            field_lengths_dev,
            var_indices_dev,
            d_var_lens,
            d_nulls_all
        )
        cuda.synchronize()
        print("--- Finished Pass 1 TRADITIONAL ---")

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

    print("--- Finished Prefix Sum & Reallocation ---")

    # ----------------------------------
    # Pass2処理: 統合版では一部スキップ
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
    # Pass2固定長: Decimal統合版ではスキップ
    # ----------------------------------
    print("--- Running Pass 2 FixedLen ---")
    
    for cidx, name in fixedlen_meta:
        col = columns[cidx]
        d_vals, d_nulls_col, stride = bufs[name]

        if col.arrow_id == DECIMAL128:
            if use_pass1_integration:
                print(f"Skipping Pass 2 for DECIMAL128 column {name} (already processed in Pass 1)")
                continue
            else:
                # 従来版のDecimal処理
                print(f"Running traditional Pass 2 for DECIMAL128 column {name}")
                pass2_scatter_decimal128[blocks, threads](
                    raw_dev,
                    field_offsets_dev[:, cidx],
                    field_lengths_dev[:, cidx],
                    d_vals,
                    stride
                )
        else:
            # その他の固定長列
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
    # Arrow RecordBatch 組立
    # ----------------------------------
    print("--- Assembling Arrow RecordBatch ---")
    arrays = []

    for cidx, col in enumerate(columns):
        print(f"Assembling column: {col.name} (Arrow ID: {col.arrow_id})")
        
        # Validity bitmap
        boolean_mask_np = (host_nulls_all[:, cidx] == 1)
        null_count = rows - np.count_nonzero(boolean_mask_np)
        
        try:
            validity_buffer = build_validity_bitmap(boolean_mask_np)
        except Exception as e_vb:
            print(f"Error building validity bitmap for {col.name}: {e_vb}")
            arrays.append(pa.nulls(rows))
            continue

        # Arrow Type
        pa_type = None
        if col.arrow_id == DECIMAL128:
            precision, scale = col.arrow_param or (38, 0)
            # Handle PostgreSQL NUMERIC default (precision=0, scale=0)
            if precision == 0:
                precision, scale = 38, 0
            elif not isinstance(precision, int) or not (1 <= precision <= 38):
                warnings.warn(f"Invalid precision {precision} for DECIMAL column {col.name}. Using (38, 0).")
                precision, scale = 38, 0
            
            if not isinstance(scale, int) or scale < 0 or scale > precision:
                warnings.warn(f"Invalid scale {scale} for DECIMAL column {col.name}. Using scale=0.")
                scale = 0
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
            pa_type = pa.timestamp('us', tz=tz_info)
        else:
            warnings.warn(f"Unhandled arrow_id {col.arrow_id} for column {col.name}")
            pa_type = pa.binary()

        # Array creation
        arr = None
        try:
            if col.is_variable:
                d_values_col = bufs[col.name][0]
                d_offsets_col = bufs[col.name][2]

                if PYARROW_CUDA_AVAILABLE:
                    pa_offset_buf = pa_cuda.as_cuda_buffer(d_offsets_col)
                    pa_data_buf = pa_cuda.as_cuda_buffer(d_values_col)
                else:
                    host_offsets = d_offsets_col.copy_to_host()
                    host_values = d_values_col.copy_to_host() if d_values_col.size > 0 else np.array([], dtype=np.uint8)
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
                elif pa.types.is_binary(pa_type):
                    arr = pa.BinaryArray.from_buffers(
                        length=rows,
                        value_offsets=pa_offset_buf,
                        data=pa_data_buf,
                        null_bitmap=validity_buffer,
                        null_count=null_count
                    )

            else:  # Fixed-width
                d_values_col = bufs[col.name][0]
                stride = bufs[col.name][2]
                expected_item_size = pa_type.byte_width if hasattr(pa_type, 'byte_width') else stride

                is_contiguous = (stride == expected_item_size)

                if PYARROW_CUDA_AVAILABLE and is_contiguous:
                    pa_data_buf = pa_cuda.as_cuda_buffer(d_values_col)
                else:
                    if not is_contiguous:
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

                if pa.types.is_boolean(pa_type):
                    host_byte_bools = d_values_col.copy_to_host()
                    if not is_contiguous:
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
                else:
                    arr = pa.Array.from_buffers(pa_type, rows, [validity_buffer, pa_data_buf], null_count=null_count)

        except Exception as e_assembly:
            print(f"Error assembling Arrow array for column {col.name}: {e_assembly}")
            arr = pa.nulls(rows, type=pa_type)

        if arr is None:
            arr = pa.nulls(rows, type=pa_type)

        arrays.append(arr)

    batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    print("--- Finished Arrow Assembly ---")
    return batch

__all__ = ["decode_chunk_decimal_column_wise"]