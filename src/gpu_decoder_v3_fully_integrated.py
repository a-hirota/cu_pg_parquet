"""GPU COPY BINARY → Arrow RecordBatch Pass1完全統合版

革新的な最適化:
1. 固定長列を統合バッファに配置
2. 1回のカーネル起動で全列処理
3. ブロック単位メモリコアレッシング
4. 可変長列は従来方式維持

期待効果: 5-15倍の性能向上
"""

from __future__ import annotations

from typing import List, Dict, Any
import os
import warnings
import numpy as np
import cupy as cp
import pyarrow as pa
try:
    import pyarrow.cuda as pa_cuda
    PYARROW_CUDA_AVAILABLE = True
except ImportError:
    pa_cuda = None
    PYARROW_CUDA_AVAILABLE = False

from numba import cuda

from .type_map import *
from .gpu_memory_manager_v3_unified import GPUMemoryManagerV3Unified

# カーネルのインポート
from .cuda_kernels.arrow_gpu_pass1_fully_integrated import pass1_fully_integrated
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.arrow_gpu_pass2_decimal128 import (
    POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
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

def decode_chunk_fully_integrated(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # uint8[:]
    field_offsets_dev,  # int32[:, :]
    field_lengths_dev,  # int32[:, :]
    columns: List[ColumnMeta],
) -> pa.RecordBatch:
    """
    Pass1完全統合版のGPU デコード
    
    革新的特徴:
    - 1回のカーネル起動で全固定長列処理
    - 統合バッファによるメモリ効率最大化
    - ブロック単位の最適化
    """
    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    print(f"\n=== Pass1完全統合版デコード開始 ===")
    print(f"行数: {rows:,}, 列数: {ncols}")

    # 10のべき乗テーブルをGPUに転送
    d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
    d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)

    # ----------------------------------
    # 1. 統合メモリバッファシステム初期化
    # ----------------------------------
    print("1. 統合バッファシステム初期化中...")
    gmm_v3 = GPUMemoryManagerV3Unified()
    unified_info = gmm_v3.initialize_unified_buffers(columns, rows)
    
    fixed_layouts = unified_info.fixed_layouts
    var_column_indices = unified_info.var_column_indices
    
    print(f"   固定長列: {len(fixed_layouts)}列 (統合バッファ)")
    print(f"   可変長列: {len(var_column_indices)}列 (個別バッファ)")
    print(f"   行ストライド: {unified_info.row_stride}バイト")

    # ----------------------------------
    # 2. カーネル引数準備
    # ----------------------------------
    print("2. カーネル引数準備中...")
    
    # 固定長列情報の配列化
    fixed_count = len(fixed_layouts)
    if fixed_count > 0:
        fixed_types = np.array([layout.arrow_type_id for layout in fixed_layouts], dtype=np.int32)
        fixed_offsets = np.array([layout.buffer_offset for layout in fixed_layouts], dtype=np.int32)  
        fixed_sizes = np.array([layout.element_size for layout in fixed_layouts], dtype=np.int32)
        fixed_indices = np.array([layout.column_index for layout in fixed_layouts], dtype=np.int32)
        fixed_scales = np.array([layout.decimal_scale for layout in fixed_layouts], dtype=np.int32)
        
        # GPU転送
        d_fixed_types = cuda.to_device(fixed_types)
        d_fixed_offsets = cuda.to_device(fixed_offsets)
        d_fixed_sizes = cuda.to_device(fixed_sizes)
        d_fixed_indices = cuda.to_device(fixed_indices)
        d_fixed_scales = cuda.to_device(fixed_scales)
    else:
        # 空配列
        d_fixed_types = cuda.to_device(np.array([], dtype=np.int32))
        d_fixed_offsets = cuda.to_device(np.array([], dtype=np.int32))
        d_fixed_sizes = cuda.to_device(np.array([], dtype=np.int32))
        d_fixed_indices = cuda.to_device(np.array([], dtype=np.int32))
        d_fixed_scales = cuda.to_device(np.array([], dtype=np.int32))

    # 可変長列情報
    var_count = len(var_column_indices)
    if var_count > 0:
        var_indices_array = np.array(var_column_indices, dtype=np.int32)
        d_var_indices = cuda.to_device(var_indices_array)
        d_var_lens = cuda.device_array((var_count, rows), dtype=np.int32)
    else:
        d_var_indices = cuda.to_device(np.array([], dtype=np.int32))
        d_var_lens = cuda.device_array((1, rows), dtype=np.int32)  # ダミー

    # 共通NULL配列
    d_nulls_all = cuda.device_array((rows, ncols), dtype=np.uint8)

    # ----------------------------------
    # 3. Pass1完全統合カーネル実行
    # ----------------------------------
    print("3. Pass1完全統合カーネル実行中...")
    
    threads = 256
    blocks = (rows + threads - 1) // threads
    
    print(f"   カーネル構成: blocks={blocks}, threads={threads}")
    print(f"   統合バッファサイズ: {unified_info.buffer.size:,}バイト")
    
    # 1回のカーネル起動で全列処理
    pass1_fully_integrated[blocks, threads](
        raw_dev,
        field_offsets_dev,
        field_lengths_dev,
        
        # 統合固定長バッファ
        unified_info.buffer,
        unified_info.row_stride,
        
        # 固定長レイアウト情報
        fixed_count,
        d_fixed_types,
        d_fixed_offsets,
        d_fixed_sizes,
        d_fixed_indices,
        d_fixed_scales,
        
        # 可変長情報
        var_count,
        d_var_indices,
        d_var_lens,
        
        # 共通出力
        d_nulls_all,
        
        # Decimal処理用
        d_pow10_table_lo,
        d_pow10_table_hi
    )
    cuda.synchronize()
    print("   Pass1完全統合カーネル完了！")

    # ----------------------------------
    # 4. 可変長列のPrefix-sum処理
    # ----------------------------------
    if var_count > 0:
        print("4. 可変長列のPrefix-sum処理中...")
        var_buffers = gmm_v3.get_variable_buffers()
        
        for var_idx, col_idx in enumerate(var_column_indices):
            col = columns[col_idx]
            print(f"   処理中: {col.name}")
            
            # Prefix sum計算
            cp_len = cp.asarray(d_var_lens[var_idx])
            cp_off = cp.cumsum(cp_len, dtype=np.int32)
            total_bytes = int(cp_off[-1].get()) if rows > 0 else 0
            
            # オフセットバッファ更新
            d_values, d_nulls, d_offsets, old_size = var_buffers[col.name]
            d_offsets[0] = 0
            if rows > 0:
                d_offsets[1:rows+1] = cp.asarray(cp_off, dtype=np.int32)
            
            # データバッファ再確保
            gmm_v3.replace_varlen_data_buffer(col.name, total_bytes)
            
            print(f"     総バイト数: {total_bytes:,}")
        
        print("   Prefix-sum処理完了")
    else:
        print("4. 可変長列なし - Prefix-sum処理スキップ")

    # ----------------------------------
    # 5. 可変長列のPass2処理
    # ----------------------------------
    if var_count > 0:
        print("5. 可変長列のPass2処理中...")
        var_buffers = gmm_v3.get_variable_buffers()
        
        for var_idx, col_idx in enumerate(var_column_indices):
            col = columns[col_idx]
            
            if col.arrow_id == UTF8 or col.arrow_id == BINARY:
                d_values, d_nulls, d_offsets, size = var_buffers[col.name]
                
                pass2_scatter_varlen[blocks, threads](
                    raw_dev,
                    field_offsets_dev[:, col_idx],
                    field_lengths_dev[:, col_idx],
                    d_offsets,
                    d_values
                )
                cuda.synchronize()
                print(f"   完了: {col.name}")
        
        print("   Pass2処理完了")
    else:
        print("5. 可変長列なし - Pass2処理スキップ")

    # ----------------------------------
    # 6. 統合バッファから個別列バッファ抽出
    # ----------------------------------
    print("6. 統合バッファから個別列バッファ抽出中...")
    
    # デバッグ: 抽出前の統合バッファ内容確認
    print("\n=== デバッグ: 抽出前の統合バッファ確認 ===")
    try:
        # 統合バッファの最初の数行をチェック
        debug_size = min(1000, unified_info.buffer.size)
        debug_buffer = unified_info.buffer[:debug_size].copy_to_host()
        
        # int32列の位置を特定
        int32_layouts = [layout for layout in fixed_layouts if layout.arrow_type_id == INT32]
        print(f"int32列数: {len(int32_layouts)}")
        
        for layout in int32_layouts[:2]:  # 最初の2つのint32列
            print(f"\n列: {layout.name}")
            print(f"  buffer_offset: {layout.buffer_offset}")
            print(f"  element_size: {layout.element_size}")
            
            # 最初の数行の値を確認
            for row in range(min(3, rows)):
                row_start = row * unified_info.row_stride
                field_start = row_start + layout.buffer_offset
                
                if field_start + 4 <= debug_size:
                    raw_bytes = debug_buffer[field_start:field_start+4]
                    # リトルエンディアンで読み取り（統合バッファはCPUと同じエンディアン）
                    value = int.from_bytes(raw_bytes, byteorder='little', signed=True)
                    hex_str = " ".join(f"{b:02x}" for b in raw_bytes)
                    print(f"  行{row}: offset={field_start}, bytes=[{hex_str}], value={value}")
                else:
                    print(f"  行{row}: offset={field_start}, データ範囲外")
    except Exception as e:
        print(f"統合バッファデバッグエラー: {e}")
    
    fixed_buffers = gmm_v3.extract_fixed_column_arrays(unified_info, rows)
    print(f"   抽出完了: {len(fixed_buffers)}列")

    # ----------------------------------
    # 7. Arrow RecordBatch 組立
    # ----------------------------------
    print("7. Arrow RecordBatch組立中...")
    arrays = []
    host_nulls_all = d_nulls_all.copy_to_host()
    
    for cidx, col in enumerate(columns):
        print(f"   組立中: {col.name}")
        
        # Validity bitmap
        boolean_mask_np = (host_nulls_all[:, cidx] == 1)
        null_count = rows - np.count_nonzero(boolean_mask_np)
        validity_buffer = build_validity_bitmap(boolean_mask_np)
        
        # Arrow Type
        pa_type = None
        if col.arrow_id == DECIMAL128:
            precision, scale = col.arrow_param or (38, 0)
            if precision == 0:
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
            pa_type = pa.timestamp('us', tz=col.arrow_param)
        else:
            warnings.warn(f"Unhandled arrow_id {col.arrow_id} for column {col.name}")
            pa_type = pa.binary()

        # Array creation
        arr = None
        try:
            if col.is_variable:
                # 可変長列
                var_buffers = gmm_v3.get_variable_buffers()
                d_values_col, _, d_offsets_col, _ = var_buffers[col.name]

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

            else:
                # 固定長列
                d_values_col = fixed_buffers[col.name]
                
                if PYARROW_CUDA_AVAILABLE:
                    pa_data_buf = pa_cuda.as_cuda_buffer(d_values_col)
                else:
                    pa_data_buf = pa.py_buffer(d_values_col.copy_to_host())

                if pa.types.is_boolean(pa_type):
                    host_byte_bools = d_values_col.copy_to_host()
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
    print("=== Pass1完全統合版デコード完了 ===")
    return batch

__all__ = ["decode_chunk_fully_integrated"]