"""GPU COPY BINARY → Arrow RecordBatch Ultimate統合版

Pass2完全廃止の革命的アーキテクチャ:
1. 固定長列統合処理（Decimal最適化継承）
2. 可変長文字列統合処理（新規）
3. 1回のカーネル起動で全処理完了
4. GPU並列Prefix-sum内蔵
5. メモリアクセス効率最大化

期待効果: 3-5倍の性能向上
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
from .gpu_memory_manager_v4_ultimate import GPUMemoryManagerV4Ultimate

# カーネルのインポート
from .cuda_kernels.arrow_gpu_pass1_ultimate_integrated import pass1_ultimate_integrated
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

def decode_chunk_ultimate_integrated(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # uint8[:]
    field_offsets_dev,  # int32[:, :]
    field_lengths_dev,  # int32[:, :]
    columns: List[ColumnMeta],
) -> pa.RecordBatch:
    """
    Ultimate統合版GPUデコード - Pass2完全廃止
    
    革新的特徴:
    - 1回のカーネル起動で全列処理
    - 固定長列統合バッファ
    - 可変長文字列統合バッファ
    - GPU並列Prefix-sum
    - メモリコアレッシング最適化
    """
    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    print(f"\n=== Ultimate統合版デコード開始 ===")
    print(f"行数: {rows:,}, 列数: {ncols}")

    # 10のべき乗テーブルをGPUに転送
    d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
    d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)

    # ----------------------------------
    # 1. Ultimate統合バッファシステム初期化
    # ----------------------------------
    print("1. Ultimate統合バッファシステム初期化中...")
    gmm_v4 = GPUMemoryManagerV4Ultimate()
    ultimate_info = gmm_v4.initialize_ultimate_buffers(columns, rows)
    
    fixed_layouts = ultimate_info.fixed_layouts
    var_layouts = ultimate_info.var_layouts
    
    print(f"   固定長列: {len(fixed_layouts)}列 (統合バッファ)")
    print(f"   可変長文字列列: {len(var_layouts)}列 (統合バッファ)")
    print(f"   固定長行ストライド: {ultimate_info.row_stride}バイト")

    # ----------------------------------
    # 2. Ultimate統合カーネル引数準備
    # ----------------------------------
    print("2. Ultimate統合カーネル引数準備中...")
    
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

    # 可変長文字列列情報
    var_count = len(var_layouts)
    if var_count > 0:
        var_indices_array = np.array([layout.column_index for layout in var_layouts], dtype=np.int32)
        d_var_indices = cuda.to_device(var_indices_array)
        
        # オフセット配列を2Dにまとめる
        d_var_offset_arrays = cuda.device_array((var_count, rows + 1), dtype=np.int32)
        
        # 各列の最初の要素を0で初期化
        @cuda.jit
        def init_offsets_kernel(offset_arrays, var_count):
            i = cuda.grid(1)
            if i < var_count:
                offset_arrays[i, 0] = 0
        
        threads = min(var_count, 256)
        blocks = (var_count + threads - 1) // threads
        if blocks > 0:
            init_offsets_kernel[blocks, threads](d_var_offset_arrays, var_count)
            cuda.synchronize()
        
    else:
        d_var_indices = cuda.to_device(np.array([], dtype=np.int32))
        d_var_offset_arrays = cuda.device_array((1, rows + 1), dtype=np.int32)  # ダミー

    # 共通NULL配列
    d_nulls_all = cuda.device_array((rows, ncols), dtype=np.uint8)

    # ----------------------------------
    # 3. Ultimate統合カーネル実行
    # ----------------------------------
    print("3. Ultimate統合カーネル実行中...")
    
    threads = 256
    blocks = (rows + threads - 1) // threads
    
    print(f"   カーネル構成: blocks={blocks}, threads={threads}")
    print(f"   固定長統合バッファサイズ: {ultimate_info.fixed_buffer.size:,}バイト")
    if var_count > 0:
        print(f"   可変長統合バッファサイズ: {ultimate_info.var_data_buffer.size:,}バイト")
    
    # 1回のカーネル起動で全処理完了
    pass1_ultimate_integrated[blocks, threads](
        raw_dev,
        field_offsets_dev,
        field_lengths_dev,
        
        # 統合固定長バッファ
        ultimate_info.fixed_buffer,
        ultimate_info.row_stride,
        
        # 固定長レイアウト情報
        fixed_count,
        d_fixed_types,
        d_fixed_offsets,
        d_fixed_sizes,
        d_fixed_indices,
        d_fixed_scales,
        
        # 可変長文字列バッファ
        var_count,
        d_var_indices,
        ultimate_info.var_data_buffer,
        
        # 一時作業領域
        ultimate_info.var_lens_buffer,
        d_var_offset_arrays,
        
        # 共通出力
        d_nulls_all,
        
        # Decimal処理用
        d_pow10_table_lo,
        d_pow10_table_hi
    )
    cuda.synchronize()
    print("   Ultimate統合カーネル完了！Pass2完全廃止達成！")

    # ----------------------------------
    # 4. 統合バッファから個別列バッファ抽出
    # ----------------------------------
    print("4. 統合バッファから個別列バッファ抽出中...")
    
    # 固定長列の抽出
    fixed_buffers = gmm_v4.extract_fixed_column_arrays(ultimate_info, rows)
    print(f"   固定長列抽出完了: {len(fixed_buffers)}列")
    
    # 可変長列のオフセットバッファ更新
    if var_count > 0:
        for i, layout in enumerate(var_layouts):
            col_name = layout.name
            ultimate_info.var_offset_buffers[col_name][:] = d_var_offset_arrays[i, :]
        print(f"   可変長列オフセット更新完了: {var_count}列")

    # ----------------------------------
    # 5. Arrow RecordBatch 組立
    # ----------------------------------
    print("5. Arrow RecordBatch組立中...")
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
            if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY):
                # 可変長文字列列
                d_values_col = ultimate_info.var_data_buffer
                d_offsets_col = ultimate_info.var_offset_buffers[col.name]

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
    print("=== Ultimate統合版デコード完了: Pass2完全廃止達成！ ===")
    return batch

__all__ = ["decode_chunk_ultimate_integrated"]