"""
GPU Decoder V5: 真の並列統合処理
===============================

課題：V4では可変長文字列が逐次処理
解決：Pass1でPrefix-sum事前計算→並列文字列コピー
"""

import time
import numpy as np
import pyarrow as pa
from numba import cuda
import cupy as cp

from .gpu_memory_manager_v4_ultimate import GPUMemoryManagerV4Ultimate

def decode_chunk_true_parallel(raw_dev, field_offsets_dev, field_lengths_dev, columns):
    """
    Pass1で完全並列処理を実現したデコーダー
    
    **アーキテクチャ**：
    1. 可変長列の長さ収集（Pass1-Phase1）
    2. CuPyでPrefix-sum高速計算
    3. 固定長+可変長を完全並列処理（Pass1-Phase2）
    4. Arrow組立
    """
    
    print("\n=== Pass1真の並列版デコード開始 ===")
    rows, cols = field_offsets_dev.shape
    print(f"行数: {rows:,}, 列数: {cols}")
    
    start_total = time.time()
    
    # ===== Step 1: 統合バッファ初期化 =====
    print("1. 統合バッファシステム初期化中...")
    gmm_v5 = GPUMemoryManagerV4Ultimate()
    ultimate_info = gmm_v5.initialize_ultimate_buffers(columns, rows)
    
    print(f"   固定長列: {len(ultimate_info.fixed_layouts)}列 (統合バッファ)")
    print(f"   可変長列: {len(ultimate_info.var_layouts)}列 (統合バッファ)")
    print(f"   行ストライド: {ultimate_info.row_stride}バイト")
    
    # ===== Step 2: Phase1 - 可変長列長さ収集 =====
    print("2. Phase1: 可変長列長さ収集中...")
    
    var_layouts = ultimate_info.var_layouts
    var_count = len(var_layouts)
    
    if var_count > 0:
        # 長さ収集カーネル実行
        from .cuda_kernels.arrow_gpu_pass1_ultimate_simple import pass1_ultimate_simple
        
        # 長さ収集用バッファ
        d_var_lens = cuda.device_array((var_count, rows), dtype=np.int32)
        d_nulls_temp = cuda.device_array((rows, cols), dtype=np.uint8)
        
        # 10のべき乗テーブル
        from .cuda_kernels.arrow_gpu_pass2_decimal128 import POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
        d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
        d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)
        
        # 固定長列情報配列
        fixed_layouts = ultimate_info.fixed_layouts
        fixed_count = len(fixed_layouts)
        
        if fixed_count > 0:
            fixed_types = np.array([layout.arrow_type_id for layout in fixed_layouts], dtype=np.int32)
            fixed_offsets = np.array([layout.buffer_offset for layout in fixed_layouts], dtype=np.int32)
            fixed_sizes = np.array([layout.element_size for layout in fixed_layouts], dtype=np.int32)
            fixed_indices = np.array([layout.column_index for layout in fixed_layouts], dtype=np.int32)
            fixed_scales = np.array([layout.decimal_scale for layout in fixed_layouts], dtype=np.int32)
            
            d_fixed_types = cuda.to_device(fixed_types)
            d_fixed_offsets = cuda.to_device(fixed_offsets)
            d_fixed_sizes = cuda.to_device(fixed_sizes)
            d_fixed_indices = cuda.to_device(fixed_indices)
            d_fixed_scales = cuda.to_device(fixed_scales)
        else:
            d_fixed_types = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_offsets = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_sizes = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_indices = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_scales = cuda.to_device(np.array([], dtype=np.int32))
        
        # 可変長列情報
        var_indices_array = np.array([layout.column_index for layout in var_layouts], dtype=np.int32)
        d_var_indices = cuda.to_device(var_indices_array)
        
        # Phase1カーネル実行（長さ収集のみ）
        threads = 256
        blocks = (rows + threads - 1) // threads
        
        print(f"   Phase1カーネル実行: blocks={blocks}, threads={threads}")
        pass1_ultimate_simple[blocks, threads](
            raw_dev, field_offsets_dev, field_lengths_dev,
            ultimate_info.fixed_buffer, ultimate_info.row_stride,
            fixed_count, d_fixed_types, d_fixed_offsets, d_fixed_sizes, d_fixed_indices, d_fixed_scales,
            var_count, d_var_indices, d_var_lens, d_nulls_temp,
            d_pow10_table_lo, d_pow10_table_hi
        )
        cuda.synchronize()
        print("   Phase1完了（固定長処理+可変長長さ収集）")
        
        # ===== Step 3: CuPyでPrefix-sum高速計算 =====
        print("3. CuPyでPrefix-sum高速計算中...")
        
        var_offset_arrays_list = []
        var_data_buffers_list = []
        
        for var_idx, layout in enumerate(var_layouts):
            # 長さ配列をCuPyに変換
            lengths_cupy = cp.asarray(d_var_lens[var_idx, :])
            
            # CuPyの高速Prefix-sum
            offsets_cupy = cp.zeros(rows + 1, dtype=cp.int32)
            offsets_cupy[1:] = cp.cumsum(lengths_cupy)
            
            total_bytes = int(offsets_cupy[-1])
            print(f"   {layout.name}: 総バイト数={total_bytes:,}")
            
            # オフセット配列をGPUに保存
            offset_array = cuda.device_array(rows + 1, dtype=np.int32)
            offset_array[:] = offsets_cupy
            var_offset_arrays_list.append(offset_array)
            
            # データバッファ作成
            data_buffer = cuda.device_array(total_bytes, dtype=np.uint8)
            var_data_buffers_list.append(data_buffer)
        
        print("   Prefix-sum計算完了")
        
        # ===== Step 4: Phase2 - 真の並列統合処理 =====
        print("4. Phase2: 真の並列統合処理中...")
        
        # 統合可変長データバッファを作成
        total_var_bytes = sum(buf.size for buf in var_data_buffers_list)
        unified_var_buffer = cuda.device_array(total_var_bytes, dtype=np.uint8)
        
        # オフセット配列を統合
        unified_offset_array = cuda.device_array((var_count, rows + 1), dtype=np.int32)
        var_buffer_start = 0
        
        for var_idx, (offset_array, data_buffer) in enumerate(zip(var_offset_arrays_list, var_data_buffers_list)):
            # 統合バッファ内でのオフセット調整
            offset_host = offset_array.copy_to_host()
            offset_host += var_buffer_start
            unified_offset_array[var_idx, :] = offset_host
            var_buffer_start += data_buffer.size
        
        # NULL配列
        d_nulls_all = cuda.device_array((rows, cols), dtype=np.uint8)
        
        # Phase2カーネル実行（真の並列処理）
        from .cuda_kernels.arrow_gpu_pass1_true_parallel import pass1_true_parallel_kernel
        
        print(f"   Phase2カーネル実行: blocks={blocks}, threads={threads}")
        pass1_true_parallel_kernel[blocks, threads](
            raw_dev, field_offsets_dev, field_lengths_dev,
            ultimate_info.fixed_buffer, ultimate_info.row_stride,
            fixed_count, d_fixed_types, d_fixed_offsets, d_fixed_sizes, d_fixed_indices, d_fixed_scales,
            var_count, d_var_indices, unified_var_buffer, unified_offset_array,
            d_nulls_all, d_pow10_table_lo, d_pow10_table_hi
        )
        cuda.synchronize()
        print("   Phase2完了（真の並列統合処理）")
        
        # ===== Step 5: Arrow組立 =====
        print("5. Arrow RecordBatch組立中...")
        
        arrays = []
        
        # 固定長列の組立
        for layout in fixed_layouts:
            col = columns[layout.column_index]
            
            # 統合バッファから抽出
            column_data = cuda.device_array((rows, layout.element_size), dtype=np.uint8)
            
            for row in range(rows):
                src_offset = row * ultimate_info.row_stride + layout.buffer_offset
                column_data[row, :] = ultimate_info.fixed_buffer[src_offset:src_offset + layout.element_size]
            
            # Arrowの型に応じて変換
            if layout.arrow_type_id == 5:  # DECIMAL128
                precision = 38
                scale = 0
                arrow_type = pa.decimal128(precision, scale)
                host_data = column_data.copy_to_host()
                arrow_array = pa.array(host_data.tobytes(), type=arrow_type)
            elif layout.arrow_type_id == 1:  # INT32
                host_data = column_data.copy_to_host().view(np.int32).flatten()
                arrow_array = pa.array(host_data)
            else:
                # その他の型
                host_data = column_data.copy_to_host()
                arrow_array = pa.array(host_data.tobytes())
            
            arrays.append(arrow_array)
        
        # 可変長列の組立
        var_buffer_start = 0
        for var_idx, layout in enumerate(var_layouts):
            data_size = var_data_buffers_list[var_idx].size
            
            # データ抽出
            var_data = unified_var_buffer[var_buffer_start:var_buffer_start + data_size].copy_to_host()
            var_offsets = var_offset_arrays_list[var_idx].copy_to_host()
            
            # PyArrowバッファ作成
            pa_data_buf = pa.py_buffer(var_data)
            pa_offset_buf = pa.py_buffer(var_offsets)
            
            # StringArray作成
            arrow_array = pa.StringArray.from_buffers(
                length=rows,
                value_offsets=pa_offset_buf,
                data=pa_data_buf
            )
            arrays.append(arrow_array)
            
            var_buffer_start += data_size
    
    else:
        # 可変長列がない場合
        print("2-5. 固定長列のみの処理...")
        arrays = []
        
        # 固定長列のみの組立処理
        # （省略：上記と同様）
    
    # RecordBatch作成
    field_names = [col.name for col in columns]
    record_batch = pa.RecordBatch.from_arrays(arrays, names=field_names)
    
    total_time = time.time() - start_total
    print(f"=== Pass1真の並列版デコード完了 ({total_time:.4f}秒) ===")
    
    return record_batch

__all__ = ["decode_chunk_true_parallel"]