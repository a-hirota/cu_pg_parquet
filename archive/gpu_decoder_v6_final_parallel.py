"""
GPU Decoder V6: 最終並列統合処理
===============================

Ultimate統合版のデッドロック問題を完全に回避し、
シンプル版をベースに真の並列処理を実現
"""

import time
import numpy as np
import pyarrow as pa
from numba import cuda
import cupy as cp

from .gpu_memory_manager_v4_ultimate import GPUMemoryManagerV4Ultimate

def decode_chunk_final_parallel(raw_dev, field_offsets_dev, field_lengths_dev, columns):
    """
    最終並列統合デコーダー
    
    **安全なアーキテクチャ**：
    1. Phase1: 動作確認済みシンプル版で固定長処理+可変長長さ収集
    2. CuPyでPrefix-sum高速計算
    3. Phase2: シンプルな並列文字列コピーカーネル
    4. Arrow組立
    """
    
    print("\n=== 最終並列統合版デコード開始 ===")
    rows, cols = field_offsets_dev.shape
    print(f"行数: {rows:,}, 列数: {cols}")
    
    start_total = time.time()
    
    # ===== Step 1: 統合バッファ初期化 =====
    print("1. 統合バッファシステム初期化中...")
    gmm_v6 = GPUMemoryManagerV4Ultimate()
    ultimate_info = gmm_v6.initialize_ultimate_buffers(columns, rows)
    
    print(f"   固定長列: {len(ultimate_info.fixed_layouts)}列")
    print(f"   可変長列: {len(ultimate_info.var_layouts)}列")
    
    # ===== Step 2: Phase1 - 動作確認済みシンプル版 =====
    print("2. Phase1: 統合処理（動作確認済みシンプル版）...")
    
    var_layouts = ultimate_info.var_layouts
    var_count = len(var_layouts)
    
    # 長さ収集用バッファ
    d_var_lens = cuda.device_array((var_count, rows), dtype=np.int32)
    d_nulls_all = cuda.device_array((rows, cols), dtype=np.uint8)
    
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
    if var_count > 0:
        var_indices_array = np.array([layout.column_index for layout in var_layouts], dtype=np.int32)
        d_var_indices = cuda.to_device(var_indices_array)
    else:
        d_var_indices = cuda.to_device(np.array([], dtype=np.int32))
    
    # Phase1カーネル実行（動作確認済み）
    threads = 256
    blocks = (rows + threads - 1) // threads
    
    print(f"   Phase1カーネル実行: blocks={blocks}, threads={threads}")
    
    from .cuda_kernels.arrow_gpu_pass1_ultimate_simple import pass1_ultimate_simple
    
    pass1_ultimate_simple[blocks, threads](
        raw_dev, field_offsets_dev, field_lengths_dev,
        ultimate_info.fixed_buffer, ultimate_info.row_stride,
        fixed_count, d_fixed_types, d_fixed_offsets, d_fixed_sizes, d_fixed_indices, d_fixed_scales,
        var_count, d_var_indices, d_var_lens, d_nulls_all,
        d_pow10_table_lo, d_pow10_table_hi
    )
    cuda.synchronize()
    print("   Phase1完了（固定長統合処理+可変長長さ収集）")
    
    # ===== Step 3: CuPyでPrefix-sum高速計算 =====
    print("3. CuPyでPrefix-sum高速計算中...")
    
    var_data_buffers = []
    var_offset_arrays = []
    
    if var_count > 0:
        for var_idx, layout in enumerate(var_layouts):
            # 長さ配列をCuPyに変換
            lengths_cupy = cp.asarray(d_var_lens[var_idx, :])
            
            # CuPyの高速Prefix-sum
            offsets_cupy = cp.zeros(rows + 1, dtype=cp.int32)
            offsets_cupy[1:] = cp.cumsum(lengths_cupy)
            
            total_bytes = int(offsets_cupy[-1])
            print(f"   {layout.name}: 総バイト数={total_bytes:,}")
            
            # データバッファとオフセット配列作成
            data_buffer = cuda.device_array(total_bytes, dtype=np.uint8)
            offset_array = cuda.device_array(rows + 1, dtype=np.int32)
            offset_array[:] = offsets_cupy
            
            var_data_buffers.append(data_buffer)
            var_offset_arrays.append(offset_array)
    
    print("   Prefix-sum計算完了")
    
    # ===== Step 4: Phase2 - 並列文字列コピー =====
    print("4. Phase2: 並列文字列コピー中...")
    
    if var_count > 0:
        # 各可変長列について並列コピー実行
        for var_idx, layout in enumerate(var_layouts):
            col_idx = layout.column_index
            data_buffer = var_data_buffers[var_idx]
            offset_array = var_offset_arrays[var_idx]
            
            # シンプルな並列コピーカーネル
            @cuda.jit
            def simple_string_copy_kernel(
                raw_data, field_offsets_2d, field_lengths_2d,
                target_col_idx, dest_buffer, dest_offsets
            ):
                row = cuda.grid(1)
                if row >= field_offsets_2d.shape[0]:
                    return
                
                # フィールド情報取得
                src_offset = field_offsets_2d[row, target_col_idx]
                field_length = field_lengths_2d[row, target_col_idx]
                
                # NULL または無効な長さの場合はスキップ
                if field_length <= 0 or src_offset < 0:
                    return
                
                # 目的地オフセット取得
                dest_offset = dest_offsets[row]
                
                # 安全な範囲チェック付きコピー
                max_copy_len = min(
                    field_length,
                    raw_data.size - src_offset,
                    dest_buffer.size - dest_offset
                )
                
                if max_copy_len > 0:
                    for i in range(max_copy_len):
                        dest_buffer[dest_offset + i] = raw_data[src_offset + i]
            
            # 並列コピー実行
            simple_string_copy_kernel[blocks, threads](
                raw_dev, field_offsets_dev, field_lengths_dev,
                col_idx, data_buffer, offset_array
            )
            cuda.synchronize()
            
            print(f"   完了: {layout.name}")
    
    print("   Phase2完了（並列文字列コピー）")
    
    # ===== Step 5: Arrow組立 =====
    print("5. Arrow RecordBatch組立中...")
    
    arrays = []
    
    # 固定長列の組立
    for layout in fixed_layouts:
        col = columns[layout.column_index]
        
        # 統合バッファから列データ抽出
        column_data = cuda.device_array(rows * layout.element_size, dtype=np.uint8)
        
        # 効率的な抽出（行単位コピー）
        @cuda.jit
        def extract_fixed_column(
            unified_buffer, row_stride, col_offset, col_size,
            output_buffer, total_rows
        ):
            row = cuda.grid(1)
            if row >= total_rows:
                return
            
            src_base = row * row_stride + col_offset
            dest_base = row * col_size
            
            for i in range(col_size):
                if src_base + i < unified_buffer.size and dest_base + i < output_buffer.size:
                    output_buffer[dest_base + i] = unified_buffer[src_base + i]
        
        extract_blocks = (rows + threads - 1) // threads
        extract_fixed_column[extract_blocks, threads](
            ultimate_info.fixed_buffer, ultimate_info.row_stride,
            layout.buffer_offset, layout.element_size,
            column_data, rows
        )
        cuda.synchronize()
        
        # Arrow配列作成
        host_data = column_data.copy_to_host()
        
        if layout.arrow_type_id == 5:  # DECIMAL128
            precision, scale = 38, 0
            arrow_type = pa.decimal128(precision, scale)
            arrow_array = pa.Array.from_buffers(
                arrow_type, rows, [None, pa.py_buffer(host_data)]
            )
        elif layout.arrow_type_id == 1:  # INT32
            arrow_array = pa.array(host_data.view(np.int32))
        elif layout.arrow_type_id == 2:  # INT64
            arrow_array = pa.array(host_data.view(np.int64))
        else:
            # フォールバック
            arrow_array = pa.array(host_data.tobytes())
        
        arrays.append(arrow_array)
    
    # 可変長列の組立
    if var_count > 0:
        for var_idx, layout in enumerate(var_layouts):
            # データとオフセットをホストに転送
            var_data = var_data_buffers[var_idx].copy_to_host()
            var_offsets = var_offset_arrays[var_idx].copy_to_host()
            
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
    
    # RecordBatch作成
    field_names = [col.name for col in columns]
    record_batch = pa.RecordBatch.from_arrays(arrays, names=field_names)
    
    total_time = time.time() - start_total
    print(f"=== 最終並列統合版デコード完了 ({total_time:.4f}秒) ===")
    
    return record_batch

__all__ = ["decode_chunk_final_parallel"]