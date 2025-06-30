#!/usr/bin/env python3
"""
GPU使用率とスレッド行数制限の詳細分析
"""

import sys
sys.path.append('/home/ubuntu/gpupgparser')

from src.cuda_kernels.postgres_binary_parser import (
    get_device_properties, calculate_optimal_grid_sm_aware,
    estimate_row_size_from_columns
)
from src.types import ColumnMeta, INT32, INT64, DECIMAL128, DATE32, UTF8

def analyze_gpu_utilization():
    """GPU使用率と行数制限の分析"""
    
    # GPUデバイス情報
    props = get_device_properties()
    sm_count = props['MULTIPROCESSOR_COUNT']
    max_threads_per_block = props['MAX_THREADS_PER_BLOCK']
    
    print("=== GPU Device Information ===")
    print(f"SM Count: {sm_count}")
    print(f"Max Threads per Block: {max_threads_per_block}")
    
    # 理論上の最大同時実行スレッド数
    # A100: 各SMあたり最大2048スレッド
    # 他のGPU: 通常1536-2048スレッド/SM
    max_threads_per_sm = 2048  # A100の値
    theoretical_max_threads = sm_count * max_threads_per_sm
    
    print(f"\n=== Theoretical Maximum ===")
    print(f"Max Threads per SM: {max_threads_per_sm}")
    print(f"Total Resident Threads: {theoretical_max_threads:,}")
    
    # 実際の計算例（8GBチャンク）
    chunk_size = 8 * 1024**3
    header_size = 19
    data_size = chunk_size - header_size
    row_size = 288  # lineorderテーブル
    
    blocks_x, blocks_y, threads_per_block = calculate_optimal_grid_sm_aware(data_size, row_size)
    total_threads = blocks_x * blocks_y * threads_per_block
    total_blocks = blocks_x * blocks_y
    
    print(f"\n=== Actual Configuration (8GB chunk) ===")
    print(f"Grid: ({blocks_x}, {blocks_y})")
    print(f"Threads per Block: {threads_per_block}")
    print(f"Total Blocks: {total_blocks:,}")
    print(f"Total Threads: {total_threads:,}")
    
    # GPU占有率の計算
    # 占有率 = 実行中のワープ数 / 最大ワープ数
    warp_size = 32
    warps_per_block = threads_per_block // warp_size
    total_warps = total_blocks * warps_per_block
    
    # 各SMが処理するブロック数
    blocks_per_sm = (total_blocks + sm_count - 1) // sm_count
    warps_per_sm = blocks_per_sm * warps_per_block
    max_warps_per_sm = max_threads_per_sm // warp_size
    
    print(f"\n=== Occupancy Analysis ===")
    print(f"Warps per Block: {warps_per_block}")
    print(f"Total Warps: {total_warps:,}")
    print(f"Blocks per SM: {blocks_per_sm}")
    print(f"Warps per SM: {warps_per_sm}")
    print(f"Max Warps per SM: {max_warps_per_sm}")
    print(f"SM Occupancy: {min(100, warps_per_sm/max_warps_per_sm*100):.1f}%")
    
    # スレッドあたりの処理データ
    MAX_ROWS_PER_THREAD = 200
    thread_stride = data_size // total_threads
    max_thread_stride = row_size * MAX_ROWS_PER_THREAD
    actual_thread_stride = min(thread_stride, max_thread_stride)
    rows_per_thread = actual_thread_stride / row_size
    
    print(f"\n=== Thread Workload ===")
    print(f"Thread Stride (theoretical): {thread_stride:,} bytes")
    print(f"Thread Stride (limited): {actual_thread_stride:,} bytes")
    print(f"Rows per Thread: {rows_per_thread:.1f}")
    print(f"MAX_ROWS_PER_THREAD: {MAX_ROWS_PER_THREAD}")
    
    # ローカル配列サイズの制約
    print(f"\n=== Local Memory Constraints ===")
    print(f"Local Array Size: 256 rows")
    print(f"Safety Margin: {256 - MAX_ROWS_PER_THREAD} rows")
    
    # CUDAローカルメモリサイズ計算
    local_memory_per_thread = (
        256 * 8 +       # local_positions (uint64)
        256 * 17 * 8 +  # local_field_offsets (uint64)
        256 * 17 * 4    # local_field_lengths (int32)
    )
    print(f"Local Memory per Thread: {local_memory_per_thread:,} bytes ({local_memory_per_thread/1024:.1f} KB)")
    
    # 256行にした場合の影響
    print(f"\n=== Impact of Changing to 256 Rows ===")
    if MAX_ROWS_PER_THREAD < 256:
        new_thread_stride = row_size * 256
        new_required_threads = data_size // new_thread_stride
        new_total_blocks = (new_required_threads + threads_per_block - 1) // threads_per_block
        
        print(f"Current Threads: {total_threads:,}")
        print(f"New Required Threads: {new_required_threads:,}")
        print(f"Thread Reduction: {(1 - new_required_threads/total_threads)*100:.1f}%")
        print(f"New Total Blocks: {new_total_blocks:,}")
        
        # 487行欠損問題への影響
        print(f"\n⚠️ WARNING: Increasing to 256 removes safety margin")
        print(f"- Risk of data loss if any thread processes exactly 256 rows")
        print(f"- No buffer for boundary cases or variable row sizes")

if __name__ == "__main__":
    analyze_gpu_utilization()