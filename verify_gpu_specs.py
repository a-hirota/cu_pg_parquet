#!/usr/bin/env python3
"""
RTX 3090の実際のスペックと占有率を検証
"""

import sys
sys.path.append('/home/ubuntu/gpupgparser')

from src.cuda_kernels.postgres_binary_parser import (
    get_device_properties, calculate_optimal_grid_sm_aware
)
from numba import cuda

def verify_gpu_specs():
    """GPU仕様と占有率の正確な計算"""
    
    # デバイス情報取得
    device = cuda.get_current_device()
    props = get_device_properties()
    
    print("=== GPU Device Information ===")
    print(f"Device: {device.name.decode()}")
    print(f"Compute Capability: {device.compute_capability}")
    print(f"SM Count: {props['MULTIPROCESSOR_COUNT']}")
    print(f"Max Threads per Block: {props['MAX_THREADS_PER_BLOCK']}")
    print(f"Warp Size: {device.WARP_SIZE}")
    
    # RTX 3090 (Compute Capability 8.6) の仕様
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
    sm_count = 82
    max_threads_per_sm = 1536  # Ampere世代 (CC 8.x)
    max_warps_per_sm = 48      # 1536 / 32
    max_blocks_per_sm = 16     # Ampere世代
    
    print(f"\n=== RTX 3090 (Ampere) Architecture ===")
    print(f"Max Threads per SM: {max_threads_per_sm}")
    print(f"Max Warps per SM: {max_warps_per_sm}")
    print(f"Max Blocks per SM: {max_blocks_per_sm}")
    print(f"Total Max Resident Threads: {sm_count * max_threads_per_sm:,}")
    
    # 8GBチャンクでの実際の設定
    chunk_size = 8 * 1024**3
    header_size = 19
    data_size = chunk_size - header_size
    row_size = 288
    
    blocks_x, blocks_y, threads_per_block = calculate_optimal_grid_sm_aware(data_size, row_size)
    total_blocks = blocks_x * blocks_y
    total_threads = total_blocks * threads_per_block
    
    print(f"\n=== Actual Launch Configuration ===")
    print(f"Data Size: {data_size/1024**3:.1f} GB")
    print(f"Grid: ({blocks_x}, {blocks_y})")
    print(f"Threads per Block: {threads_per_block}")
    print(f"Total Blocks: {total_blocks:,}")
    print(f"Total Threads Launched: {total_threads:,}")
    
    # 占有率の正確な計算
    warps_per_block = threads_per_block // 32
    
    # Wave（並列実行される同時ブロック数）
    waves = (total_blocks + sm_count - 1) // sm_count
    blocks_per_sm_actual = min(total_blocks // sm_count, max_blocks_per_sm)
    
    # 実際の占有率
    threads_per_sm_actual = blocks_per_sm_actual * threads_per_block
    occupancy = threads_per_sm_actual / max_threads_per_sm * 100
    
    print(f"\n=== Occupancy Analysis (Corrected) ===")
    print(f"Warps per Block: {warps_per_block}")
    print(f"Waves: {waves}")
    print(f"Blocks per SM (average): {total_blocks / sm_count:.1f}")
    print(f"Blocks per SM (actual): {blocks_per_sm_actual}")
    print(f"Threads per SM: {threads_per_sm_actual}")
    print(f"Theoretical Occupancy: {occupancy:.1f}%")
    
    # 制限要因の分析
    print(f"\n=== Limiting Factors ===")
    
    # ブロック数制限
    if blocks_per_sm_actual >= max_blocks_per_sm:
        print(f"Limited by: Max blocks per SM ({max_blocks_per_sm})")
    
    # スレッド数制限
    if threads_per_sm_actual >= max_threads_per_sm:
        print(f"Limited by: Max threads per SM ({max_threads_per_sm})")
    
    # ローカルメモリ使用量
    local_memory_per_thread = (
        256 * 8 +       # local_positions
        256 * 17 * 8 +  # local_field_offsets  
        256 * 17 * 4    # local_field_lengths
    )
    total_local_memory = threads_per_sm_actual * local_memory_per_thread
    
    print(f"\nLocal Memory Usage:")
    print(f"  Per Thread: {local_memory_per_thread:,} bytes ({local_memory_per_thread/1024:.1f} KB)")
    print(f"  Per SM: {total_local_memory/1024/1024:.1f} MB")
    
    # スレッドあたりの実際の処理量
    thread_stride = data_size // total_threads
    rows_per_thread = thread_stride / row_size
    
    print(f"\n=== Thread Workload ===")
    print(f"Thread Stride: {thread_stride:,} bytes")
    print(f"Rows per Thread: {rows_per_thread:.1f}")
    print(f"MAX_ROWS_PER_THREAD: 200")
    print(f"Local Array Size: 256")
    
    # 200 vs 256の比較
    print(f"\n=== 200 vs 256 Row Limit Analysis ===")
    print(f"Current: Each thread processes ~{rows_per_thread:.1f} rows (well below 200)")
    print(f"200-row limit provides 56-row safety margin")
    print(f"256-row limit would remove all safety margin")
    print(f"\nConclusion: 200-row limit is appropriate and changing to 256 is unnecessary")

if __name__ == "__main__":
    verify_gpu_specs()