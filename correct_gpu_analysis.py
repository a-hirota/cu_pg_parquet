#!/usr/bin/env python3
"""
GPUの正確な使用状況分析（修正版）
"""

import sys
sys.path.append('/home/ubuntu/gpupgparser')

from src.cuda_kernels.postgres_binary_parser import (
    get_device_properties, calculate_optimal_grid_sm_aware
)
from numba import cuda

def correct_gpu_analysis():
    """GPU使用状況の正確な分析"""
    
    # デバイス情報
    device = cuda.get_current_device()
    props = get_device_properties()
    
    print("=== RTX 3090 Specifications ===")
    print(f"Device: {device.name.decode()}")
    print(f"Compute Capability: {device.compute_capability}")
    print(f"SM Count: {props['MULTIPROCESSOR_COUNT']}")
    
    # RTX 3090 (Ampere GA102) の正確な仕様
    sm_count = 82
    max_threads_per_sm = 1536     # Ampere世代の仕様
    max_warps_per_sm = 48         # 1536 / 32
    max_blocks_per_sm = 16        # Ampere世代の制限
    warp_size = 32
    
    print(f"\nArchitecture Limits:")
    print(f"  Max Threads per SM: {max_threads_per_sm}")
    print(f"  Max Warps per SM: {max_warps_per_sm}")
    print(f"  Max Blocks per SM: {max_blocks_per_sm}")
    
    # 8GBチャンクでの設定
    chunk_size = 8 * 1024**3
    data_size = chunk_size - 19
    row_size = 288
    
    blocks_x, blocks_y, threads_per_block = calculate_optimal_grid_sm_aware(data_size, row_size)
    total_blocks = blocks_x * blocks_y
    total_threads_launched = total_blocks * threads_per_block
    
    print(f"\n=== Launch Configuration ===")
    print(f"Grid Dimensions: ({blocks_x}, {blocks_y})")
    print(f"Threads per Block: {threads_per_block}")
    print(f"Total Blocks: {total_blocks:,}")
    print(f"Total Threads (起動数): {total_threads_launched:,}")
    
    # 重要：起動されるスレッド数と同時実行可能なスレッド数は異なる
    print(f"\n=== Thread Execution Analysis ===")
    
    # Wave計算（全ブロックを実行するのに必要な回数）
    waves = (total_blocks + sm_count - 1) // sm_count
    print(f"Execution Waves: {waves}")
    print(f"Average Blocks per SM: {total_blocks / sm_count:.1f}")
    
    # 各SMでの実際の同時実行状況
    # 制限要因を考慮
    warps_per_block = threads_per_block // warp_size
    
    # 最大同時実行可能ブロック数（制限要因により決定）
    max_blocks_limited_by_threads = max_threads_per_sm // threads_per_block
    max_blocks_limited_by_blocks = max_blocks_per_sm
    actual_blocks_per_sm = min(max_blocks_limited_by_threads, max_blocks_limited_by_blocks)
    
    print(f"\nPer-SM Execution:")
    print(f"  Max blocks (thread limit): {max_blocks_limited_by_threads}")
    print(f"  Max blocks (block limit): {max_blocks_limited_by_blocks}")
    print(f"  Actual blocks per SM: {actual_blocks_per_sm}")
    
    # 実際の占有率
    actual_threads_per_sm = actual_blocks_per_sm * threads_per_block
    actual_warps_per_sm = actual_blocks_per_sm * warps_per_block
    
    print(f"\nOccupancy:")
    print(f"  Threads per SM: {actual_threads_per_sm}")
    print(f"  Warps per SM: {actual_warps_per_sm}")
    print(f"  Thread Occupancy: {actual_threads_per_sm / max_threads_per_sm * 100:.1f}%")
    print(f"  Warp Occupancy: {actual_warps_per_sm / max_warps_per_sm * 100:.1f}%")
    
    # 全GPU での同時実行
    total_resident_threads = sm_count * actual_threads_per_sm
    print(f"\nGPU-wide Execution:")
    print(f"  Total Resident Threads: {total_resident_threads:,}")
    print(f"  Total Launched Threads: {total_threads_launched:,}")
    print(f"  Ratio: {total_threads_launched / total_resident_threads:.1f}x")
    
    # ローカルメモリ分析
    local_memory_per_thread = 54272  # 53KB
    print(f"\n=== Local Memory Analysis ===")
    print(f"Local Memory per Thread: {local_memory_per_thread:,} bytes")
    print(f"Local Memory per Block: {local_memory_per_thread * threads_per_block / 1024 / 1024:.1f} MB")
    
    # スレッドの実際の作業量
    thread_stride = data_size // total_threads_launched
    rows_per_thread = thread_stride / row_size
    
    print(f"\n=== Thread Workload ===")
    print(f"Data per Thread: {thread_stride:,} bytes")
    print(f"Rows per Thread: {rows_per_thread:.1f}")
    
    # 行数制限の分析
    print(f"\n=== Row Limit Analysis ===")
    print(f"MAX_ROWS_PER_THREAD: 200")
    print(f"Local Array Size: 256")
    print(f"Safety Margin: 56 rows")
    print(f"Actual Rows per Thread: {rows_per_thread:.1f}")
    
    if rows_per_thread < 200:
        print(f"\n✓ 各スレッドの処理行数（{rows_per_thread:.1f}）は制限値（200）より十分少ない")
        print(f"✓ 200行制限は適切で、256行への変更は不要")
    
    # GPU使用率の結論
    print(f"\n=== Conclusion ===")
    print(f"1. GPU使用率: {actual_warps_per_sm / max_warps_per_sm * 100:.1f}% （十分高い）")
    print(f"2. 各スレッドは平均{rows_per_thread:.1f}行を処理（200行制限の{rows_per_thread/200*100:.1f}%）")
    print(f"3. 200行制限は安全マージンを確保し、現在の処理に十分")
    print(f"4. GPUは{waves}波で全データを処理（効率的）")

if __name__ == "__main__":
    correct_gpu_analysis()