#!/usr/bin/env python3
"""256MB境界でのCUDAカーネルスレッドスキップ問題の正確な再現"""

import cupy as cp
import numpy as np
from numba import cuda, uint64, int32
import math

@cuda.jit
def simulate_kernel_bug(results, header_size, thread_stride, data_size):
    """オリジナルカーネルの位置計算ロジックを再現"""
    # オリジナルコードと同じスレッドID計算
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    
    # オリジナルコードの位置計算（520行目）
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)
    
    # データサイズチェック（523行目）
    if start_pos >= data_size:
        return
    
    # 結果を記録
    if tid < results.size:
        results[tid] = start_pos

def test_exact_bug():
    """実際のパラメータで256MB境界バグを再現"""
    print("=== 256MB境界バグの正確な再現 ===\n")
    
    # 実際のパラメータ（CLAUDE.mdから）
    header_size = 11  # PostgreSQL COPY BINARYヘッダサイズ
    thread_stride = 192
    data_size = 800 * 1024 * 1024  # 800MB
    
    # 必要なスレッド数
    total_threads = (data_size - header_size + thread_stride - 1) // thread_stride
    print(f"データサイズ: {data_size:,} bytes ({data_size/1024/1024:.1f}MB)")
    print(f"ヘッダサイズ: {header_size} bytes")
    print(f"スレッドストライド: {thread_stride} bytes")
    print(f"必要スレッド数: {total_threads:,}")
    
    # 問題のスレッド番号付近を詳細に確認
    print("\n=== 問題のスレッド番号付近の計算 ===")
    
    # Thread 1398101付近
    for tid in range(1398099, 1398105):
        start_pos = header_size + tid * thread_stride
        print(f"Thread {tid}: start_pos = {header_size} + {tid} * {thread_stride} = {start_pos} (0x{start_pos:08X})")
    
    print("\n256MB境界 = 268,435,456 (0x10000000)")
    
    # Thread 3404032付近
    for tid in range(3404030, 3404035):
        start_pos = header_size + tid * thread_stride
        print(f"Thread {tid}: start_pos = {header_size} + {tid} * {thread_stride} = {start_pos} (0x{start_pos:08X})")
    
    # GPU実行テスト
    print("\n=== GPU実行テスト ===")
    
    # 結果配列（-1で初期化）
    results_gpu = cp.full(total_threads, -1, dtype=cp.int64)
    
    # カーネル実行パラメータ
    threads_per_block = 256
    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    
    print(f"ブロック数: {blocks}")
    print(f"ブロックあたりスレッド数: {threads_per_block}")
    
    # カーネル実行
    simulate_kernel_bug[blocks, threads_per_block](
        results_gpu, header_size, thread_stride, data_size
    )
    
    # 結果を取得
    results = results_gpu.get()
    
    # 処理されたスレッド数を確認
    processed = np.sum(results >= 0)
    print(f"\n処理されたスレッド数: {processed:,}")
    print(f"期待されるスレッド数: {total_threads:,}")
    print(f"欠落スレッド数: {total_threads - processed:,}")
    
    # ギャップを検出
    print("\n=== ギャップ検出 ===")
    gaps = []
    for i in range(1, len(results)):
        if results[i-1] >= 0 and results[i] == -1:
            # ギャップの開始
            gap_start = i
            while i < len(results) and results[i] == -1:
                i += 1
            gap_end = i
            if gap_end < len(results):
                gaps.append((gap_start-1, gap_end, gap_end - gap_start))
    
    if gaps:
        for last_thread, next_thread, gap_size in gaps[:5]:  # 最初の5個のギャップを表示
            last_pos = results[last_thread]
            next_pos = results[next_thread] if next_thread < len(results) else -1
            print(f"\nギャップ検出:")
            print(f"  最後の処理: Thread {last_thread}, pos=0x{last_pos:08X}")
            print(f"  次の処理: Thread {next_thread}, pos=0x{next_pos:08X}")
            print(f"  スキップされたスレッド数: {gap_size:,}")
            print(f"  スキップされたバイト数: {gap_size * thread_stride:,} bytes")

def analyze_multiplication_overflow():
    """整数乗算のオーバーフロー可能性を分析"""
    print("\n=== 整数乗算オーバーフロー分析 ===\n")
    
    thread_stride = 192
    
    # 32ビット整数の最大値
    max_int32 = 2**31 - 1
    max_uint32 = 2**32 - 1
    
    # オーバーフローするスレッド番号を計算
    overflow_thread_int32 = max_int32 // thread_stride
    overflow_thread_uint32 = max_uint32 // thread_stride
    
    print(f"32ビット符号付き整数でオーバーフローするスレッド番号: {overflow_thread_int32:,}")
    print(f"  対応する位置: {overflow_thread_int32 * thread_stride:,} bytes ({(overflow_thread_int32 * thread_stride)/1024/1024:.1f}MB)")
    
    print(f"\n32ビット符号なし整数でオーバーフローするスレッド番号: {overflow_thread_uint32:,}")
    print(f"  対応する位置: {overflow_thread_uint32 * thread_stride:,} bytes ({(overflow_thread_uint32 * thread_stride)/1024/1024:.1f}MB)")
    
    # 問題のスレッド番号での計算
    print("\n問題のスレッド番号での計算:")
    for tid in [1398101, 3404032]:
        product = tid * thread_stride
        print(f"Thread {tid}: {tid} * {thread_stride} = {product:,} (0x{product:08X})")
        print(f"  32ビットに収まる: {product <= max_uint32}")

def test_thread_id_calculation():
    """スレッドID計算の問題を検証"""
    print("\n=== スレッドID計算の検証 ===\n")
    
    # CUDAグリッド設定（仮定）
    threads_per_block = 256
    blocks_per_grid = 16384  # 仮の値
    
    # 問題のスレッド番号
    target_threads = [1398101, 1398102, 3404031, 3404032]
    
    for tid in target_threads:
        # ブロックとスレッドインデックスを逆算
        block_idx = tid // threads_per_block
        thread_idx = tid % threads_per_block
        
        print(f"Thread {tid}:")
        print(f"  ブロックインデックス: {block_idx}")
        print(f"  ブロック内スレッドインデックス: {thread_idx}")
        
        # 2次元グリッドの場合の計算
        grid_dim_y = 256  # 仮の値
        block_x = block_idx // grid_dim_y
        block_y = block_idx % grid_dim_y
        
        print(f"  2Dグリッド: blockIdx.x={block_x}, blockIdx.y={block_y}")
        print()

if __name__ == "__main__":
    test_exact_bug()
    analyze_multiplication_overflow()
    test_thread_id_calculation()