#!/usr/bin/env python3
"""256MB境界でのCUDAカーネルスレッドスキップ問題の再現テスト"""

import cupy as cp
import numpy as np
from numba import cuda
import math

# カーネル定義
@cuda.jit
def detect_rows_kernel(data, offsets, n_offsets):
    """行オフセットを検出するカーネル（オリジナルの実装を簡略化）"""
    thread_id = cuda.grid(1)
    
    if thread_id >= n_offsets:
        return
    
    # 192バイトストライドでのアドレス計算
    start_pos = thread_id * 192
    
    # 256MB境界チェック
    if start_pos >= 0x10000000:  # 256MB = 268435456
        # オリジナルと同じ問題を再現するため、ここで何もしない
        return
    
    # 行の検出ロジック（簡略版）
    if start_pos < len(data):
        offsets[thread_id] = start_pos

def test_256mb_boundary():
    """256MB境界でのスレッドスキップを再現"""
    print("=== 256MB境界スレッドスキップ再現テスト ===\n")
    
    # 400MBのデータを作成（256MB境界を跨ぐ）
    data_size = 400 * 1024 * 1024  # 400MB
    data = cp.zeros(data_size, dtype=cp.uint8)
    
    # スレッド数計算（192バイトストライド）
    stride = 192
    n_threads = (data_size + stride - 1) // stride
    print(f"データサイズ: {data_size:,} bytes ({data_size/1024/1024:.1f}MB)")
    print(f"ストライド: {stride} bytes")
    print(f"必要スレッド数: {n_threads:,}")
    
    # 256MB境界の計算
    boundary_256mb = 0x10000000  # 268,435,456
    threads_before_boundary = boundary_256mb // stride
    threads_after_boundary = n_threads - threads_before_boundary
    
    print(f"\n256MB境界 (0x{boundary_256mb:08X}):")
    print(f"  境界前スレッド数: {threads_before_boundary:,}")
    print(f"  境界後スレッド数: {threads_after_boundary:,}")
    
    # オフセット配列（-1で初期化）
    offsets = cp.full(n_threads, -1, dtype=cp.int64)
    
    # カーネル実行（256MBで分割）
    threads_per_block = 256
    
    # 256MB境界前のカーネル
    blocks1 = (threads_before_boundary + threads_per_block - 1) // threads_per_block
    print(f"\n第1カーネル: ブロック数={blocks1}, スレッド数={threads_before_boundary}")
    detect_rows_kernel[blocks1, threads_per_block](data, offsets, threads_before_boundary)
    
    # 256MB境界後のカーネル（問題のある実装を再現）
    # オリジナルコードでは、ここでスレッドIDがリセットされず、
    # 大きなギャップが生じる
    blocks2 = (threads_after_boundary + threads_per_block - 1) // threads_per_block
    print(f"第2カーネル: ブロック数={blocks2}, スレッド数={threads_after_boundary}")
    
    # 問題の再現：スレッドIDが正しく計算されない
    # （本来はthreads_before_boundaryからの続きであるべき）
    
    # 結果確認
    offsets_host = offsets.get()
    
    # 有効なオフセットをカウント
    valid_offsets = offsets_host[offsets_host >= 0]
    print(f"\n検出された行数: {len(valid_offsets):,}")
    
    # ギャップを検出
    print("\n=== ギャップ分析 ===")
    gap_found = False
    for i in range(1, len(offsets_host)):
        if offsets_host[i-1] >= 0 and offsets_host[i] == -1:
            # ギャップの開始
            gap_start = i
            gap_start_addr = i * stride
            
            # ギャップの終了を探す
            gap_end = gap_start
            while gap_end < len(offsets_host) and offsets_host[gap_end] == -1:
                gap_end += 1
            
            if gap_end < len(offsets_host):
                gap_size = gap_end - gap_start
                gap_end_addr = gap_end * stride
                
                print(f"\nギャップ検出:")
                print(f"  スレッド {gap_start-1}: 0x{(gap_start-1)*stride:08X} (最後の処理)")
                print(f"  スレッド {gap_start} - {gap_end-1}: 未処理（{gap_size:,}スレッド）")
                print(f"  スレッド {gap_end}: 0x{gap_end_addr:08X} (次の処理)")
                print(f"  ギャップサイズ: {gap_size*stride:,} bytes ({gap_size*stride/1024/1024:.1f}MB)")
                
                gap_found = True
                break
    
    if not gap_found:
        print("ギャップは検出されませんでした")
    
    # 256MB境界付近の詳細
    print("\n=== 256MB境界付近の詳細 ===")
    boundary_thread = threads_before_boundary - 1
    for i in range(max(0, boundary_thread-5), min(len(offsets_host), boundary_thread+5)):
        if i < len(offsets_host):
            addr = i * stride
            status = "処理済" if offsets_host[i] >= 0 else "未処理"
            marker = " ← 256MB境界" if addr == boundary_256mb else ""
            print(f"スレッド {i}: 0x{addr:08X} ({status}){marker}")

def simulate_original_bug():
    """オリジナルのバグを詳細にシミュレート"""
    print("\n=== オリジナルバグのシミュレーション ===\n")
    
    # オリジナルのスレッド番号
    thread_1398101 = 1398101
    thread_3404032 = 3404032
    stride = 192
    
    # アドレス計算
    addr_1398101 = thread_1398101 * stride
    addr_3404032 = thread_3404032 * stride
    
    print(f"Thread {thread_1398101}:")
    print(f"  アドレス: 0x{addr_1398101:08X} ({addr_1398101:,} bytes)")
    print(f"  = {addr_1398101/1024/1024:.2f}MB")
    
    print(f"\nThread {thread_3404032}:")
    print(f"  アドレス: 0x{addr_3404032:08X} ({addr_3404032:,} bytes)")
    print(f"  = {addr_3404032/1024/1024:.2f}MB")
    
    # ギャップ分析
    gap_threads = thread_3404032 - thread_1398101 - 1
    gap_bytes = (thread_3404032 - thread_1398101) * stride
    
    print(f"\nギャップ:")
    print(f"  スキップされたスレッド数: {gap_threads:,}")
    print(f"  スキップされたバイト数: {gap_bytes:,} ({gap_bytes/1024/1024:.2f}MB)")
    
    # ビット28の確認
    print(f"\nビット28の状態:")
    print(f"  Thread {thread_1398101}: bit28 = {(addr_1398101 >> 28) & 1}")
    print(f"  Thread {thread_3404032}: bit28 = {(addr_3404032 >> 28) & 1}")

if __name__ == "__main__":
    test_256mb_boundary()
    simulate_original_bug()