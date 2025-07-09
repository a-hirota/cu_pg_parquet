#!/usr/bin/env python3
"""thread_idギャップパターンを検証"""

import pandas as pd
import numpy as np

def verify_pattern():
    """欠落パターンを検証"""
    print("=== thread_idギャップパターン検証 ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # 欠落キーと周辺のthread_id
    missing_key_info = [
        {"key": 476029, "before_tid": 2446678, "after_tid": 4012015, "gap_threads": [2446679]},
        {"key": 1227856, "before_tid": 2796203, "after_tid": 4214620, "gap_threads": []},
        {"key": 2731633, "before_tid": 349524, "after_tid": 2796205, "gap_threads": [1048576, 1398102, 2097153, 2446679]},
        {"key": 3483451, "before_tid": 1747627, "after_tid": 3606755, "gap_threads": [2097153, 2446679]},
        {"key": 4235332, "before_tid": 699050, "after_tid": 2998798, "gap_threads": [1048576, 1398102, 2097153, 2446679]},
        {"key": 4987296, "before_tid": 2097152, "after_tid": 3809386, "gap_threads": [2097153, 2446679]},
        {"key": 5739177, "before_tid": 1398101, "after_tid": 3404093, "gap_threads": [1398102, 2097153, 2446679]}
    ]
    
    # パターンを分析
    print("欠落キーと欠落thread_idの関係:")
    print("-" * 80)
    
    for info in missing_key_info:
        print(f"\n欠落キー {info['key']}:")
        print(f"  thread_idギャップ: {info['before_tid']} → {info['after_tid']} (差: {info['after_tid'] - info['before_tid']})")
        
        if info['gap_threads']:
            print(f"  ギャップ内の欠落thread_id: {info['gap_threads']}")
            print(f"  ⚠️ 欠落thread_idが原因の可能性大")
        else:
            print(f"  ギャップ内に欠落thread_idなし")
            print(f"  ❓ 別の原因？")
    
    # 欠落thread_idの特徴を分析
    print("\n\n=== 欠落thread_idの特徴分析 ===")
    
    for tid in missing_threads:
        print(f"\nthread_id {tid}:")
        # 16進数表記
        print(f"  16進数: 0x{tid:08X}")
        # ビット表現
        print(f"  ビット: {bin(tid)}")
        # 2のべき乗か確認
        if tid & (tid - 1) == 0:
            print(f"  ⚠️ 2のべき乗: 2^{int(np.log2(tid))}")
        
        # 64MBごとの境界か確認
        mb_64 = 64 * 1024 * 1024
        if tid * 1150 % mb_64 == 0:
            print(f"  ⚠️ 64MB境界に該当")
    
    # thread_id計算の検証
    print("\n\n=== thread_id計算検証 ===")
    
    # CUDAカーネルでのthread_id計算をシミュレート
    grid_dim_x = 192  # threads_per_block_x
    grid_dim_y = 23040  # grid_2d.y
    
    print(f"グリッド設定: ({grid_dim_x}, {grid_dim_y})")
    print(f"総スレッド数: {grid_dim_x * grid_dim_y:,}")
    
    # 欠落thread_idがどのブロックに属するか
    for tid in missing_threads:
        block_y = tid // grid_dim_x
        thread_x = tid % grid_dim_x
        
        print(f"\nthread_id {tid}:")
        print(f"  ブロック位置: blockIdx.y = {block_y}, threadIdx.x = {thread_x}")
        
        # 特殊な位置か確認
        if thread_x == 0:
            print(f"  ⚠️ ブロックの最初のスレッド")
        if thread_x == grid_dim_x - 1:
            print(f"  ⚠️ ブロックの最後のスレッド")
    
    # メモリアライメントの確認
    print("\n\n=== メモリアライメント分析 ===")
    
    thread_stride = 1150  # バイト
    
    for tid in missing_threads:
        start_pos = tid * thread_stride
        
        print(f"\nthread_id {tid}:")
        print(f"  開始位置: 0x{start_pos:08X} ({start_pos/(1024*1024):.2f}MB)")
        
        # アライメントチェック
        alignments = [256, 512, 1024, 4096, 65536, 1048576]
        for align in alignments:
            if start_pos % align == 0:
                print(f"  ✓ {align}バイトアライメント")

if __name__ == "__main__":
    verify_pattern()