#!/usr/bin/env python3
"""384MBと448MB境界での欠落パターンを詳細分析"""

import pandas as pd

def analyze_special_boundaries():
    """384MBと448MB境界での特殊なパターンを分析"""
    print("=== 384MBと448MB境界での欠落パターン分析 ===\n")
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # すべてのユニークなthread情報を取得
    unique_threads = chunk0[['_thread_id', '_thread_start_pos', '_thread_end_pos']].drop_duplicates('_thread_id')
    unique_threads = unique_threads.sort_values('_thread_id')
    
    # 64MB単位の境界
    mb_64 = 64 * 1024 * 1024
    mb_384 = 384 * 1024 * 1024
    mb_448 = 448 * 1024 * 1024
    
    # 384MB境界の分析
    print("384MB境界の詳細分析:")
    print("="*80)
    
    # 384MB境界前後のthreadを確認
    near_384mb = unique_threads[
        (unique_threads['_thread_start_pos'] > mb_384 - 2048) &
        (unique_threads['_thread_start_pos'] < mb_384 + 2048)
    ].sort_values('_thread_start_pos')
    
    print("\n384MB境界前後のthread:")
    prev_tid = None
    for _, row in near_384mb.iterrows():
        tid = int(row['_thread_id'])
        start = int(row['_thread_start_pos'])
        end = int(row['_thread_end_pos'])
        distance = start - mb_384
        
        # thread_idのギャップを確認
        gap_info = ""
        if prev_tid is not None and tid - prev_tid > 1:
            gap_info = f" [前threadとのギャップ: {tid - prev_tid}]"
        
        # 境界を跨ぐか確認
        crosses = ""
        if start < mb_384 <= end:
            crosses = " ⚠️ 384MB境界を跨ぐ"
        
        print(f"  thread_id {tid}: 境界から{distance:+6d}バイト (0x{start:08X}-0x{end:08X}){crosses}{gap_info}")
        prev_tid = tid
    
    # 320MB境界（384MBの前の64MB境界）も確認
    mb_320 = 320 * 1024 * 1024
    print("\n\n320MB境界（384MBの前の64MB境界）:")
    print("-"*80)
    
    near_320mb = unique_threads[
        (unique_threads['_thread_start_pos'] > mb_320 - 1024) &
        (unique_threads['_thread_start_pos'] < mb_320 + 1024)
    ].sort_values('_thread_start_pos')
    
    for _, row in near_320mb.iterrows():
        tid = int(row['_thread_id'])
        start = int(row['_thread_start_pos'])
        end = int(row['_thread_end_pos'])
        distance = start - mb_320
        
        crosses = ""
        if start < mb_320 <= end:
            crosses = " ⚠️ 320MB境界を跨ぐ"
        
        print(f"  thread_id {tid}: 境界から{distance:+6d}バイト{crosses}")
    
    # 448MB境界の分析
    print("\n\n448MB境界の詳細分析:")
    print("="*80)
    
    # 448MB境界前後のthreadを確認
    near_448mb = unique_threads[
        (unique_threads['_thread_start_pos'] > mb_448 - 2048) &
        (unique_threads['_thread_start_pos'] < mb_448 + 2048)
    ].sort_values('_thread_start_pos')
    
    print("\n448MB境界前後のthread:")
    prev_tid = None
    for _, row in near_448mb.iterrows():
        tid = int(row['_thread_id'])
        start = int(row['_thread_start_pos'])
        end = int(row['_thread_end_pos'])
        distance = start - mb_448
        
        # thread_idのギャップを確認
        gap_info = ""
        if prev_tid is not None and tid - prev_tid > 1:
            gap_info = f" [前threadとのギャップ: {tid - prev_tid}]"
        
        # 境界を跨ぐか確認
        crosses = ""
        if start < mb_448 <= end:
            crosses = " ⚠️ 448MB境界を跨ぐ"
        
        print(f"  thread_id {tid}: 境界から{distance:+6d}バイト (0x{start:08X}-0x{end:08X}){crosses}{gap_info}")
        prev_tid = tid
    
    # thread_idの数値パターンを確認
    print("\n\n=== thread_idの数値パターン ===")
    print("="*80)
    
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    for tid in missing_threads:
        print(f"\nthread_id {tid}:")
        print(f"  10進数: {tid}")
        print(f"  16進数: 0x{tid:06X}")
        print(f"  2進数: {bin(tid)}")
        
        # 特殊な値か確認
        if tid == 1048576:
            print(f"  = 2^20")
        elif tid == 2097152:
            print(f"  = 2^21")
        elif tid == 2097153:
            print(f"  = 2^21 + 1")
        
        # 349525（64MB内のthread数）との関係
        threads_per_64mb = 349525
        quotient = tid // threads_per_64mb
        remainder = tid % threads_per_64mb
        print(f"  = {quotient} * 349525 + {remainder}")
    
    # 実際のブロック最初のthread_idを確認
    print("\n\n=== 各64MBブロックの実際の最初のthread_id ===")
    print("="*80)
    
    for block_idx in range(10):  # 最初の10ブロック
        block_start = block_idx * mb_64
        
        # このブロックの最初のthread
        first_in_block = unique_threads[
            unique_threads['_thread_start_pos'] >= block_start
        ].head(1)
        
        if len(first_in_block) > 0:
            tid = int(first_in_block.iloc[0]['_thread_id'])
            start = int(first_in_block.iloc[0]['_thread_start_pos'])
            
            print(f"{block_idx * 64:3d}MB: thread_id {tid:7d} (開始位置: 0x{start:08X})")

if __name__ == "__main__":
    analyze_special_boundaries()