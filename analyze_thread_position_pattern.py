#!/usr/bin/env python3
"""thread位置の実際のパターンを分析"""

import pandas as pd
import numpy as np

def analyze_pattern():
    """thread位置の実際のパターンを分析"""
    print("=== thread位置パターン分析 ===\n")
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # すべてのユニークなthread位置を取得
    unique_threads = chunk0[['_thread_id', '_thread_start_pos', '_thread_end_pos']].drop_duplicates('_thread_id')
    unique_threads = unique_threads.sort_values('_thread_id')
    
    print(f"総thread数: {len(unique_threads):,}")
    
    # 各threadのサイズを計算
    unique_threads['size'] = unique_threads['_thread_end_pos'] - unique_threads['_thread_start_pos']
    
    # サイズの統計
    print(f"\nthread処理サイズ:")
    print(f"  一定値: {unique_threads['size'].iloc[0]}バイト")
    print(f"  全thread同一: {unique_threads['size'].nunique() == 1}")
    
    # 64MB境界を確認
    mb_64 = 64 * 1024 * 1024
    
    # start_posを64MBで割った商を計算
    unique_threads['mb_group'] = unique_threads['_thread_start_pos'] // mb_64
    
    # 各64MBグループのthread数を確認
    group_counts = unique_threads.groupby('mb_group').size()
    
    print(f"\n64MBグループごとのthread数:")
    for group, count in group_counts.head(10).items():
        start_mb = group * 64
        end_mb = (group + 1) * 64
        print(f"  {start_mb:4d}MB - {end_mb:4d}MB: {count:,} threads")
    
    # 欠落thread_idが属するはずの64MBグループを確認
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    print(f"\n欠落thread_idの予想64MBグループ:")
    
    # 実際のthread_idとstart_posの関係を推定
    # いくつかのサンプルから規則を探る
    sample_threads = unique_threads.head(1000)
    
    # thread_idと64MBグループの関係を分析
    for tid in missing_threads:
        # 前後のthread_idから推定
        before_tid = tid - 1
        after_tid = tid + 1
        
        before_info = unique_threads[unique_threads['_thread_id'] == before_tid]
        after_info = unique_threads[unique_threads['_thread_id'] == after_tid]
        
        if len(before_info) > 0 and len(after_info) > 0:
            before_group = before_info.iloc[0]['mb_group']
            after_group = after_info.iloc[0]['mb_group']
            
            print(f"\n  thread_id {tid}:")
            print(f"    前thread({before_tid}): {before_group * 64}MB グループ")
            print(f"    後thread({after_tid}): {after_group * 64}MB グループ")
            
            if before_group != after_group:
                print(f"    ⚠️ 64MBグループ境界を跨ぐ！")
    
    # 各64MBグループ内でのthread配置パターンを確認
    print(f"\n\n=== 64MBグループ内のthread配置パターン ===")
    
    # グループ3（192-256MB）の詳細
    group_3_threads = unique_threads[unique_threads['mb_group'] == 3].sort_values('_thread_id')
    
    print(f"\n192-256MBグループ（グループ3）:")
    print(f"  thread数: {len(group_3_threads)}")
    print(f"  thread_id範囲: {group_3_threads['_thread_id'].min()} - {group_3_threads['_thread_id'].max()}")
    
    # 最初と最後のいくつかを表示
    print(f"\n  最初の5thread:")
    for _, row in group_3_threads.head(5).iterrows():
        print(f"    thread_id={row['_thread_id']}, start=0x{row['_thread_start_pos']:08X}")
    
    print(f"\n  最後の5thread:")
    for _, row in group_3_threads.tail(5).iterrows():
        print(f"    thread_id={row['_thread_id']}, start=0x{row['_thread_start_pos']:08X}")
    
    # thread_idの連続性を確認
    thread_ids = group_3_threads['_thread_id'].values
    diffs = np.diff(thread_ids)
    
    if np.all(diffs == 1):
        print(f"\n  ✓ thread_idは連続している")
    else:
        gaps = np.where(diffs > 1)[0]
        print(f"\n  ⚠️ thread_idにギャップあり: {len(gaps)}箇所")
        for gap_idx in gaps[:5]:
            print(f"    {thread_ids[gap_idx]} → {thread_ids[gap_idx + 1]} (ギャップ: {diffs[gap_idx]})")
    
    # 1つの64MBに何thread入るか計算
    thread_size = 192  # バイト
    threads_per_64mb = mb_64 // thread_size
    
    print(f"\n\n理論値:")
    print(f"  1threadあたり: {thread_size}バイト")
    print(f"  64MBあたり: {threads_per_64mb:,} threads")
    print(f"  実際の最大値: {group_counts.max():,} threads")

if __name__ == "__main__":
    analyze_pattern()