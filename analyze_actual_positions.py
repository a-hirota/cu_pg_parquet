#!/usr/bin/env python3
"""実際のthread位置パターンを分析"""

import pandas as pd
import numpy as np

def analyze_actual_positions():
    """実際のthread位置パターンを分析"""
    print("=== 実際のthread位置パターン分析 ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # 実際の位置パターンを確認
    print("実際のthread_start_pos分析:")
    print("-" * 80)
    
    # 欠落thread_id周辺の実際の位置を詳しく見る
    for missing_tid in missing_threads:
        print(f"\n欠落thread_id {missing_tid} (0x{missing_tid:08X})周辺:")
        
        # 前後10個のthread_idを確認
        for offset in range(-5, 6):
            tid = missing_tid + offset
            if tid in chunk0['_thread_id'].values:
                rows = chunk0[chunk0['_thread_id'] == tid]
                if len(rows) > 0:
                    row = rows.iloc[0]
                    start_pos = row['_thread_start_pos']
                    end_pos = row['_thread_end_pos']
                    size = end_pos - start_pos
                    
                    # 特殊な境界を確認
                    mb_pos = start_pos / (1024 * 1024)
                    
                    mark = ""
                    if start_pos % (64 * 1024 * 1024) == 0:
                        mark = " ⚠️ 64MB境界"
                    elif start_pos % (256 * 1024 * 1024) == 0:
                        mark = " ⚠️ 256MB境界"
                    
                    print(f"  Thread {tid}: start=0x{start_pos:08X} ({mb_pos:7.2f}MB), size={size:4d}バイト{mark}")
            else:
                if tid == missing_tid:
                    print(f"  Thread {tid}: *** 欠落 ***")
    
    # 実際のthread_strideパターンを分析
    print("\n\n=== 実際のthread間隔パターン ===")
    
    # いくつかのthread_idグループで間隔を確認
    test_ranges = [
        (1048570, 1048585),
        (1398095, 1398110),
        (2097145, 2097160),
        (2446670, 2446685)
    ]
    
    for start_tid, end_tid in test_ranges:
        print(f"\nthread_id {start_tid}〜{end_tid}:")
        
        positions = []
        for tid in range(start_tid, end_tid):
            if tid in chunk0['_thread_id'].values:
                rows = chunk0[chunk0['_thread_id'] == tid]
                if len(rows) > 0:
                    positions.append((tid, rows.iloc[0]['_thread_start_pos']))
        
        if len(positions) > 1:
            # 位置の差分を計算
            for i in range(1, len(positions)):
                tid1, pos1 = positions[i-1]
                tid2, pos2 = positions[i]
                
                # thread_id差分
                tid_diff = tid2 - tid1
                # 位置差分
                pos_diff = pos2 - pos1
                
                if tid_diff == 1:
                    print(f"  {tid1} → {tid2}: 位置差={pos_diff:4d}バイト (0x{pos_diff:04X})")
                else:
                    print(f"  {tid1} → {tid2}: thread_idギャップ={tid_diff}, 位置差={pos_diff:,}バイト")
    
    # 64MB境界での特殊な処理を確認
    print("\n\n=== 境界での位置パターン ===")
    
    mb_64 = 64 * 1024 * 1024
    mb_256 = 256 * 1024 * 1024
    
    # 各境界付近のthread_idを探す
    boundaries = []
    
    # すべてのユニークなstart_posを取得
    unique_positions = chunk0.groupby('_thread_start_pos').first().reset_index()
    
    for _, row in unique_positions.iterrows():
        start_pos = row['_thread_start_pos']
        tid = row['_thread_id']
        
        # 64MB境界付近（前後1MB）
        for i in range(10):  # 最初の10個の64MB境界
            boundary = i * mb_64
            if abs(start_pos - boundary) < 1024 * 1024:  # 1MB以内
                boundaries.append({
                    'tid': tid,
                    'pos': start_pos,
                    'boundary': boundary,
                    'boundary_type': '64MB',
                    'distance': start_pos - boundary
                })
    
    # 境界付近のパターンを表示
    boundaries.sort(key=lambda x: x['boundary'])
    
    print("\n境界付近のthread:")
    for b in boundaries[:20]:  # 最初の20個
        distance_mb = b['distance'] / (1024 * 1024)
        print(f"  {b['boundary_type']}境界 {b['boundary']/(1024*1024):.0f}MB: "
              f"thread_id={b['tid']}, 位置=0x{b['pos']:08X}, "
              f"境界からの距離={distance_mb:+.2f}MB")

if __name__ == "__main__":
    analyze_actual_positions()