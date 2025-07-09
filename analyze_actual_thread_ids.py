#!/usr/bin/env python3
"""実際に記録されたthread_idを使って256MB境界問題を分析"""

import pandas as pd
import numpy as np

def analyze_thread_ids():
    """実際のthread_idデータを分析"""
    print("=== 実際のthread_idによる256MB境界分析 ===\n")
    
    # 複数のParquetファイルを確認
    parquet_files = ["output/customer_chunk_0_queue.parquet", "output/customer_chunk_1_queue.parquet"]
    
    for parquet_file in parquet_files:
        if not pd.io.common.file_exists(parquet_file):
            continue
            
        print(f"\n読み込み中: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        
        print(f"総行数: {len(df):,}")
        print(f"ユニークなthread_id数: {df['_thread_id'].nunique():,}")
        print(f"thread_idの分布: {df['_thread_id'].value_counts().head()}")
        
        # thread_idが0以外のデータがあるか確認
        non_zero_threads = df[df['_thread_id'] != 0]
        if len(non_zero_threads) > 0:
            print(f"thread_id != 0 の行数: {len(non_zero_threads)}")
            print(non_zero_threads[['c_custkey', '_thread_id', '_row_position']].head())
    
    # 最初のファイルで詳細分析
    df = pd.read_parquet(parquet_files[0])
    
    print(f"総行数: {len(df):,}")
    print(f"ユニークなthread_id数: {df['_thread_id'].nunique():,}")
    
    # Thread 1398101付近の確認
    print("\n1. Thread 1398101付近の実際のデータ:")
    thread_range = df[(df['_thread_id'] >= 1398099) & (df['_thread_id'] <= 1398103)]
    
    if len(thread_range) > 0:
        print(f"Thread 1398099-1398103の行数: {len(thread_range)}")
        print(thread_range[['c_custkey', '_thread_id', '_row_position', '_thread_start_pos', '_thread_end_pos']].to_string())
    
    # Thread 3404032付近の確認
    print("\n2. Thread 3404032付近の実際のデータ:")
    thread_range2 = df[(df['_thread_id'] >= 3404030) & (df['_thread_id'] <= 3404034)]
    
    if len(thread_range2) > 0:
        print(f"Thread 3404030-3404034の行数: {len(thread_range2)}")
        print(thread_range2[['c_custkey', '_thread_id', '_row_position', '_thread_start_pos', '_thread_end_pos']].to_string())
    
    # スレッドギャップの検出
    print("\n3. スレッドギャップの検出:")
    unique_threads = sorted(df['_thread_id'].unique())
    
    # 最大のギャップを見つける
    max_gap = 0
    max_gap_info = None
    
    for i in range(1, len(unique_threads)):
        gap = unique_threads[i] - unique_threads[i-1]
        if gap > max_gap:
            max_gap = gap
            max_gap_info = (unique_threads[i-1], unique_threads[i])
    
    if max_gap_info:
        print(f"最大のスレッドギャップ:")
        print(f"  Thread {max_gap_info[0]} → Thread {max_gap_info[1]}")
        print(f"  ギャップ: {max_gap:,}スレッド")
        
        # これらのスレッドの詳細
        print(f"\nThread {max_gap_info[0]}の詳細:")
        thread_data = df[df['_thread_id'] == max_gap_info[0]]
        print(f"  処理行数: {len(thread_data)}")
        print(f"  開始位置: 0x{thread_data['_thread_start_pos'].iloc[0]:08X}")
        print(f"  終了位置: 0x{thread_data['_thread_end_pos'].iloc[0]:08X}")
        
        print(f"\nThread {max_gap_info[1]}の詳細:")
        thread_data = df[df['_thread_id'] == max_gap_info[1]]
        print(f"  処理行数: {len(thread_data)}")
        print(f"  開始位置: 0x{thread_data['_thread_start_pos'].iloc[0]:08X}")
        print(f"  終了位置: 0x{thread_data['_thread_end_pos'].iloc[0]:08X}")
    
    # 欠落キー3483451の周辺を確認
    print("\n4. 欠落キー3483451の周辺確認:")
    # c_custkeyでソートして確認
    df_sorted = df.sort_values('c_custkey')
    
    # 3483451の前後を探す
    target_key = 3483451
    
    # 前の10行と後の10行を取得
    before_mask = df_sorted['c_custkey'] < target_key
    after_mask = df_sorted['c_custkey'] > target_key
    
    if before_mask.any() and after_mask.any():
        # 前の最後の10行
        before_rows = df_sorted[before_mask].tail(10)
        # 後の最初の10行
        after_rows = df_sorted[after_mask].head(10)
        
        print("欠落キー3483451の前10行:")
        print(before_rows[['c_custkey', '_thread_id', '_row_position']].to_string())
        
        print("\n欠落キー3483451の後10行:")
        print(after_rows[['c_custkey', '_thread_id', '_row_position']].to_string())
        
        # ギャップの確認
        last_before = before_rows.iloc[-1]['c_custkey']
        first_after = after_rows.iloc[0]['c_custkey']
        print(f"\nキーのギャップ: {last_before} → {first_after} (差: {first_after - last_before})")
    
    # 256MB境界の確認
    print("\n5. 256MB境界（0x10000000）付近のスレッド:")
    mb_256 = 0x10000000
    
    # 各スレッドの境界確認
    boundary_threads = []
    for thread_id in unique_threads:
        thread_data = df[df['_thread_id'] == thread_id]
        if len(thread_data) > 0:
            start_pos = thread_data['_thread_start_pos'].iloc[0]
            end_pos = thread_data['_thread_end_pos'].iloc[0]
            
            # 256MB境界を跨ぐか確認
            if start_pos < mb_256 <= end_pos:
                boundary_threads.append((thread_id, start_pos, end_pos))
    
    if boundary_threads:
        print("256MB境界を跨ぐスレッド:")
        for tid, start, end in boundary_threads:
            print(f"  Thread {tid}: 0x{start:08X} - 0x{end:08X}")
    else:
        print("256MB境界を跨ぐスレッドは見つかりませんでした")

if __name__ == "__main__":
    analyze_thread_ids()