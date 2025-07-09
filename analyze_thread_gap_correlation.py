#!/usr/bin/env python3
"""thread_idギャップと欠落キーの相関を分析"""

import pandas as pd
import numpy as np

def analyze_correlation():
    """thread_idギャップと欠落キーの相関を分析"""
    print("=== thread_idギャップと欠落キーの相関分析 ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # 欠落キー
    missing_keys = [476029, 1227856, 2731633, 3483451, 4235332, 4987296, 5739177,
                    6491094, 7243028, 7994887, 8746794, 9498603, 10250384, 11754161]
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    chunk1 = pd.read_parquet("output/customer_chunk_1_queue.parquet")
    
    # Decimal型をintに変換
    chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
    chunk1['c_custkey'] = chunk1['c_custkey'].astype('int64')
    
    print(f"チャンク0: {len(chunk0):,}行")
    print(f"チャンク1: {len(chunk1):,}行")
    print(f"\n欠落thread_id: {missing_threads}")
    print(f"欠落キー数: {len(missing_keys)}")
    
    # 欠落thread_id周辺のthread_idを確認
    print("\n=== 欠落thread_id周辺の分析 ===")
    
    all_data = pd.concat([chunk0, chunk1], ignore_index=True)
    
    for missing_thread in missing_threads:
        print(f"\n欠落thread_id: {missing_thread}")
        
        # 前後のthread_idを探す
        before_thread = missing_thread - 1
        after_thread = missing_thread + 1
        
        # 実際に存在する最も近いthread_idを探す
        existing_threads = sorted(all_data['_thread_id'].unique())
        
        # before_threadから遡って存在するthread_idを探す
        actual_before = None
        for tid in range(missing_thread - 1, -1, -1):
            if tid in existing_threads:
                actual_before = tid
                break
        
        # after_threadから進んで存在するthread_idを探す
        actual_after = None
        for tid in range(missing_thread + 1, max(existing_threads) + 1):
            if tid in existing_threads:
                actual_after = tid
                break
        
        if actual_before is not None:
            before_rows = all_data[all_data['_thread_id'] == actual_before]
            print(f"  前のthread_id {actual_before}: {len(before_rows)}行")
            if len(before_rows) > 0:
                keys = sorted(before_rows['c_custkey'].unique())
                print(f"    キー範囲: {keys[0]} - {keys[-1]}")
        
        if actual_after is not None:
            after_rows = all_data[all_data['_thread_id'] == actual_after]
            print(f"  後のthread_id {actual_after}: {len(after_rows)}行")
            if len(after_rows) > 0:
                keys = sorted(after_rows['c_custkey'].unique())
                print(f"    キー範囲: {keys[0]} - {keys[-1]}")
        
        # このthread_idギャップの間に欠落キーがあるか確認
        if actual_before is not None and actual_after is not None:
            before_max_key = before_rows['c_custkey'].max()
            after_min_key = after_rows['c_custkey'].min()
            
            # この範囲内の欠落キーを探す
            keys_in_gap = [k for k in missing_keys if before_max_key < k < after_min_key]
            if keys_in_gap:
                print(f"  ⚠️ このthread_idギャップ内の欠落キー: {keys_in_gap}")
    
    # thread_idごとの処理行数統計
    print("\n\n=== thread_id処理行数統計 ===")
    thread_counts = all_data['_thread_id'].value_counts()
    
    # ゼロ行処理のthread_idは既に特定済み（missing_threads）
    print(f"0行処理のthread_id数: {len(missing_threads)}")
    print(f"0行処理のthread_id: {missing_threads}")
    
    # 1行しか処理していないthread_idを探す
    single_row_threads = thread_counts[thread_counts == 1]
    print(f"\n1行だけ処理のthread_id数: {len(single_row_threads)}")
    
    # thread_stride（1150バイト）を確認
    print("\n\n=== thread_strideパターン分析 ===")
    thread_stride = 1150  # バイト
    
    # 各欠落thread_idのストライドパターンを分析
    for i, missing_thread in enumerate(missing_threads):
        print(f"\n欠落thread_id {missing_thread}:")
        
        # このthread_idが処理すべきだった行位置を計算
        expected_positions = []
        for row_idx in range(12030000):  # 全行数
            if row_idx % len(existing_threads) == missing_thread:
                expected_positions.append(row_idx)
        
        print(f"  期待される処理行数: {len(expected_positions)}")
        
        # 実際の欠落キーとの対応を確認
        if i < len(missing_keys) // len(missing_threads):
            related_keys = missing_keys[i * (len(missing_keys) // len(missing_threads)):(i + 1) * (len(missing_keys) // len(missing_threads))]
            print(f"  関連する欠落キー: {related_keys}")

if __name__ == "__main__":
    analyze_correlation()