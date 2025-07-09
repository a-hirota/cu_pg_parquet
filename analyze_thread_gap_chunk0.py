#!/usr/bin/env python3
"""thread_idギャップと欠落キーの相関を分析（チャンク0のみ）"""

import pandas as pd
import numpy as np

def analyze_correlation():
    """thread_idギャップと欠落キーの相関を分析"""
    print("=== thread_idギャップと欠落キーの相関分析（チャンク0） ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # 欠落キー（チャンク0のみ）
    missing_keys_chunk0 = [476029, 1227856, 2731633, 3483451, 4235332, 4987296, 5739177]
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # Decimal型をintに変換
    chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
    
    print(f"チャンク0: {len(chunk0):,}行")
    print(f"\n欠落thread_id: {missing_threads}")
    print(f"チャンク0の欠落キー数: {len(missing_keys_chunk0)}")
    print(f"チャンク0の欠落キー: {missing_keys_chunk0}")
    
    # thread_idの分布を確認
    print("\n=== thread_id分布 ===")
    thread_counts = chunk0['_thread_id'].value_counts()
    print(f"ユニークthread_id数: {len(thread_counts):,}")
    print(f"thread_id範囲: {chunk0['_thread_id'].min()} - {chunk0['_thread_id'].max()}")
    
    # 欠落thread_id周辺の分析
    print("\n=== 欠落thread_id周辺の分析 ===")
    
    for missing_thread in missing_threads:
        print(f"\n欠落thread_id: {missing_thread}")
        
        # 前後のthread_idで処理されたキーを確認
        before_thread = missing_thread - 1
        after_thread = missing_thread + 1
        
        # 実際に存在するthread_idを確認
        if before_thread in chunk0['_thread_id'].values:
            before_rows = chunk0[chunk0['_thread_id'] == before_thread]
            print(f"  前のthread_id {before_thread}: {len(before_rows)}行")
            if len(before_rows) > 0:
                keys = sorted(before_rows['c_custkey'].unique())
                print(f"    キー範囲: {keys[0]} - {keys[-1]}")
                # 最後の数個のキーを表示
                print(f"    最後の5キー: {keys[-5:]}")
        
        if after_thread in chunk0['_thread_id'].values:
            after_rows = chunk0[chunk0['_thread_id'] == after_thread]
            print(f"  後のthread_id {after_thread}: {len(after_rows)}行")
            if len(after_rows) > 0:
                keys = sorted(after_rows['c_custkey'].unique())
                print(f"    キー範囲: {keys[0]} - {keys[-1]}")
                # 最初の数個のキーを表示
                print(f"    最初の5キー: {keys[:5]}")
        
        # より広い範囲で探す
        print(f"\n  近傍のthread_id:")
        for offset in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
            tid = missing_thread + offset
            if tid in chunk0['_thread_id'].values:
                rows = chunk0[chunk0['_thread_id'] == tid]
                keys = sorted(rows['c_custkey'].unique())
                if keys:
                    print(f"    Thread {tid}: {len(rows)}行, キー範囲: {keys[0]} - {keys[-1]}")
    
    # 欠落キー周辺のthread_idを確認
    print("\n\n=== 欠落キー周辺のthread_id ===")
    
    for missing_key in missing_keys_chunk0:
        print(f"\n欠落キー {missing_key}:")
        
        # 前後のキーを探す
        before_key = missing_key - 1
        after_key = missing_key + 1
        
        # 前のキーのthread_id
        before_rows = chunk0[chunk0['c_custkey'] == before_key]
        if len(before_rows) > 0:
            row = before_rows.iloc[0]
            print(f"  前(key={before_key}): thread_id={row['_thread_id']}, pos=0x{row['_row_position']:08X}")
        
        # 後のキーのthread_id
        after_rows = chunk0[chunk0['c_custkey'] == after_key]
        if len(after_rows) > 0:
            row = after_rows.iloc[0]
            print(f"  後(key={after_key}): thread_id={row['_thread_id']}, pos=0x{row['_row_position']:08X}")
        
        # thread_idギャップがあるか確認
        if len(before_rows) > 0 and len(after_rows) > 0:
            thread_before = before_rows.iloc[0]['_thread_id']
            thread_after = after_rows.iloc[0]['_thread_id']
            
            if thread_after - thread_before > 1:
                print(f"  ⚠️ thread_idギャップ: {thread_before} → {thread_after} (差: {thread_after - thread_before})")
                
                # ギャップ内に欠落thread_idがあるか確認
                gap_missing_threads = [t for t in missing_threads if thread_before < t < thread_after]
                if gap_missing_threads:
                    print(f"    ギャップ内の欠落thread_id: {gap_missing_threads}")
    
    # thread_stride分析
    print("\n\n=== thread_stride分析 ===")
    
    # 各thread_idが処理している行の位置パターンを確認
    sample_threads = [0, 1, 2, 100, 1000, 10000]
    
    for tid in sample_threads:
        if tid in chunk0['_thread_id'].values:
            rows = chunk0[chunk0['_thread_id'] == tid]
            positions = sorted(rows['_row_position'].values)
            
            if len(positions) > 1:
                # 位置の差分を計算
                diffs = np.diff(positions)
                unique_diffs = np.unique(diffs)
                
                print(f"\nThread {tid}: {len(positions)}行")
                print(f"  位置差分のユニーク値数: {len(unique_diffs)}")
                if len(unique_diffs) <= 5:
                    print(f"  位置差分: {unique_diffs}")
                else:
                    print(f"  位置差分（最初の5個）: {unique_diffs[:5]}")
                    print(f"  最頻値: {np.bincount(diffs).argmax()}")

if __name__ == "__main__":
    analyze_correlation()