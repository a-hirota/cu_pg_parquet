#!/usr/bin/env python3
"""新しく作成されたthread_id付きParquetファイルを分析"""

import pandas as pd
import numpy as np

def analyze_with_thread_ids():
    """thread_id付きParquetファイルを分析"""
    print("=== thread_id付きParquetファイル分析 ===\n")
    
    # Parquetファイルを読み込み
    try:
        chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
        chunk1 = pd.read_parquet("output/customer_chunk_1_queue.parquet")
    except FileNotFoundError:
        print("エラー: Parquetファイルが見つかりません")
        print("現在のディレクトリ:")
        import os
        print(os.getcwd())
        print("\noutputディレクトリの内容:")
        if os.path.exists("output"):
            for f in os.listdir("output"):
                print(f"  {f}")
        return
    
    # Decimal型をintに変換
    chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
    chunk1['c_custkey'] = chunk1['c_custkey'].astype('int64')
    
    print(f"チャンク0: {len(chunk0):,}行")
    print(f"チャンク1: {len(chunk1):,}行")
    print(f"\nチャンク0の列: {list(chunk0.columns)}")
    
    # thread_id統計
    if '_thread_id' in chunk0.columns:
        print(f"\n=== thread_id統計 ===")
        print(f"チャンク0:")
        print(f"  ユニークthread_id数: {chunk0['_thread_id'].nunique():,}")
        print(f"  thread_id範囲: {chunk0['_thread_id'].min()} - {chunk0['_thread_id'].max()}")
        
        # thread_idが0以外の値を持つか確認
        non_zero = chunk0[chunk0['_thread_id'] != 0]
        print(f"  thread_id != 0 の行数: {len(non_zero):,} ({len(non_zero)/len(chunk0)*100:.1f}%)")
        
        if len(non_zero) > 0:
            print("\n  thread_id分布（上位10）:")
            for tid, count in chunk0['_thread_id'].value_counts().head(10).items():
                print(f"    Thread {tid}: {count:,}行")
    else:
        print("\n⚠️ _thread_id列が存在しません")
        return
    
    # 欠落キーの確認
    print("\n=== 欠落キーの確認 ===")
    
    # 全キーを取得
    all_keys = set(chunk0['c_custkey']) | set(chunk1['c_custkey'])
    
    # 欠落キーを探す
    missing_keys = []
    for i in range(1, 12030001):
        if i not in all_keys:
            missing_keys.append(i)
    
    print(f"欠落キー数: {len(missing_keys)}")
    
    if missing_keys:
        print("\n欠落キー:")
        for key in missing_keys[:20]:
            print(f"  {key}")
        
        # 欠落キー周辺のthread_idを確認
        print("\n=== 欠落キー周辺のthread_id ===")
        
        for missing_key in missing_keys[:5]:  # 最初の5個
            # どちらのチャンクに属するか
            chunk = chunk0 if missing_key < 6015105 else chunk1
            
            # 前後のキーを探す
            before = chunk[chunk['c_custkey'] == missing_key - 1]
            after = chunk[chunk['c_custkey'] == missing_key + 1]
            
            print(f"\n欠落キー {missing_key}:")
            if len(before) > 0:
                row = before.iloc[0]
                print(f"  前(key={missing_key-1}): thread_id={row['_thread_id']}, pos=0x{row['_row_position']:08X}")
                if '_thread_start_pos' in row:
                    print(f"    スレッド範囲: 0x{row['_thread_start_pos']:08X}-0x{row['_thread_end_pos']:08X}")
            
            if len(after) > 0:
                row = after.iloc[0]
                print(f"  後(key={missing_key+1}): thread_id={row['_thread_id']}, pos=0x{row['_row_position']:08X}")
                if '_thread_start_pos' in row:
                    print(f"    スレッド範囲: 0x{row['_thread_start_pos']:08X}-0x{row['_thread_end_pos']:08X}")
            
            if len(before) > 0 and len(after) > 0:
                thread_before = before.iloc[0]['_thread_id']
                thread_after = after.iloc[0]['_thread_id']
                
                if thread_before != thread_after:
                    print(f"  ⚠️ 異なるスレッドで処理: Thread {thread_before} → Thread {thread_after}")
                    
                    # そのスレッド間にギャップがあるか確認
                    gap = thread_after - thread_before
                    if gap > 1:
                        print(f"    スレッドギャップ: {gap}")
                        
                        # 間のスレッドが存在するか確認
                        for tid in range(thread_before + 1, thread_after):
                            if tid in chunk['_thread_id'].values:
                                tid_rows = chunk[chunk['_thread_id'] == tid]
                                print(f"    Thread {tid}: {len(tid_rows)}行処理")
                            else:
                                print(f"    Thread {tid}: 処理なし ← 問題の可能性")

if __name__ == "__main__":
    analyze_with_thread_ids()