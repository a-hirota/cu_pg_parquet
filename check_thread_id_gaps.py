#!/usr/bin/env python3
"""thread_idのギャップを確認"""

import pandas as pd
import numpy as np

def check_thread_gaps():
    """thread_idで並べ替えてギャップを確認"""
    print("=== thread_idギャップ確認 ===\n")
    
    try:
        # チャンク0のみ読み込み（エラー回避のため）
        chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
        print(f"チャンク0: {len(chunk0):,}行読み込み成功")
    except Exception as e:
        print(f"エラー: {e}")
        return
    
    # thread_id列が存在するか確認
    if '_thread_id' not in chunk0.columns:
        print("\n⚠️ _thread_id列が存在しません")
        print(f"利用可能な列: {list(chunk0.columns)}")
        return
    
    # thread_id統計
    print(f"\n=== thread_id統計 ===")
    unique_threads = sorted(chunk0['_thread_id'].unique())
    print(f"ユニークthread_id数: {len(unique_threads):,}")
    print(f"thread_id範囲: {unique_threads[0]} - {unique_threads[-1]}")
    
    # thread_idのギャップを検出
    print(f"\n=== thread_idギャップ検出 ===")
    thread_gaps = []
    
    for i in range(1, len(unique_threads)):
        expected = unique_threads[i-1] + 1
        actual = unique_threads[i]
        
        if actual > expected:
            gap_size = actual - expected
            thread_gaps.append((unique_threads[i-1], actual, gap_size))
    
    print(f"thread_idギャップ数: {len(thread_gaps)}")
    
    if thread_gaps:
        print("\nthread_idギャップ（最初の10個）:")
        for prev, next, gap in thread_gaps[:10]:
            print(f"  Thread {prev} → Thread {next} (ギャップ: {gap})")
        
        # 最大のギャップ
        max_gap = max(thread_gaps, key=lambda x: x[2])
        print(f"\n最大のギャップ:")
        print(f"  Thread {max_gap[0]} → Thread {max_gap[1]} (ギャップ: {max_gap[2]:,})")
        
        # ギャップ内のthread_idが本当に存在しないか確認
        print(f"\n=== ギャップ内のthread_id存在確認 ===")
        for prev, next, gap in thread_gaps[:3]:  # 最初の3個のギャップ
            print(f"\nThread {prev} → Thread {next}:")
            missing_threads = list(range(prev + 1, next))
            print(f"  欠落thread_id数: {len(missing_threads)}")
            if len(missing_threads) <= 10:
                print(f"  欠落thread_id: {missing_threads}")
    else:
        print("\nthread_idギャップはありません（連続しています）")
    
    # thread_idごとの処理行数
    print(f"\n=== thread_idごとの処理行数 ===")
    thread_counts = chunk0['_thread_id'].value_counts().sort_index()
    
    # 0行処理のthread_idを探す（ギャップと関連）
    all_thread_ids = set(range(unique_threads[0], unique_threads[-1] + 1))
    existing_thread_ids = set(unique_threads)
    missing_thread_ids = all_thread_ids - existing_thread_ids
    
    print(f"\n存在しないthread_id数: {len(missing_thread_ids)}")
    if missing_thread_ids and len(missing_thread_ids) <= 20:
        print(f"存在しないthread_id: {sorted(missing_thread_ids)}")
    
    # c_custkeyの欠落との関連を確認
    if 'c_custkey' in chunk0.columns:
        print(f"\n=== c_custkeyの欠落確認 ===")
        chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
        
        # c_custkeyの欠落を確認
        custkeys = sorted(chunk0['c_custkey'].unique())
        custkey_missing = []
        
        # チャンク0の範囲（1〜6015104と仮定）
        for i in range(1, 6015105):
            if i not in custkeys:
                custkey_missing.append(i)
        
        print(f"c_custkey欠落数: {len(custkey_missing)}")
        
        if custkey_missing:
            print(f"\n欠落c_custkey（最初の10個）: {custkey_missing[:10]}")

if __name__ == "__main__":
    check_thread_gaps()