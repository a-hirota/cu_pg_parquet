#!/usr/bin/env python3
"""
実際のParquetファイルに存在するスレッドIDの分布を分析
"""

import cudf
from pathlib import Path
import numpy as np

def analyze_thread_distribution():
    """実際に存在するスレッドIDを分析"""
    
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    all_thread_ids = []
    
    print("各チャンクのスレッドID分布:")
    print("="*80)
    
    for pf in parquet_files:
        print(f"\n{pf.name}:")
        df = cudf.read_parquet(pf, columns=['_thread_id'])
        
        # ユニークなスレッドIDを取得
        unique_threads = df['_thread_id'].unique().to_pandas()
        unique_threads = np.sort(unique_threads)
        
        print(f"  総行数: {len(df):,}")
        print(f"  ユニークスレッド数: {len(unique_threads):,}")
        print(f"  スレッドID範囲: {unique_threads[0]:,} - {unique_threads[-1]:,}")
        
        # 連続性をチェック
        expected_threads = np.arange(unique_threads[0], unique_threads[-1] + 1)
        missing_threads = np.setdiff1d(expected_threads, unique_threads)
        
        if len(missing_threads) > 0:
            print(f"  欠落スレッド数: {len(missing_threads):,}")
            
            # 大きなギャップを探す
            thread_diffs = np.diff(unique_threads)
            large_gaps = np.where(thread_diffs > 1000)[0]
            
            if len(large_gaps) > 0:
                print(f"  大きなギャップ（>1000）: {len(large_gaps)} 箇所")
                for i in large_gaps[:5]:  # 最初の5つ
                    gap_size = thread_diffs[i]
                    thread_before = unique_threads[i]
                    thread_after = unique_threads[i+1]
                    print(f"    - Thread {thread_before:,} → {thread_after:,} (ギャップ: {gap_size:,})")
        else:
            print(f"  → 連続しています")
        
        all_thread_ids.extend(unique_threads)
    
    # 全体の分析
    print("\n\n全体のスレッドID分析:")
    print("="*80)
    
    all_thread_ids = np.unique(all_thread_ids)
    print(f"総ユニークスレッド数: {len(all_thread_ids):,}")
    print(f"スレッドID範囲: {all_thread_ids[0]:,} - {all_thread_ids[-1]:,}")
    
    # 大きなギャップを詳しく分析
    thread_diffs = np.diff(all_thread_ids)
    large_gap_indices = np.where(thread_diffs > 1000)[0]
    
    print(f"\n大きなギャップ（>1000）の詳細:")
    for idx in large_gap_indices:
        thread_before = all_thread_ids[idx]
        thread_after = all_thread_ids[idx + 1]
        gap = thread_after - thread_before
        
        # ブロック番号を計算
        block_before = thread_before // 256
        block_after = thread_after // 256
        
        print(f"\n{thread_before:,} → {thread_after:,}")
        print(f"  ギャップ: {gap:,} スレッド")
        print(f"  ブロック: {block_before:,} → {block_after:,} (差: {block_after - block_before:,})")
        
        # 16進数でパターンを確認
        print(f"  16進数: 0x{thread_before:X} → 0x{thread_after:X}")
        
        # ビットパターンを確認
        if thread_before == 1048575:  # 0xFFFFF
            print(f"  → 20ビット境界（0xFFFFF = 2^20 - 1）")
        if thread_before == 2097151:  # 0x1FFFFF
            print(f"  → 21ビット境界（0x1FFFFF = 2^21 - 1）")

def check_16_parallel_pattern():
    """16並列処理のパターンを確認"""
    print("\n\n16並列処理パターンの確認:")
    print("="*80)
    
    # チャンクごとの開始スレッドIDを確認
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    chunk_info = []
    for i, pf in enumerate(parquet_files):
        df = cudf.read_parquet(pf, columns=['_thread_id'])
        first_thread = int(df['_thread_id'].iloc[0])
        
        # 最後の行も確認
        df_last = cudf.read_parquet(pf, columns=['_thread_id'])
        last_thread = int(df_last['_thread_id'].iloc[-1])
        
        chunk_info.append((i, first_thread, last_thread))
        print(f"チャンク {i}: Thread {first_thread:,} - {last_thread:,}")
    
    # パターンを分析
    print("\n開始スレッドIDのパターン:")
    for i in range(1, len(chunk_info)):
        prev_last = chunk_info[i-1][2]
        curr_first = chunk_info[i][1]
        gap = curr_first - prev_last
        print(f"  チャンク{i-1}の最後({prev_last:,}) → チャンク{i}の最初({curr_first:,}): ギャップ {gap:,}")

if __name__ == "__main__":
    analyze_thread_distribution()
    check_16_parallel_pattern()