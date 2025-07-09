#!/usr/bin/env python3
"""Parquetファイルに記録されたthread_idを確認"""

import pandas as pd
import pyarrow.parquet as pq

def check_thread_ids():
    """実際のParquetファイルのthread_id情報を確認"""
    print("=== Parquetファイルのthread_id確認 ===\n")
    
    # Parquetファイルを読み込み
    parquet_file = "output/customer_chunk_0_queue.parquet"
    
    # まずスキーマを確認
    print(f"1. {parquet_file}のスキーマ確認:")
    parquet_file_obj = pq.ParquetFile(parquet_file)
    schema = parquet_file_obj.schema_arrow
    print("列名とデータ型:")
    for field in schema:
        print(f"  - {field.name}: {field.type}")
    
    # データを読み込み（最初の100行だけ）
    print("\n2. データの最初の100行を確認:")
    df = pd.read_parquet(parquet_file, columns=None).head(100)
    
    # thread_id関連の列を探す
    thread_cols = [col for col in df.columns if 'thread' in col.lower() or 'tid' in col.lower()]
    print(f"\nthread関連の列: {thread_cols}")
    
    # _row_positionからthread_idを計算
    if '_row_position' in df.columns:
        HEADER_SIZE = 19
        THREAD_STRIDE = 192
        df['calculated_thread_id'] = ((df['_row_position'] - HEADER_SIZE) // THREAD_STRIDE).astype(int)
        
        print("\n3. _row_positionから計算したthread_id:")
        print(df[['_row_position', 'calculated_thread_id']].head(10))
        
        # Thread 1398101付近を探す
        print("\n4. Thread 1398101付近の確認:")
        df_full = pd.read_parquet(parquet_file)
        df_full['calculated_thread_id'] = ((df_full['_row_position'] - HEADER_SIZE) // THREAD_STRIDE).astype(int)
        
        # Thread 1398100-1398102の範囲
        thread_range = df_full[(df_full['calculated_thread_id'] >= 1398100) & 
                               (df_full['calculated_thread_id'] <= 1398102)]
        
        if len(thread_range) > 0:
            print(f"Thread 1398100-1398102の行数: {len(thread_range)}")
            print(thread_range[['c_custkey', '_row_position', 'calculated_thread_id']].head(10))
        else:
            print("Thread 1398100-1398102の範囲にデータが見つかりません")
        
        # Thread 3404030-3404034の範囲
        print("\n5. Thread 3404032付近の確認:")
        thread_range2 = df_full[(df_full['calculated_thread_id'] >= 3404030) & 
                                (df_full['calculated_thread_id'] <= 3404034)]
        
        if len(thread_range2) > 0:
            print(f"Thread 3404030-3404034の行数: {len(thread_range2)}")
            print(thread_range2[['c_custkey', '_row_position', 'calculated_thread_id']].head(10))
        else:
            print("Thread 3404030-3404034の範囲にデータが見つかりません")
        
        # スレッド分布の統計
        print("\n6. スレッド分布の統計:")
        thread_stats = df_full['calculated_thread_id'].describe()
        print(thread_stats)
        
        # ギャップを検出
        print("\n7. 大きなスレッドギャップの検出:")
        threads = sorted(df_full['calculated_thread_id'].unique())
        gaps = []
        for i in range(1, len(threads)):
            gap = threads[i] - threads[i-1]
            if gap > 1000:  # 1000スレッド以上のギャップ
                gaps.append((threads[i-1], threads[i], gap))
        
        if gaps:
            print("大きなギャップ（>1000スレッド）:")
            for prev, next, gap in gaps[:5]:  # 最初の5個
                print(f"  Thread {prev} → {next}: ギャップ {gap:,}スレッド")
                print(f"    アドレス: 0x{(HEADER_SIZE + prev * THREAD_STRIDE):08X} → 0x{(HEADER_SIZE + next * THREAD_STRIDE):08X}")
        else:
            print("大きなギャップは見つかりませんでした")

if __name__ == "__main__":
    check_thread_ids()