#!/usr/bin/env python3
"""
重複レコードの詳細チェック
"""

import cudf
from pathlib import Path
import numpy as np

def check_duplicates():
    """重複レコードを詳細にチェック"""
    
    print("重複レコードの詳細チェック:")
    print("="*80)
    
    # 各ファイルごとにチェック
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        print(f"\n{pf.name}:")
        df = cudf.read_parquet(pf)
        
        # c_custkeyの重複をチェック
        if hasattr(df['c_custkey'].dtype, 'precision'):
            custkeys = df['c_custkey'].astype('int64')
        else:
            custkeys = df['c_custkey']
        
        # 重複をカウント
        value_counts = custkeys.value_counts()
        duplicates = value_counts[value_counts > 1]
        
        print(f"  総行数: {len(df):,}")
        print(f"  ユニークc_custkey数: {custkeys.nunique():,}")
        print(f"  重複があるc_custkey数: {len(duplicates):,}")
        
        if len(duplicates) > 0:
            # 重複の詳細を表示
            dup_pd = duplicates.to_pandas()
            print(f"\n  重複の詳細（最初の10個）:")
            for key, count in list(dup_pd.items())[:10]:
                print(f"    c_custkey={key}: {count}回出現")
                
                # このキーのすべての行を表示
                mask = custkeys == key
                dup_rows = df[mask][['c_custkey', '_thread_id', '_row_position']]
                dup_rows_pd = dup_rows.to_pandas()
                
                for idx, row in dup_rows_pd.iterrows():
                    thread_id = int(row['_thread_id'])
                    row_pos = int(row['_row_position'])
                    print(f"      Thread {thread_id}, Position 0x{row_pos:08X}")
            
            # 重複数の分布
            print(f"\n  重複数の分布:")
            for i in range(2, min(6, int(dup_pd.max()) + 1)):
                count = (dup_pd == i).sum()
                if count > 0:
                    print(f"    {i}回重複: {count}個のキー")

def analyze_64mb_boundary():
    """64MB境界の詳細分析"""
    
    print("\n\n64MB境界の詳細分析:")
    print("="*80)
    
    # 0x04000000 (64MB) から 0x08000000 (128MB) の範囲のデータを確認
    boundary_64mb = 0x04000000
    boundary_128mb = 0x08000000
    
    print(f"分析範囲: 0x{boundary_64mb:08X} - 0x{boundary_128mb:08X}")
    
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        print(f"\n{pf.name}:")
        df = cudf.read_parquet(pf)
        
        # この範囲のデータを抽出
        mask = ((df['_row_position'] >= boundary_64mb) & 
                (df['_row_position'] < boundary_128mb))
        
        boundary_data = df[mask]
        
        if len(boundary_data) > 0:
            print(f"  この範囲の行数: {len(boundary_data):,}")
            
            # c_custkeyの範囲
            if hasattr(boundary_data['c_custkey'].dtype, 'precision'):
                min_key = int(boundary_data['c_custkey'].min())
                max_key = int(boundary_data['c_custkey'].max())
            else:
                min_key = int(boundary_data['c_custkey'].min())
                max_key = int(boundary_data['c_custkey'].max())
            
            print(f"  c_custkey範囲: {min_key} - {max_key}")
            
            # スレッドIDの分布
            unique_threads = boundary_data['_thread_id'].nunique()
            print(f"  ユニークスレッド数: {unique_threads}")
            
            # 最初と最後の行
            first_row = boundary_data.iloc[0]
            last_row = boundary_data.iloc[-1]
            
            # pandasに変換
            first_pd = first_row.to_pandas()
            last_pd = last_row.to_pandas()
            
            print(f"\n  最初の行:")
            print(f"    c_custkey: {int(first_pd['c_custkey'].iloc[0])}")
            print(f"    Thread: {int(first_pd['_thread_id'].iloc[0])}")
            print(f"    Position: 0x{int(first_pd['_row_position'].iloc[0]):08X}")
            
            print(f"\n  最後の行:")
            print(f"    c_custkey: {int(last_pd['c_custkey'].iloc[0])}")
            print(f"    Thread: {int(last_pd['_thread_id'].iloc[0])}")
            print(f"    Position: 0x{int(last_pd['_row_position'].iloc[0]):08X}")
        else:
            print(f"  この範囲にデータなし！")

def check_thread_coverage():
    """スレッドカバレッジをチェック"""
    
    print("\n\nスレッドカバレッジの分析:")
    print("="*80)
    
    # Thread 349525 から 699050 までのスレッドをチェック
    start_thread = 349525
    end_thread = 699050
    
    print(f"チェック範囲: Thread {start_thread} - {end_thread}")
    print(f"総スレッド数: {end_thread - start_thread + 1:,}")
    
    # 実際に存在するスレッドを確認
    existing_threads = set()
    
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        df = cudf.read_parquet(pf, columns=['_thread_id'])
        threads = df['_thread_id'].unique().to_pandas()
        
        # 範囲内のスレッドを抽出
        range_threads = threads[(threads >= start_thread) & (threads <= end_thread)]
        existing_threads.update(range_threads)
    
    print(f"\n実際に存在するスレッド: {len(existing_threads):,}")
    
    # 欠落しているスレッドを特定
    all_threads = set(range(start_thread, end_thread + 1))
    missing_threads = all_threads - existing_threads
    
    if len(missing_threads) > 0:
        print(f"欠落スレッド: {len(missing_threads):,}")
        print(f"欠落スレッドの例: {sorted(missing_threads)[:10]}")
    else:
        print("すべてのスレッドが存在します")
    
    # これらのスレッドが処理したデータ量を確認
    print("\n\nスレッド範囲のデータ処理量:")
    
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        df = cudf.read_parquet(pf)
        
        # 範囲内のスレッドのデータ
        mask = ((df['_thread_id'] >= start_thread) & (df['_thread_id'] <= end_thread))
        range_data = df[mask]
        
        if len(range_data) > 0:
            print(f"\n{pf.name}:")
            print(f"  この範囲のスレッドが処理した行数: {len(range_data):,}")
            
            # 位置の範囲
            min_pos = int(range_data['_row_position'].min())
            max_pos = int(range_data['_row_position'].max())
            print(f"  位置範囲: 0x{min_pos:08X} - 0x{max_pos:08X}")
            
            # 64MB境界との関係
            if min_pos < 0x04000000 and max_pos > 0x08000000:
                print(f"  → 64MB-128MB境界をまたいでいる")
            elif max_pos < 0x04000000:
                print(f"  → 64MB境界より前")
            elif min_pos > 0x08000000:
                print(f"  → 128MB境界より後")

if __name__ == "__main__":
    check_duplicates()
    analyze_64mb_boundary()
    check_thread_coverage()