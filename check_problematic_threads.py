#!/usr/bin/env python3
"""
前回の分析で見つかった問題のあるスレッドIDが実際に存在するか確認
"""

import cudf
from pathlib import Path

def check_specific_threads():
    """特定のスレッドIDが存在するか確認"""
    
    # 前回の分析で見つかったスレッドID
    problem_threads = [
        1048575, 3201426, 1747626, 3606663, 2097152, 3809255,
        699049, 2998833, 1398101, 3404032, 2446677, 4214529,
        349524, 2796205, 716296, 717145, 719349, 719434
    ]
    
    parquet_files = sorted(Path("output").glob("*chunk_*_queue.parquet"))
    
    print("問題のあるスレッドIDの存在確認:")
    print("="*80)
    
    found_threads = {}
    
    for pf in parquet_files:
        print(f"\n{pf.name}を確認中...")
        df = cudf.read_parquet(pf, columns=['_thread_id', 'c_custkey'])
        
        for thread_id in problem_threads:
            thread_data = df[df['_thread_id'] == thread_id]
            if len(thread_data) > 0:
                found_threads[thread_id] = {
                    'file': pf.name,
                    'count': len(thread_data),
                    'custkeys': thread_data['c_custkey'].to_pandas().tolist()[:5]  # 最初の5件
                }
                print(f"  Thread {thread_id}: {len(thread_data)} 行")
    
    print("\n\n見つかったスレッド:")
    print("="*80)
    for tid, info in sorted(found_threads.items()):
        print(f"Thread {tid}:")
        print(f"  ファイル: {info['file']}")
        print(f"  行数: {info['count']}")
        print(f"  c_custkey例: {info['custkeys']}")
    
    print("\n\n見つからなかったスレッド:")
    print("="*80)
    not_found = [t for t in problem_threads if t not in found_threads]
    print(f"合計 {len(not_found)} 個: {not_found}")

def analyze_missing_key_neighborhoods():
    """欠落キー周辺の実際のデータを確認"""
    
    missing_keys = [476029, 1227856, 1979731, 2731633, 3483451]
    
    print("\n\n欠落キー周辺の実際のデータ:")
    print("="*80)
    
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    for missing_key in missing_keys[:3]:  # 最初の3つを詳しく分析
        print(f"\n欠落キー {missing_key} の周辺:")
        
        found = False
        for pf in parquet_files:
            df = cudf.read_parquet(pf)
            
            # c_custkeyを整数に変換
            if hasattr(df['c_custkey'].dtype, 'precision'):
                df['c_custkey_int'] = df['c_custkey'].astype('int64')
            else:
                df['c_custkey_int'] = df['c_custkey']
            
            # 前後10件を取得
            mask = ((df['c_custkey_int'] >= missing_key - 5) & 
                   (df['c_custkey_int'] <= missing_key + 5))
            
            if mask.sum() > 0:
                found = True
                near_df = df[mask].sort_values('c_custkey_int')
                near_pd = near_df[['c_custkey_int', '_thread_id', '_row_position']].to_pandas()
                
                print(f"  {pf.name}:")
                for _, row in near_pd.iterrows():
                    marker = " ← 欠落" if row['c_custkey_int'] == missing_key else ""
                    print(f"    c_custkey={int(row['c_custkey_int']):8d}, thread={int(row['_thread_id']):6d}, pos=0x{int(row['_row_position']):08X}{marker}")
        
        if not found:
            print(f"  → このキー周辺のデータが見つかりません")

if __name__ == "__main__":
    check_specific_threads()
    analyze_missing_key_neighborhoods()