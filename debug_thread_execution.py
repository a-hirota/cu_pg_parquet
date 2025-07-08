#!/usr/bin/env python3
"""
テストモード実行後のParquetファイルからスレッド実行情報を分析
"""

import cudf
from pathlib import Path
import numpy as np

def analyze_thread_execution():
    """スレッド実行パターンを分析"""
    
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    print("スレッド実行分析:")
    print("="*80)
    
    for pf in parquet_files:
        print(f"\n{pf.name}:")
        df = cudf.read_parquet(pf)
        
        # 基本統計
        total_rows = len(df)
        unique_threads = df['_thread_id'].nunique()
        
        print(f"  総行数: {total_rows:,}")
        print(f"  ユニークスレッド数: {unique_threads:,}")
        
        # スレッドごとの行数を集計
        thread_counts = df.groupby('_thread_id').size().to_pandas()
        
        # 行数の分布
        print(f"\n  スレッドあたりの行数分布:")
        print(f"    最小: {thread_counts.min()}")
        print(f"    最大: {thread_counts.max()}")
        print(f"    平均: {thread_counts.mean():.2f}")
        print(f"    中央値: {thread_counts.median():.1f}")
        
        # 1行しか処理していないスレッドを特定
        single_row_threads = thread_counts[thread_counts == 1].index.tolist()
        print(f"\n  1行のみ処理したスレッド: {len(single_row_threads)}個")
        
        if len(single_row_threads) > 0:
            # 最初の10個を表示
            print(f"    例: {single_row_threads[:10]}")
            
            # これらのスレッドのパターンを分析
            analyze_single_row_threads(df, single_row_threads[:20])

def analyze_single_row_threads(df, thread_ids):
    """1行しか処理していないスレッドを詳しく分析"""
    
    print("\n  1行スレッドの詳細分析:")
    
    for tid in thread_ids[:5]:  # 最初の5個を詳しく分析
        thread_data = df[df['_thread_id'] == tid]
        
        if len(thread_data) > 0:
            row = thread_data.iloc[0]
            
            # 各フィールドの値を取得
            thread_start = int(row['_thread_start_pos'].iloc[0] if hasattr(row['_thread_start_pos'], 'iloc') else row['_thread_start_pos'])
            thread_end = int(row['_thread_end_pos'].iloc[0] if hasattr(row['_thread_end_pos'], 'iloc') else row['_thread_end_pos'])
            row_pos = int(row['_row_position'].iloc[0] if hasattr(row['_row_position'], 'iloc') else row['_row_position'])
            
            print(f"\n    Thread {tid}:")
            print(f"      スレッド範囲: 0x{thread_start:08X} - 0x{thread_end:08X} ({thread_end - thread_start} bytes)")
            print(f"      行位置: 0x{row_pos:08X}")
            print(f"      行位置からスレッド終了までの距離: {thread_end - row_pos} bytes")
            
            # c_custkeyの値
            custkey_val = row['c_custkey'].iloc[0] if hasattr(row['c_custkey'], 'iloc') else row['c_custkey']
            if hasattr(custkey_val, 'dtype') and hasattr(custkey_val.dtype, 'precision'):
                custkey = int(custkey_val)
            else:
                custkey = int(custkey_val)
            print(f"      c_custkey: {custkey}")
            
            # このスレッドの前後のスレッドを確認
            all_threads = df['_thread_id'].unique().to_pandas()
            all_threads = np.sort(all_threads)
            tid_idx = np.where(all_threads == tid)[0][0]
            
            if tid_idx > 0:
                prev_tid = all_threads[tid_idx - 1]
                prev_data = df[df['_thread_id'] == prev_tid]
                if len(prev_data) > 0:
                    print(f"      前のスレッド {prev_tid}: {len(prev_data)}行")
            
            if tid_idx < len(all_threads) - 1:
                next_tid = all_threads[tid_idx + 1]
                next_data = df[df['_thread_id'] == next_tid]
                if len(next_data) > 0:
                    print(f"      次のスレッド {next_tid}: {len(next_data)}行")

def check_thread_boundaries():
    """スレッド境界での問題を確認"""
    
    print("\n\nスレッド境界の分析:")
    print("="*80)
    
    # 欠落している2つのキーの周辺を確認
    missing_keys = [3483509, 6491094]
    
    for missing_key in missing_keys:
        print(f"\n欠落キー {missing_key} の周辺:")
        
        for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
            df = cudf.read_parquet(pf)
            
            # c_custkeyを整数に変換
            if hasattr(df['c_custkey'].dtype, 'precision'):
                df['c_custkey_int'] = df['c_custkey'].astype('int64')
            else:
                df['c_custkey_int'] = df['c_custkey']
            
            # 欠落キーの前後を取得
            mask = ((df['c_custkey_int'] >= missing_key - 5) & 
                   (df['c_custkey_int'] <= missing_key + 5))
            
            if mask.sum() > 0:
                print(f"\n  {pf.name}:")
                near_df = df[mask].sort_values('c_custkey_int')
                cols = ['c_custkey_int', '_thread_id', '_row_position', '_thread_end_pos']
                near_pd = near_df[cols].to_pandas()
                
                for _, row in near_pd.iterrows():
                    ckey = int(row['c_custkey_int'])
                    tid = int(row['_thread_id'])
                    pos = int(row['_row_position'])
                    end_pos = int(row['_thread_end_pos'])
                    
                    marker = " ← 欠落" if ckey == missing_key else ""
                    remaining = end_pos - pos
                    print(f"    c_custkey={ckey}, thread={tid}, pos=0x{pos:08X}, 残り={remaining}bytes{marker}")

if __name__ == "__main__":
    analyze_thread_execution()
    check_thread_boundaries()