#!/usr/bin/env python3
"""
PostgreSQLとParquetファイルを比較して欠落している顧客を特定
"""

import cudf
import psycopg
import os
import numpy as np
from pathlib import Path

def get_postgres_custkeys():
    """PostgreSQLから全c_custkeyを取得"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    print("PostgreSQLから全c_custkeyを取得中...")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # c_custkeyを昇順で取得
            cur.execute("SELECT c_custkey FROM customer ORDER BY c_custkey")
            pg_custkeys = np.array([row[0] for row in cur.fetchall()], dtype=np.int64)
    
    print(f"PostgreSQL: {len(pg_custkeys):,} 件")
    return pg_custkeys

def get_parquet_custkeys():
    """Parquetファイルから全c_custkeyを取得"""
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    if not parquet_files:
        # 旧形式のファイル名も試す
        parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    print(f"\nParquetファイル: {len(parquet_files)} 個")
    
    all_custkeys = []
    for pf in parquet_files:
        print(f"  読み込み中: {pf.name}")
        df = cudf.read_parquet(pf)
        
        # c_custkeyを取得（Decimal型の場合は変換）
        if hasattr(df['c_custkey'].dtype, 'precision'):
            custkeys = df['c_custkey'].astype('int64').to_pandas().values
        else:
            custkeys = df['c_custkey'].to_pandas().values
        
        all_custkeys.append(custkeys)
        print(f"    → {len(custkeys):,} 件")
    
    # 結合してソート
    parquet_custkeys = np.concatenate(all_custkeys)
    parquet_custkeys = np.sort(parquet_custkeys)
    
    print(f"Parquet合計: {len(parquet_custkeys):,} 件")
    return parquet_custkeys

def find_missing_custkeys(pg_keys, parquet_keys):
    """欠落しているc_custkeyを特定"""
    # numpyのsetdiff1dを使用（pg_keysにあってparquet_keysにないもの）
    missing_keys = np.setdiff1d(pg_keys, parquet_keys)
    
    print(f"\n欠落: {len(missing_keys)} 件")
    
    if len(missing_keys) > 0:
        print("\n欠落しているc_custkey:")
        for i, key in enumerate(missing_keys):
            print(f"  {i+1:2d}. {key}")
            
            # 前後のキーも表示
            idx_in_pg = np.searchsorted(pg_keys, key)
            if idx_in_pg > 0:
                print(f"      前: {pg_keys[idx_in_pg-1]}")
            print(f"      → 欠落: {key}")
            if idx_in_pg < len(pg_keys) - 1:
                print(f"      次: {pg_keys[idx_in_pg+1]}")
            
            # Parquet側での位置
            insert_pos = np.searchsorted(parquet_keys, key)
            print(f"      Parquetでの挿入位置: {insert_pos:,}")
            if insert_pos > 0 and insert_pos < len(parquet_keys):
                print(f"      Parquet前後: {parquet_keys[insert_pos-1]} < [欠落] < {parquet_keys[insert_pos]}")
            print()
    
    return missing_keys

def analyze_missing_positions(missing_keys):
    """欠落キーがどのチャンクに属するべきか分析"""
    # 各チャンクの範囲を調べる
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    if not parquet_files:
        parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    print("\nチャンク別のc_custkey範囲:")
    chunk_ranges = []
    
    for pf in parquet_files:
        df = cudf.read_parquet(pf, columns=['c_custkey'])
        
        if hasattr(df['c_custkey'].dtype, 'precision'):
            min_key = int(df['c_custkey'].min())
            max_key = int(df['c_custkey'].max())
        else:
            min_key = int(df['c_custkey'].min())
            max_key = int(df['c_custkey'].max())
        
        chunk_ranges.append((pf.name, min_key, max_key))
        print(f"  {pf.name}: {min_key:,} - {max_key:,}")
    
    # 欠落キーがどのチャンクに属するべきか
    print("\n欠落キーの所属チャンク:")
    for key in missing_keys:
        for chunk_name, min_key, max_key in chunk_ranges:
            if min_key <= key <= max_key:
                print(f"  {key} → {chunk_name} (範囲内)")
                break
        else:
            # どのチャンクにも属さない場合
            print(f"  {key} → チャンク境界?")
            # より詳細な分析
            for i, (chunk_name, min_key, max_key) in enumerate(chunk_ranges):
                if key < min_key:
                    if i > 0:
                        prev_chunk = chunk_ranges[i-1]
                        print(f"    → {prev_chunk[0]}の最大値({prev_chunk[2]:,})と{chunk_name}の最小値({min_key:,})の間")
                    break

def check_thread_boundaries(missing_keys):
    """欠落キーがスレッド境界に関連するか確認"""
    # スレッド情報を含むParquetファイルを読み込み
    parquet_files = sorted(Path("output").glob("*chunk_*_queue.parquet"))
    
    print("\n欠落キー周辺のスレッド情報:")
    
    for pf in parquet_files:
        df = cudf.read_parquet(pf)
        
        # c_custkeyを整数に変換
        if hasattr(df['c_custkey'].dtype, 'precision'):
            df['c_custkey_int'] = df['c_custkey'].astype('int64')
        else:
            df['c_custkey_int'] = df['c_custkey']
        
        # 各欠落キーの前後を確認
        for key in missing_keys:
            # 欠落キーの前後の行を探す
            mask_before = df['c_custkey_int'] < key
            mask_after = df['c_custkey_int'] > key
            
            if mask_before.sum() > 0 and mask_after.sum() > 0:
                # 直前の行
                before_df = df[mask_before].sort_values('c_custkey_int', ascending=False)
                if len(before_df) > 0:
                    row_before = before_df.iloc[0]
                else:
                    continue
                
                # 直後の行
                after_df = df[mask_after].sort_values('c_custkey_int', ascending=True)
                if len(after_df) > 0:
                    row_after = after_df.iloc[0]
                else:
                    continue
                
                # 値を取得（cuDFのDataFrameからSeriesを経由）
                custkey_before = int(row_before['c_custkey_int'].to_pandas())
                thread_before = int(row_before['_thread_id'].to_pandas())
                pos_before = int(row_before['_row_position'].to_pandas())
                
                custkey_after = int(row_after['c_custkey_int'].to_pandas())
                thread_after = int(row_after['_thread_id'].to_pandas())
                pos_after = int(row_after['_row_position'].to_pandas())
                
                print(f"\n欠落キー {key}:")
                print(f"  直前の行:")
                print(f"    c_custkey: {custkey_before}")
                print(f"    thread_id: {thread_before}")
                print(f"    row_position: 0x{pos_before:08X}")
                
                print(f"  直後の行:")
                print(f"    c_custkey: {custkey_after}")
                print(f"    thread_id: {thread_after}")
                print(f"    row_position: 0x{pos_after:08X}")
                
                # スレッドが異なる場合は境界
                if thread_before != thread_after:
                    print(f"  → スレッド境界! (Thread {thread_before} → {thread_after})")

def main():
    # PostgreSQLとParquetのキーを取得
    pg_keys = get_postgres_custkeys()
    parquet_keys = get_parquet_custkeys()
    
    # 欠落キーを特定
    missing_keys = find_missing_custkeys(pg_keys, parquet_keys)
    
    if len(missing_keys) > 0:
        # 欠落位置を分析
        analyze_missing_positions(missing_keys)
        
        # スレッド境界を確認
        check_thread_boundaries(missing_keys)
    else:
        print("\n✅ 欠落はありません！")

if __name__ == "__main__":
    main()