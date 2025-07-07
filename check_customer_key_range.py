#!/usr/bin/env python3
"""
customerテーブルのキー範囲を確認
"""

import pandas as pd
import psycopg2

# Parquetファイルのキー範囲
print("=== Parquetファイルのキー範囲 ===")
df = pd.read_parquet("/home/ubuntu/gpupgparser/output/chunk_0_queue.parquet", columns=['c_custkey'])
print(f"行数: {len(df):,}")
print(f"c_custkey最小値: {df['c_custkey'].min()}")
print(f"c_custkey最大値: {df['c_custkey'].max()}")

# キーの分布をチェック
key_ranges = []
for i in range(0, 12, 1):  # 0M, 1M, 2M, ... 11M
    count = len(df[(df['c_custkey'] >= i*1000000) & (df['c_custkey'] < (i+1)*1000000)])
    if count > 0:
        key_ranges.append(f"{i}M-{i+1}M: {count:,}行")

print("\nキー範囲ごとの行数:")
for r in key_ranges:
    print(f"  {r}")

# PostgreSQLのキー範囲
print("\n=== PostgreSQLのキー範囲 ===")
conn = psycopg2.connect(dbname="postgres", user="postgres", password="postgres")
with conn.cursor() as cur:
    cur.execute("SELECT MIN(c_custkey), MAX(c_custkey), COUNT(*) FROM customer")
    min_key, max_key, total = cur.fetchone()
    print(f"c_custkey最小値: {min_key}")
    print(f"c_custkey最大値: {max_key}")
    print(f"総行数: {total:,}")
    
    # カバー範囲を確認
    cur.execute(f"""
        SELECT COUNT(*) 
        FROM customer 
        WHERE c_custkey >= {df['c_custkey'].min()} 
        AND c_custkey <= {df['c_custkey'].max()}
    """)
    covered = cur.fetchone()[0]
    print(f"\nParquetキー範囲内の行数: {covered:,}")
    print(f"カバー率: {len(df) / covered * 100:.1f}%")

conn.close()