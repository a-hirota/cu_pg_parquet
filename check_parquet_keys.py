#!/usr/bin/env python3
"""
Parquetファイルのキー範囲を確認
"""

import pandas as pd
import glob

# Parquetファイルを確認
parquet_files = sorted(glob.glob("output/*.parquet"))
print(f"Parquetファイル数: {len(parquet_files)}")

total_rows = 0
all_orderkeys = []

for i, file in enumerate(parquet_files[:3]):  # 最初の3ファイルのみ
    print(f"\n=== {file} ===")
    
    # lo_orderkeyとlo_linenumberを読み込み
    df = pd.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
    
    # Decimal型の変換
    if df['lo_orderkey'].dtype == 'object':
        # Decimal型を整数に変換
        df['lo_orderkey'] = pd.to_numeric(df['lo_orderkey'].astype(str).str.replace('.0', '', regex=False))
    
    rows = len(df)
    total_rows += rows
    
    # 基本統計
    print(f"行数: {rows:,}")
    print(f"OrderKey範囲: {df['lo_orderkey'].min()} - {df['lo_orderkey'].max()}")
    print(f"LineNumber範囲: {df['lo_linenumber'].min()} - {df['lo_linenumber'].max()}")
    
    # 最初と最後の5行を表示
    print("\n最初の5行:")
    print(df.head()[['lo_orderkey', 'lo_linenumber']])
    print("\n最後の5行:")
    print(df.tail()[['lo_orderkey', 'lo_linenumber']])
    
    # OrderKeyのリストを保存
    all_orderkeys.extend(df['lo_orderkey'].unique()[:1000])  # 最初の1000個

print(f"\n総行数（最初の3ファイル）: {total_rows:,}")

# 重複チェック
unique_orderkeys = len(set(all_orderkeys))
print(f"\nユニークなOrderKey数: {unique_orderkeys}")
print(f"重複数: {len(all_orderkeys) - unique_orderkeys}")