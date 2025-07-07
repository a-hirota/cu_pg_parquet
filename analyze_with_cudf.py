#!/usr/bin/env python3
"""
cuDFを使用した高速欠落行分析
"""

import cudf
import pandas as pd
import numpy as np
import psycopg2
import glob
import os

# ParquetファイルをcuDFで読み込み
print("=== cuDFでParquetファイルを読み込み ===")
parquet_files = sorted(glob.glob("output/*.parquet"))
print(f"Parquetファイル数: {len(parquet_files)}")

# 全ファイルをcuDFで読み込み
all_dfs = []
for i, file in enumerate(parquet_files):
    print(f"[{i+1}/{len(parquet_files)}] {file}を読み込み中...", end="", flush=True)
    gdf = cudf.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
    all_dfs.append(gdf)
    print(f" {len(gdf):,}行")

# 全データを結合
print("\n全データを結合中...")
combined_gdf = cudf.concat(all_dfs, ignore_index=True)
print(f"総行数: {len(combined_gdf):,}")

# 基本統計
print("\n=== 基本統計 ===")
print(f"lo_orderkey範囲: {combined_gdf['lo_orderkey'].min()} - {combined_gdf['lo_orderkey'].max()}")
print(f"lo_linenumber範囲: {combined_gdf['lo_linenumber'].min()} - {combined_gdf['lo_linenumber'].max()}")

# ユニークなキーの数
print("\nユニークキーを計算中...")
# cuDFでユニークなペアを作成
combined_gdf['key_pair'] = combined_gdf['lo_orderkey'].astype(str) + '_' + combined_gdf['lo_linenumber'].astype(str)
unique_count = combined_gdf['key_pair'].nunique()
print(f"ユニークキー数: {unique_count:,}")

# 重複チェック
duplicate_count = len(combined_gdf) - unique_count
if duplicate_count > 0:
    print(f"⚠️  重複キー数: {duplicate_count:,}")
    
    # 重複キーのサンプルを表示
    duplicates = combined_gdf[combined_gdf.duplicated(subset=['lo_orderkey', 'lo_linenumber'], keep=False)]
    if len(duplicates) > 0:
        print("\n重複キーのサンプル（最初の10個）:")
        print(duplicates.head(10)[['lo_orderkey', 'lo_linenumber']].to_pandas())

# PostgreSQLとの比較
print("\n=== PostgreSQLとの比較 ===")
print("PostgreSQLの総行数: 246,012,324")
print(f"Parquetの総行数: {len(combined_gdf):,}")
print(f"カバー率: {len(combined_gdf) / 246012324 * 100:.1f}%")
print(f"欠落行数（推定）: {246012324 - len(combined_gdf):,}")

# チャンクごとの統計
print("\n=== チャンクごとの統計 ===")
for i, (file, gdf) in enumerate(zip(parquet_files, all_dfs)):
    chunk_name = os.path.basename(file).replace('.parquet', '')
    print(f"{chunk_name}: {len(gdf):,}行 (期待値の{len(gdf)/15286790*100:.1f}%)")

# メモリクリーンアップ
del combined_gdf
del all_dfs