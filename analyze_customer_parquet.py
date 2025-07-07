#!/usr/bin/env python3
"""
customerテーブルのParquetファイル分析
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def analyze_customer_parquet():
    """customerテーブルのParquetファイルを分析"""
    
    output_dir = Path("/home/ubuntu/gpupgparser/output")
    parquet_file = output_dir / "chunk_0_queue.parquet"
    
    if not parquet_file.exists():
        print(f"ファイルが見つかりません: {parquet_file}")
        return
    
    print(f"=== {parquet_file.name} の分析 ===\n")
    
    # メタデータ確認
    parquet_meta = pq.ParquetFile(parquet_file)
    print(f"スキーマ:")
    print(parquet_meta.schema)
    print(f"\n行数: {parquet_meta.metadata.num_rows:,}")
    
    # データ読み込み
    df = pd.read_parquet(parquet_file)
    
    print(f"\n列情報:")
    print(df.dtypes)
    
    print(f"\n基本統計:")
    print(f"  総行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    
    # c_custkeyの範囲を確認
    if 'c_custkey' in df.columns:
        print(f"\nc_custkeyの範囲:")
        print(f"  最小値: {df['c_custkey'].min()}")
        print(f"  最大値: {df['c_custkey'].max()}")
        print(f"  ユニーク数: {df['c_custkey'].nunique():,}")
        
        # 重複チェック
        duplicates = df['c_custkey'].duplicated().sum()
        print(f"  重複数: {duplicates}")
        
        # 異常値チェック
        abnormal = df[df['c_custkey'] > 10_000_000]
        if len(abnormal) > 0:
            print(f"\n⚠️ 異常に大きいc_custkey: {len(abnormal)}行")
            print(abnormal[['c_custkey']].head())
    
    # 最初の5行を表示
    print(f"\n最初の5行:")
    print(df.head())
    
    # メモリ使用量
    memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"\nメモリ使用量: {memory_usage:.2f} MB")
    
    # 期待値との比較
    expected_rows = 6_000_000
    coverage = len(df) / expected_rows * 100
    print(f"\n期待値との比較:")
    print(f"  期待行数: {expected_rows:,}")
    print(f"  実際の行数: {len(df):,}")
    print(f"  カバー率: {coverage:.2f}%")
    print(f"  不足行数: {expected_rows - len(df):,}")

def main():
    analyze_customer_parquet()

if __name__ == "__main__":
    main()