#!/usr/bin/env python3
"""
Parquetファイルの素早いチェック
"""

import pandas as pd
from pathlib import Path

output_dir = Path("/home/ubuntu/gpupgparser/output") 
parquet_files = sorted(output_dir.glob("chunk_*_queue.parquet"))

print(f"Parquetファイル数: {len(parquet_files)}")

# 最初と最後のファイルをチェック
for idx in [0, -1]:
    file = parquet_files[idx]
    print(f"\n{file.name}:")
    df = pd.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
    
    # 異常値チェック
    large_keys = df[df['lo_orderkey'] > 1e12]
    print(f"  行数: {len(df):,}")
    print(f"  lo_orderkey最大値: {df['lo_orderkey'].max():,}")
    
    if len(large_keys) > 0:
        print(f"  ⚠️ 異常に大きいorderkey: {len(large_keys)}行")
        print("  最初の5つ:")
        for i, row in large_keys.head().iterrows():
            print(f"    lo_orderkey={row['lo_orderkey']:,}, lo_linenumber={row['lo_linenumber']}")

# 総行数の簡易計算
print(f"\n簡易集計:")
print(f"最初のファイル行数: {len(pd.read_parquet(parquet_files[0], columns=['lo_orderkey']))}")
print(f"ファイル数: {len(parquet_files)}")
print(f"推定総行数: 約{len(pd.read_parquet(parquet_files[0], columns=['lo_orderkey'])) * len(parquet_files):,}")