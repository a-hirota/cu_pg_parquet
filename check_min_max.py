#!/usr/bin/env python3
import cudf
import sys
from glob import glob

# 全ファイルの最小・最大値を収集
files = sorted(glob("output/lineorder_chunk_*_queue.parquet"))
print(f"合計 {len(files)} ファイルを分析中...\n")

min_vals = []
max_vals = []

for i, file in enumerate(files):
    try:
        df = cudf.read_parquet(file, columns=['lo_orderkey'])
        file_min = int(df['lo_orderkey'].min())
        file_max = int(df['lo_orderkey'].max())
        min_vals.append(file_min)
        max_vals.append(file_max)
        print(f"File {i:2d}: {file}")
        print(f"  最小値: {file_min:,}")
        print(f"  最大値: {file_max:,}")
        print()
    except Exception as e:
        print(f"Error reading {file}: {e}")

# 全体の最小・最大を計算
overall_min = min(min_vals)
overall_max = max(max_vals)

print("\n【全体の最小・最大値】")
print(f"  全ファイルの最小値: {overall_min:,}")
print(f"  全ファイルの最大値: {overall_max:,}")
print(f"  範囲: {overall_min:,} - {overall_max:,}")
print(f"  理論上の総数: {overall_max - overall_min + 1:,}")