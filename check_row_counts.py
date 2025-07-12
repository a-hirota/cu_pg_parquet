#!/usr/bin/env python3
import cudf
from glob import glob

# 全ファイルの行数をカウント
files = sorted(glob("output/lineorder_chunk_*_queue.parquet"))
print(f"合計 {len(files)} ファイルの行数を確認中...\n")

total_rows = 0
file_rows = []

for i, file in enumerate(files):
    try:
        df = cudf.read_parquet(file)
        rows = len(df)
        total_rows += rows
        file_rows.append((file, rows))
        print(f"File {i:2d}: {file}")
        print(f"  行数: {rows:,}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"\n【合計】")
print(f"  全ファイルの合計行数: {total_rows:,}")
print(f"\n行数でソート（降順）:")
for file, rows in sorted(file_rows, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {file}: {rows:,}")