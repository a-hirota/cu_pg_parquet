#!/usr/bin/env python3
"""
欠落行数を確認するスクリプト
"""
import cudf
import subprocess

# PostgreSQLから総行数を取得
result = subprocess.run(
    ["psql", "-t", "-c", "SELECT COUNT(*) FROM customer"],
    capture_output=True,
    text=True
)
pg_count = int(result.stdout.strip())
print(f"PostgreSQL総行数: {pg_count:,}")

# Parquetファイルの行数を確認
df0 = cudf.read_parquet('output/customer_chunk_0_queue.parquet')
df1 = cudf.read_parquet('output/customer_chunk_1_queue.parquet')

print(f"\nParquetファイル:")
print(f"  chunk 0: {len(df0):,} 行")
print(f"  chunk 1: {len(df1):,} 行")
print(f"  合計: {len(df0) + len(df1):,} 行")

# 欠落行数
missing = pg_count - (len(df0) + len(df1))
print(f"\n欠落行数: {missing:,} 行")

# 境界情報を表示
print("\n境界位置分析:")
print(f"  チャンク0最大: {df0['c_custkey'].max()}")
print(f"  チャンク1最小: {df1['c_custkey'].min()}")
print(f"  ギャップ: {df1['c_custkey'].min() - df0['c_custkey'].max() - 1}")

# 1GB = 1024MB = 1024*1024*1024 bytes
gb_boundaries = []
for gb in range(1, 10):  # 1GB～9GBの境界をチェック
    boundary = gb * 1024 * 1024 * 1024
    gb_boundaries.append((gb, boundary, hex(boundary)))

print("\nGB境界位置:")
for gb, bytes_val, hex_val in gb_boundaries:
    print(f"  {gb}GB: {bytes_val:,} bytes ({hex_val})")

# 実際のギャップ位置と最も近い境界を特定
gap_start = df0['c_custkey'].max()
gap_end = df1['c_custkey'].min()
print(f"\n実際のギャップ:")
print(f"  開始: c_custkey={gap_start}")
print(f"  終了: c_custkey={gap_end}")
print(f"  欠落数: {gap_end - gap_start - 1:,}")