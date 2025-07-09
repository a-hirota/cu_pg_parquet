#!/usr/bin/env python3
"""シンプルに欠落を確認"""

import pandas as pd

# 両チャンクを読み込み
chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
chunk1 = pd.read_parquet("output/customer_chunk_1_queue.parquet")

# Decimal型をintに変換
chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
chunk1['c_custkey'] = chunk1['c_custkey'].astype('int64')

# 全キーを取得
all_keys = set(chunk0['c_custkey']) | set(chunk1['c_custkey'])

print(f"総ユニークキー数: {len(all_keys):,}")
print(f"期待値: 12,030,000")
print(f"差: {12030000 - len(all_keys)}")

# 本当に欠落しているキーを確認
missing = []
for i in range(1, 12030001):
    if i not in all_keys:
        missing.append(i)

print(f"\n欠落キー数: {len(missing)}")
print("\n欠落キー:")
for key in missing:
    print(f"  {key}")