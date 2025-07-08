#!/usr/bin/env python3
"""
64MB境界での欠落を詳しく調査
"""
import cudf

# Parquetファイルを読み込み
df0 = cudf.read_parquet('output/customer_chunk_0_queue.parquet')
df1 = cudf.read_parquet('output/customer_chunk_1_queue.parquet')

# 全データを結合してソート
df = cudf.concat([df0, df1])
df = df.sort_values('c_custkey')

# キーの範囲を確認
print(f"全体の行数: {len(df):,}")
print(f"最小キー: {df['c_custkey'].min()}")
print(f"最大キー: {df['c_custkey'].max()}")

# 重複チェック
duplicates = len(df) - df['c_custkey'].nunique()
print(f"\n重複行数: {duplicates}")

# 連続性チェック（大きなギャップを探す）
custkeys = df['c_custkey'].to_pandas().values  # numpy配列に変換
gaps = []

for i in range(1, len(custkeys)):
    expected = int(custkeys[i-1]) + 1
    actual = int(custkeys[i])
    gap_size = actual - expected
    
    if gap_size > 1:
        gaps.append({
            'before': int(custkeys[i-1]),
            'after': actual,
            'gap_size': gap_size,
            'position': i
        })

# 大きなギャップ上位10個を表示
print(f"\n見つかったギャップ数: {len(gaps)}")
print("\n大きなギャップ（上位10）:")
gaps_sorted = sorted(gaps, key=lambda x: x['gap_size'], reverse=True)[:10]

for gap in gaps_sorted:
    print(f"  {gap['before']} → {gap['after']} (ギャップ: {gap['gap_size']:,}行)")
    
# チャンク境界での欠落を確認
print(f"\nチャンク0最大: {df0['c_custkey'].max()}")
print(f"チャンク1最小: {df1['c_custkey'].min()}")
print(f"チャンク境界ギャップ: {df1['c_custkey'].min() - df0['c_custkey'].max() - 1}")