#!/usr/bin/env python3
"""
欠落問題の詳細分析
"""

# 基本情報
total_pg_rows = 12_030_000
total_parquet_rows = 10_374_649
missing_rows = total_pg_rows - total_parquet_rows
missing_percent = (missing_rows / total_pg_rows) * 100

print("=== 欠落問題の分析 ===")
print(f"PostgreSQL行数: {total_pg_rows:,}")
print(f"Parquet行数: {total_parquet_rows:,}")
print(f"欠落行数: {missing_rows:,} ({missing_percent:.1f}%)")

# 64MBアライメントの影響を計算
MB_64 = 64 * 1024 * 1024
row_size = 150  # 約150バイト/行

# 8並列の場合の分析
print("\n=== 8並列での64MBアライメントの影響 ===")

# 各ワーカーの処理量
total_data_size = total_pg_rows * row_size
data_per_worker = total_data_size / 8

print(f"総データサイズ: {total_data_size / (1024**3):.2f} GB")
print(f"ワーカーあたり: {data_per_worker / (1024**3):.2f} GB")

# 各ワーカーが必要な64MBブロック数
blocks_per_worker = int(data_per_worker / MB_64) + 1
print(f"ワーカーあたり64MBブロック数: {blocks_per_worker}")

# 最悪ケースの無駄
max_waste_per_block = MB_64 - 1  # 最大63.99MB
max_total_waste = 8 * blocks_per_worker * (MB_64 * 0.2)  # 平均20%の無駄と仮定
max_waste_rows = max_total_waste / row_size

print(f"\n最大無駄容量: {max_total_waste / (1024**3):.2f} GB")
print(f"最大無駄行数: {max_waste_rows:,.0f} 行")
print(f"実際の欠落: {missing_rows:,} 行")

# 結論
print("\n=== 結論 ===")
print("現在の64MBアライメント実装では、各64MBブロックの末尾に")
print("未使用領域が発生し、大量の行が欠落しています。")
print("\n解決策：")
print("1. GPU側でオーバーラップ読み取り（推奨）")
print("2. 最後の不完全な行を次のブロックに含める")
print("3. メタデータにactual_sizeを正しく記録してGPU側で利用")