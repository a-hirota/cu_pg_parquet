#!/usr/bin/env python3
"""
bpchar修正のみの効果を計算
"""

print("=== bpchar修正のみの効果計算 ===\n")

# customerテーブルの構造
fields = [
    ("field_count", 2, "固定"),
    ("c_custkey", 4 + 16, "DECIMAL128"),
    ("c_name", 4 + 20, "varchar(25) - 現状維持"),
    ("c_address", 4 + 20, "varchar(25) - 現状維持"),
    ("c_city", 4 + 10, "bpchar(10)"),
    ("c_nation", 4 + 15, "bpchar(15)"),
    ("c_region", 4 + 12, "bpchar(12)"),
    ("c_phone", 4 + 15, "bpchar(15)"),
    ("c_mktsegment", 4 + 10, "bpchar(10)")
]

print("現在の推定:")
current_total = 0
for name, current_size, desc in fields:
    if "bpchar" in desc:
        # 現在は全て20バイトで推定
        actual_current = 4 + 20
        print(f"  {name}: {actual_current}バイト (固定20バイトで推定)")
        current_total += actual_current
    else:
        print(f"  {name}: {current_size}バイト ({desc})")
        current_total += current_size

current_aligned = ((current_total + 31) // 32) * 32
print(f"\n現在の合計: {current_total}バイト → 32バイト整列: {current_aligned}バイト")

print("\n\nbpchar修正後の推定:")
fixed_total = 0
for name, size, desc in fields:
    print(f"  {name}: {size}バイト ({desc})")
    fixed_total += size

fixed_aligned = ((fixed_total + 31) // 32) * 32
print(f"\n修正後の合計: {fixed_total}バイト → 32バイト整列: {fixed_aligned}バイト")

# 効果計算
data_size = 848_104_872
actual_rows = 6_015_118

print(f"\n\n=== 推定への影響 ===")
print(f"データサイズ: {data_size:,}バイト")
print(f"\n現在:")
current_est_rows = data_size // current_aligned
print(f"  推定行サイズ: {current_aligned}バイト")
print(f"  推定行数: {current_est_rows:,}")
print(f"  バッファ(1.05倍): {int(current_est_rows * 1.05):,}")
print(f"  実際の行数: {actual_rows:,}")
print(f"  不足: {actual_rows - int(current_est_rows * 1.05):,}行")

print(f"\nbpchar修正後:")
fixed_est_rows = data_size // fixed_aligned
print(f"  推定行サイズ: {fixed_aligned}バイト")
print(f"  推定行数: {fixed_est_rows:,}")
print(f"  バッファ(1.05倍): {int(fixed_est_rows * 1.05):,}")
print(f"  実際の行数: {actual_rows:,}")
if int(fixed_est_rows * 1.05) >= actual_rows:
    print(f"  ✅ 十分なバッファ！")
else:
    shortage = actual_rows - int(fixed_est_rows * 1.05)
    print(f"  ❌ まだ不足: {shortage:,}行")
    required_factor = actual_rows / fixed_est_rows
    print(f"  必要な安全係数: {required_factor:.2f}")

print(f"\n改善率:")
print(f"  推定精度向上: {(fixed_est_rows - current_est_rows) / current_est_rows * 100:.1f}%")
print(f"  実際の行サイズ(141バイト)との誤差:")
print(f"    現在: {(current_aligned - 141) / 141 * 100:+.1f}%")
print(f"    修正後: {(fixed_aligned - 141) / 141 * 100:+.1f}%")