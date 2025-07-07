#!/usr/bin/env python3
"""
修正された推定計算
"""

# customerテーブルの構造
print("=== 修正された推定計算 ===\n")

# 現在の推定
current_estimation = {
    'field_count': 2,  # 2バイト
    'c_custkey': 4 + 16,  # 4(長さ) + 16(decimal)
    'c_name': 4 + 20,     # 4(長さ) + 20(varchar推定)
    'c_address': 4 + 20,  # 4(長さ) + 20(varchar推定)
    'c_city': 4 + 10,     # 4(長さ) + 10(bpchar)
    'c_nation': 4 + 15,   # 4(長さ) + 15(bpchar)
    'c_region': 4 + 12,   # 4(長さ) + 12(bpchar)
    'c_phone': 4 + 15,    # 4(長さ) + 15(bpchar)
    'c_mktsegment': 4 + 10,  # 4(長さ) + 10(bpchar)
}

current_total = sum(current_estimation.values())
current_aligned = ((current_total + 31) // 32) * 32
print(f"現在の推定:")
print(f"  生サイズ: {current_total}バイト")
print(f"  32バイト整列後: {current_aligned}バイト")

# 修正された推定（実データに基づく）
corrected_estimation = {
    'field_count': 2,  # 2バイト
    'c_custkey': 4 + 16,  # 4(長さ) + 16(decimal)
    'c_name': 4 + 18,     # 4(長さ) + 18(実際の平均)
    'c_address': 4 + 15,  # 4(長さ) + 15(実際の平均)
    'c_city': 4 + 10,     # 4(長さ) + 10(bpchar)
    'c_nation': 4 + 15,   # 4(長さ) + 15(bpchar)
    'c_region': 4 + 12,   # 4(長さ) + 12(bpchar)
    'c_phone': 4 + 15,    # 4(長さ) + 15(bpchar)
    'c_mktsegment': 4 + 10,  # 4(長さ) + 10(bpchar)
}

corrected_total = sum(corrected_estimation.values())
corrected_aligned = ((corrected_total + 31) // 32) * 32
print(f"\n修正された推定:")
print(f"  生サイズ: {corrected_total}バイト")
print(f"  32バイト整列後: {corrected_aligned}バイト")

# 実際の値との比較
actual_row_size = 141.0  # 実測値
print(f"\n実際の行サイズ: {actual_row_size}バイト")
print(f"現在の推定誤差: {(current_aligned - actual_row_size) / actual_row_size * 100:+.1f}%")
print(f"修正後の推定誤差: {(corrected_aligned - actual_row_size) / actual_row_size * 100:+.1f}%")

# 推定行数への影響
data_size = 848_104_872
print(f"\nデータサイズ: {data_size:,}バイト")
print(f"現在の推定行数: {data_size // current_aligned:,}")
print(f"修正後の推定行数: {data_size // corrected_aligned:,}")
print(f"実際の行数: 6,015,118")

# バッファサイズ計算
print(f"\nバッファサイズ（1.05倍）:")
print(f"  現在: {int(data_size // current_aligned * 1.05):,}行")
print(f"  修正後: {int(data_size // corrected_aligned * 1.05):,}行")
print(f"  必要: 6,015,118行")

print(f"\nバッファサイズ（1.5倍）:")
print(f"  現在: {int(data_size // current_aligned * 1.5):,}行")
print(f"  修正後: {int(data_size // corrected_aligned * 1.5):,}行")