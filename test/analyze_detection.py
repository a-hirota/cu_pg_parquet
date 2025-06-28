#!/usr/bin/env python3
"""
Grid境界スレッドの検出位置を詳細分析
"""

# Thread ID: 70402のケースを分析
row_position = 508865678
field_0_offset = 508865684

print("=== Thread ID 70402の分析 ===")
print(f"Row Position: {row_position}")
print(f"Row Position パリティ: {'偶数' if row_position % 2 == 0 else '奇数'}")

# バイナリデータから行ヘッダの位置を推定
# Field 0のオフセットから逆算
# Field 0 offset = Row Position + 2 (行ヘッダ) + 4 (フィールド長)
expected_field_0_offset = row_position + 2 + 4
print(f"\n期待されるField 0 offset: {expected_field_0_offset}")
print(f"実際のField 0 offset: {field_0_offset}")
print(f"差分: {field_0_offset - expected_field_0_offset}")

# バイナリデータの解析
binary_data = [
    0x39, 0x37, 0x30, 0x32, 0x31, 0x30, 0x00, 0x00, 0x00, 0x0a, 
    0x46, 0x4f, 0x42, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 
    0x00, 0x11, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x02, 0x00, 0x01, 
    0x00, 0x00, 0x00, 0x00, 0x1a, 0x9b, 0x0b, 0x32, 0x00, 0x00
]

print("\nバイナリデータ分析:")
# Row Positionから20バイト前 = 508865658
sample_start = row_position - 20

for i, byte in enumerate(binary_data):
    pos = sample_start + i
    if pos == row_position:
        print(f"位置 {pos}: 0x{byte:02x} <- Row Position ここ！")
    elif pos == row_position + 1:
        print(f"位置 {pos}: 0x{byte:02x} <- 行ヘッダ2バイト目")
    else:
        print(f"位置 {pos}: 0x{byte:02x}")

# 0x00 0x11 = 17 (フィールド数)を探す
print("\n行ヘッダ (0x00 0x11 = 17列) の検索:")
for i in range(len(binary_data) - 1):
    if binary_data[i] == 0x00 and binary_data[i+1] == 0x11:
        actual_pos = sample_start + i
        print(f"発見！位置 {actual_pos} ({'偶数' if actual_pos % 2 == 0 else '奇数'})")
        print(f"Row Positionとの差: {actual_pos - row_position}")

print("\n結論:")
print("Row Positionは行ヘッダの開始位置を正確に指しているか？")