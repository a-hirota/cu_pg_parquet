#!/usr/bin/env python3
"""
実際の行サイズを分析して推定式を改善
"""

import os

# 実際のデータ
tables_data = {
    'lineorder': {
        'actual_rows': 148_915_554,
        'data_size': 3_486_556_377,  # バイト
        'postgres_rows': 246_012_324,
        'pages': 13_268_551
    },
    'customer': {
        'actual_rows': 6_015_118,  # GPU検出
        'data_size': 848_104_872,  # チャンク0
        'postgres_rows': 6_015_125,  # PostgreSQL COUNT
        'pages': 104_186,  # チャンク0
        'estimated_row_size': 192,  # 現在の推定
        'estimated_rows': 4_417_212  # 現在の推定結果
    }
}

print("=== 実際の行サイズ分析 ===\n")

for table, data in tables_data.items():
    print(f"{table}テーブル:")
    
    if 'actual_rows' in data and 'data_size' in data:
        actual_bytes_per_row = data['data_size'] / data['actual_rows']
        print(f"  実際の平均行サイズ: {actual_bytes_per_row:.1f} バイト/行")
    
    if 'postgres_rows' in data and 'pages' in data:
        rows_per_page = data['postgres_rows'] / data['pages']
        print(f"  実際の行密度: {rows_per_page:.1f} 行/ページ")
    
    if 'estimated_row_size' in data:
        print(f"  現在の推定行サイズ: {data['estimated_row_size']} バイト/行")
        error_rate = (data['estimated_row_size'] - actual_bytes_per_row) / actual_bytes_per_row * 100
        print(f"  推定誤差: {error_rate:+.1f}%")
    
    if 'estimated_rows' in data and 'postgres_rows' in data:
        coverage = data['estimated_rows'] / data['postgres_rows'] * 100
        print(f"  推定カバー率: {coverage:.1f}%")
    
    print()

# 改善案
print("\n=== 推定式の改善案 ===")
print("\n1. テーブル固有の係数を使用:")
print("   - lineorder: 実際の行サイズ ≈ 23.4 バイト/行")
print("   - customer: 実際の行サイズ ≈ 141.0 バイト/行")
print("   → 現在の推定（192バイト）は大きすぎる")

print("\n2. 動的な推定方法:")
print("   a) 最初の数MBをサンプリングして実際の行サイズを計測")
print("   b) PostgreSQLのpg_statsから平均行長を取得")
print("   c) Rustと同様にSELECT COUNT(*)で実際の行数を取得")

print("\n3. 安全係数の調整:")
customer_safety_factor = 6_015_118 / 4_417_212
print(f"   - customerテーブルに必要な係数: {customer_safety_factor:.2f}")
print("   - 現在の1.05では不十分")
print("   - 推奨: 1.50（50%マージン）")

# サンプルベースの行サイズ推定コード
print("\n4. サンプルベースの推定実装例:")
print("""
def estimate_row_size_from_sample(raw_dev, header_size, sample_size_mb=1):
    \"\"\"最初の1MBをサンプリングして実際の行サイズを推定\"\"\"
    sample_size = min(sample_size_mb * 1024 * 1024, raw_dev.size - header_size)
    sample_data = raw_dev[header_size:header_size + sample_size]
    
    # 簡易的な行検出（フィールド数マーカーを探す）
    detected_rows = 0
    pos = 0
    while pos < sample_size - 2:
        field_count = struct.unpack('>h', sample_data[pos:pos+2])[0]
        if 0 < field_count < 100:  # 妥当なフィールド数
            detected_rows += 1
            pos += estimate_row_size_from_columns(columns)  # 次の候補位置へ
        else:
            pos += 1
    
    if detected_rows > 10:  # 十分なサンプル
        return sample_size / detected_rows
    else:
        return estimate_row_size_from_columns(columns)  # フォールバック
""")