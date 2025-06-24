"""累積和オーバーフローテスト

大規模データで累積和（cumsum）がint32の範囲を超える可能性を調査
"""

import numpy as np
import cupy as cp

print("=== 累積和オーバーフローテスト ===")

# lineorderテーブルの文字列フィールドサイズ
string_fields = {
    'lo_orderpriority': 15,
    'lo_shippriority': 1,  
    'lo_commit_date': 8,
    'lo_shipmode': 10
}

# 各チャンクの行数（約1500万行）
rows_per_chunk = 15_000_000

for field_name, field_size in string_fields.items():
    # 累積和の最大値を計算
    total_size = rows_per_chunk * field_size
    
    # int32の最大値
    int32_max = 2**31 - 1
    
    print(f"\n{field_name}:")
    print(f"  フィールドサイズ: {field_size} bytes")
    print(f"  行数: {rows_per_chunk:,}")
    print(f"  総サイズ: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")
    print(f"  int32最大値: {int32_max:,}")
    
    if total_size > int32_max:
        print(f"  ⚠️ オーバーフロー！ 比率: {total_size / int32_max:.2f}x")
    else:
        print(f"  ✅ 安全（使用率: {total_size / int32_max * 100:.1f}%）")

# 実際のcumsum計算をシミュレート
print("\n=== CuPy cumsum シミュレーション ===")

# lo_orderpriority（15バイト）で1500万行
lengths = cp.full(rows_per_chunk, 15, dtype=cp.int32)

# cumsum計算
try:
    offsets = cp.cumsum(lengths, dtype=cp.int32)
    print(f"cumsum成功: 最終値 = {int(offsets[-1]):,}")
except Exception as e:
    print(f"cumsumエラー: {e}")
    
# int64で再計算
offsets_64 = cp.cumsum(lengths, dtype=cp.int64)
final_value = int(offsets_64[-1])
print(f"int64での最終値: {final_value:,}")

if final_value > int32_max:
    print(f"\n⚠️ int32オーバーフロー検出！")
    print(f"オーバーフロー位置: 行 {int(int32_max / 15):,}")
    
    # オーバーフロー時の動作を確認
    lengths_host = lengths.get()
    offsets_32_host = np.cumsum(lengths_host, dtype=np.int32)
    
    # オーバーフロー付近の値を確認
    overflow_idx = int(int32_max / 15)
    print(f"\nオーバーフロー付近の値:")
    for i in range(overflow_idx - 5, overflow_idx + 5):
        if i < len(offsets_32_host):
            print(f"  行{i}: {offsets_32_host[i]:,}")
            if offsets_32_host[i] < 0:
                print(f"    ⚠️ 負の値！")

print("\n=== 結論 ===")
print("大規模データ（1500万行）でlo_orderpriority列の累積オフセットが")
print("int32の範囲を超える可能性があります。これが奇数行破損の原因かもしれません。")