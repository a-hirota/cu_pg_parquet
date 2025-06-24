import numpy as np
import cupy as cp
from numba import cuda

# テスト用フィールドデータ（実際のPostgreSQLバイナリフォーマットをシミュレート）
# 実際の問題は、field_offsetsとfield_lengthsの計算にあるかもしれない

print("PostgreSQL Binary Parser分析")
print("=" * 50)

# 仮想的なPostgreSQLバイナリデータ行（3列）
# 各行: [列数(2バイト)][フィールド長1(4バイト)][データ1][フィールド長2(4バイト)][データ2][フィールド長3(4バイト)][データ3]

# 行0: 3列, INT(4B)=100, BIGINT(8B)=200, VARCHAR="hello"(5B)
row0 = bytearray()
row0.extend((0, 3))  # 3列
row0.extend((0, 0, 0, 4))  # INT: 4バイト
row0.extend((0, 0, 0, 100))  # 値: 100
row0.extend((0, 0, 0, 8))  # BIGINT: 8バイト
row0.extend((0, 0, 0, 0, 0, 0, 0, 200))  # 値: 200
row0.extend((0, 0, 0, 5))  # VARCHAR: 5バイト
row0.extend(b"hello")  # 値: "hello"

# 行1: 3列, INT(4B)=101, BIGINT(8B)=201, VARCHAR="test"(4B)
row1 = bytearray()
row1.extend((0, 3))  # 3列
row1.extend((0, 0, 0, 4))  # INT: 4バイト
row1.extend((0, 0, 0, 101))  # 値: 101
row1.extend((0, 0, 0, 8))  # BIGINT: 8バイト
row1.extend((0, 0, 0, 0, 0, 0, 0, 201))  # 値: 201
row1.extend((0, 0, 0, 4))  # VARCHAR: 4バイト
row1.extend(b"test")  # 値: "test"

# 実際のfield_offsets計算をシミュレート
print("\n手動計算によるフィールドオフセット:")
print("Row 0:")
print(f"  列0 (INT): offset={2+4}, length=4")
print(f"  列1 (BIGINT): offset={2+4+4+4}, length=8")
print(f"  列2 (VARCHAR): offset={2+4+4+4+8+4}, length=5")

print("\nRow 1:")
print(f"  列0 (INT): offset={2+4}, length=4")
print(f"  列1 (BIGINT): offset={2+4+4+4}, length=8") 
print(f"  列2 (VARCHAR): offset={2+4+4+4+8+4}, length=4")

# 実際のparse_binary_chunk_gpu_ultra_fast_v2の動作を確認
def validate_and_extract_fields_simple(raw_data, row_start, expected_cols):
    """簡略化された行検証とフィールド抽出"""
    if row_start + 2 > len(raw_data):
        return False, -1, None, None
    
    # フィールド数確認
    num_fields = (raw_data[row_start] << 8) | raw_data[row_start + 1]
    if num_fields != expected_cols:
        return False, -2, None, None
    
    pos = row_start + 2
    field_offsets = []
    field_lengths = []
    
    # 全フィールドを検証+抽出
    for field_idx in range(num_fields):
        if pos + 4 > len(raw_data):
            return False, -3, None, None
        
        # フィールド長読み取り（ビッグエンディアン）
        flen = (
            (raw_data[pos] << 24) | (raw_data[pos+1] << 16) |
            (raw_data[pos+2] << 8) | raw_data[pos+3]
        )
        
        if flen == 0xFFFFFFFF:  # NULL
            field_offsets.append(0)
            field_lengths.append(-1)
            pos += 4
        else:
            if flen < 0 or flen > 1000000:
                return False, -4, None, None
            
            if pos + 4 + flen > len(raw_data):
                return False, -6, None, None
            
            # フィールド情報記録
            field_offsets.append(pos + 4)  # データ開始位置
            field_lengths.append(flen)      # データ長
            pos += 4 + flen
    
    return True, pos, field_offsets, field_lengths

# テスト実行
print("\n実際のパース結果:")
test_data = row0 + row1
test_np = np.frombuffer(test_data, dtype=np.uint8)

# Row 0のパース
success0, end0, offsets0, lengths0 = validate_and_extract_fields_simple(test_np, 0, 3)
if success0:
    print(f"Row 0: 成功")
    for i, (off, len_) in enumerate(zip(offsets0, lengths0)):
        print(f"  列{i}: offset={off}, length={len_}")
        if i == 2:  # 文字列列
            string_data = test_np[off:off+len_]
            print(f"    データ: '{bytes(string_data).decode()}'")

# Row 1のパース
success1, end1, offsets1, lengths1 = validate_and_extract_fields_simple(test_np, len(row0), 3)
if success1:
    print(f"\nRow 1: 成功")
    for i, (off, len_) in enumerate(zip(offsets1, lengths1)):
        print(f"  列{i}: offset={off}, length={len_}")
        if i == 2:  # 文字列列
            string_data = test_np[off:off+len_]
            print(f"    データ: '{bytes(string_data).decode()}'")

# 奇数行の問題を分析
print("\n奇数行分析:")
print(f"Row 1の文字列フィールド:")
print(f"  絶対オフセット: {offsets1[2]}")
print(f"  長さ: {lengths1[2]}")
print(f"  Row 0からの相対位置: {offsets1[2] - len(row0)}")

# メインプログラムでの問題を再現
print("\n\nメインプログラムでの処理シミュレーション:")
print("field_offsets_devとfield_lengths_devは既に正しい値を持っているはず")
print("問題は create_string_buffers での処理にある可能性")

# 実際の列インデックス検索をシミュレート
columns = ["id", "value", "name"]  # 仮の列名
var_columns = [{"name": "name", "index": 2}]  # 文字列列は列2

print("\n列インデックス検索:")
for var_col in var_columns:
    actual_col_idx = None
    for i, c in enumerate(columns):
        if c == var_col["name"]:
            actual_col_idx = i
            break
    print(f"文字列列 '{var_col['name']}' の実際のインデックス: {actual_col_idx}")

# 問題の可能性
print("\n\n考えられる問題:")
print("1. field_offsets_dev と field_lengths_dev の値は正しい")
print("2. extract_lengths_coalesced カーネルでの列インデックスアクセスは正しい")
print("3. copy_string_data_direct_optimized での field_offsets[row, col_idx] アクセスも正しい")
print("\n問題は、並列処理時のメモリアクセスパターンまたは")
print("グリッド/ブロック計算に起因する可能性が高い")