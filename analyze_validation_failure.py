#!/usr/bin/env python3
"""
validate_and_extract_fields_liteの失敗原因を分析
"""

import struct
from pathlib import Path

def validate_fields(data, row_start, expected_cols=8):
    """validate_and_extract_fields_liteをシミュレート"""
    print(f"\n=== 行検証シミュレーション (位置: 0x{row_start:08X}) ===")
    
    if row_start + 2 > len(data):
        print("エラー: 行開始位置がデータ範囲外")
        return False, "範囲外"
    
    # フィールド数確認
    num_fields = struct.unpack('>H', data[row_start:row_start+2])[0]
    print(f"フィールド数: {num_fields} (期待値: {expected_cols})")
    
    if num_fields != expected_cols:
        return False, "フィールド数不一致"
    
    pos = row_start + 2
    
    # customerテーブルの固定長
    # c_custkey: -1 (可変長だが通常8バイト)
    # c_name: -1 (可変長)
    # c_address: -1 (可変長)
    # c_city: 10
    # c_nation: 15
    # c_region: 12
    # c_phone: 15
    # c_mktsegment: 10
    fixed_lengths = [-1, -1, -1, 10, 15, 12, 15, 10]
    
    # 各フィールドを検証
    for field_idx in range(num_fields):
        if pos + 4 > len(data):
            print(f"エラー: フィールド{field_idx}の長さ読み取り位置がデータ範囲外")
            return False, f"フィールド{field_idx}範囲外"
        
        # フィールド長
        field_len = struct.unpack('>i', data[pos:pos+4])[0]
        print(f"  フィールド{field_idx}: 長さ={field_len}", end="")
        
        if field_len == -1:  # NULL
            print(" (NULL)")
            pos += 4
        else:
            # 異常値チェック
            if field_len < 0 or field_len > 1000000:
                print(f" → 異常な長さ")
                return False, f"フィールド{field_idx}異常長"
            
            # 固定長チェック
            if field_idx < len(fixed_lengths) and fixed_lengths[field_idx] > 0:
                if field_len != fixed_lengths[field_idx]:
                    print(f" → 固定長エラー (期待値: {fixed_lengths[field_idx]})")
                    return False, f"フィールド{field_idx}固定長エラー"
                else:
                    print(f" (固定長OK)")
            else:
                print(f" (可変長)")
            
            # 境界チェック
            if pos + 4 + field_len > len(data):
                print(f" → データ境界超過")
                return False, f"フィールド{field_idx}境界超過"
            
            pos += 4 + field_len
    
    # 次の行ヘッダ検証
    if pos + 2 <= len(data):
        next_header = struct.unpack('>H', data[pos:pos+2])[0]
        print(f"\n次の行ヘッダ: 0x{next_header:04X} (位置: 0x{pos:08X})")
        if next_header != expected_cols and next_header != 0xFFFF:
            print(f"  → 次の行ヘッダが不正")
            return False, "次行ヘッダ不正"
    
    print(f"\n検証成功！行終端: 0x{pos:08X}")
    return True, "成功"

def main():
    # バイナリファイルを読み込み
    bin_file = Path("test_data_large.bin") if Path("test_data_large.bin").exists() else Path("test_data.bin")
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    
    # 問題のスレッド境界を分析
    test_positions = [
        {
            'name': '問題行2 - スレッド開始位置',
            'pos': 0x0BF48953,
            'desc': '00 08が存在するが検証失敗と推測'
        },
        {
            'name': '問題行2 - 記録された位置',
            'pos': 0x0BF48962,
            'desc': '実際に記録されたrow_position'
        },
        {
            'name': '問題行2 - 00 08の別候補',
            'pos': 0x0BF48967,
            'desc': '記録位置の5バイト後'
        }
    ]
    
    for test in test_positions:
        print(f"\n{'='*60}")
        print(f"{test['name']}")
        print(f"位置: 0x{test['pos']:08X}")
        print(f"説明: {test['desc']}")
        print(f"{'='*60}")
        
        # その位置のデータを表示
        if test['pos'] + 20 < len(binary_data):
            print(f"\nバイナリデータ:")
            for i in range(0, 20, 2):
                val = struct.unpack('>H', binary_data[test['pos']+i:test['pos']+i+2])[0]
                print(f"  +{i:2d}: {binary_data[test['pos']+i]:02X} {binary_data[test['pos']+i+1]:02X} = {val}")
        
        # 検証を実行
        success, reason = validate_fields(binary_data, test['pos'])
        print(f"\n検証結果: {'成功' if success else '失敗'} ({reason})")

if __name__ == "__main__":
    main()