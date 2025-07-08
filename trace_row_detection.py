#!/usr/bin/env python3
"""
行検出プロセスをトレース
"""

import struct
from pathlib import Path

def simulate_read_uint16_simd16_lite(data, pos, end_pos, ncols):
    """read_uint16_simd16_liteの動作をシミュレート"""
    results = []
    
    # 16バイトずつチェック
    for i in range(16):
        test_pos = pos + i
        if test_pos + 2 > len(data) or test_pos >= end_pos:
            continue
        
        # ビッグエンディアンで2バイト読み取り
        val = struct.unpack('>H', data[test_pos:test_pos+2])[0]
        results.append((test_pos, val, val == ncols))
    
    return results

def main():
    # バイナリファイルを読み込み
    bin_file = Path("test_data_large.bin") if Path("test_data_large.bin").exists() else Path("test_data.bin")
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    
    # 問題のあるスレッド境界をシミュレート
    test_cases = [
        {
            'name': '問題行1',
            'thread_start': 66002323,
            'actual_row_pos': 0x3EF1D9E,  # 記録されたrow_position
            'distance': 11
        },
        {
            'name': '問題行2', 
            'thread_start': 200575315,
            'actual_row_pos': 0xBF48962,  # 記録されたrow_position
            'distance': 15
        }
    ]
    
    for case in test_cases:
        print(f"\n=== {case['name']} ===")
        print(f"スレッド開始: 0x{case['thread_start']:X}")
        print(f"記録されたrow_position: 0x{case['actual_row_pos']:X}")
        print(f"距離: {case['distance']} bytes")
        
        # スレッド開始位置から検索をシミュレート
        pos = case['thread_start']
        print(f"\nスレッド開始位置からの検索:")
        
        # 最初の16バイトをチェック
        results = simulate_read_uint16_simd16_lite(binary_data, pos, pos + 200, 8)
        
        for test_pos, val, is_match in results:
            offset = test_pos - case['thread_start']
            marker = " <-- 記録された位置" if test_pos == case['actual_row_pos'] else ""
            match_str = " ★行ヘッダ候補★" if is_match else ""
            print(f"  +{offset:2d}: 0x{test_pos:08X} = {val:5d} (0x{val:04X}){match_str}{marker}")
        
        # 実際の00 08を探す
        print(f"\n実際の00 08パターン（前後100バイト）:")
        for i in range(max(0, case['actual_row_pos'] - 100), min(len(binary_data) - 2, case['actual_row_pos'] + 100)):
            if binary_data[i] == 0x00 and binary_data[i+1] == 0x08:
                distance = i - case['actual_row_pos']
                direction = "前" if distance < 0 else "後"
                print(f"  0x{i:08X} ({direction}{abs(distance)}バイト)")
                
                # その位置から行構造を簡単に確認
                if i + 10 < len(binary_data):
                    field1_len = struct.unpack('>i', binary_data[i+2:i+6])[0]
                    if field1_len == 8:
                        print(f"    → 最初のフィールドが8バイト（int64）なので正しい行ヘッダの可能性大")

if __name__ == "__main__":
    main()