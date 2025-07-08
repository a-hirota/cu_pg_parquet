#!/usr/bin/env python3
"""
read_uint16_simd16_liteのロジックをデバッグ
"""

import struct
from pathlib import Path

def debug_simd_logic(data, start_pos, end_pos, expected_distance):
    """SIMD検索ロジックの詳細デバッグ"""
    print(f"\n=== SIMD検索ロジックデバッグ ===")
    print(f"開始位置: 0x{start_pos:08X}")
    print(f"終了位置: 0x{end_pos:08X}")
    print(f"期待される距離: {expected_distance}")
    
    pos = start_pos
    iteration = 0
    
    while pos < end_pos and iteration < 5:  # 最初の5回のみ
        print(f"\n反復 {iteration}:")
        print(f"  現在位置: 0x{pos:08X}")
        
        # read_uint16_simd16_liteをシミュレート
        found = False
        for i in range(16):
            test_pos = pos + i
            if test_pos + 2 > len(data) or test_pos >= end_pos:
                continue
                
            val = struct.unpack('>H', data[test_pos:test_pos+2])[0]
            if val == 8:
                print(f"  → 位置 +{i} (0x{test_pos:08X}) で00 08を発見！")
                print(f"  → candidate_pos = 0x{test_pos:08X}")
                found = True
                
                # これが記録された位置と一致するか？
                if test_pos + expected_distance == start_pos + expected_distance:
                    print(f"  ⚠️ しかし、記録された位置は 0x{start_pos + expected_distance:08X} (+{expected_distance})")
                
                break
        
        if not found:
            print(f"  → 行ヘッダ未発見、pos += 15")
            pos += 15
        else:
            # validate_and_extract_fields_liteの結果をシミュレート
            # ここで何か問題があるかもしれない
            break
            
        iteration += 1

def main():
    # バイナリファイルを読み込み
    bin_file = Path("test_data_large.bin") if Path("test_data_large.bin").exists() else Path("test_data.bin")
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    
    # 問題行2のケースを詳しく調査
    thread_start = 200575315  # 0xBF48953
    recorded_pos = 0xBF48962
    distance = 15
    
    print(f"問題行2の詳細分析:")
    print(f"スレッド開始: 0x{thread_start:08X}")
    print(f"記録された位置: 0x{recorded_pos:08X}")
    print(f"距離: {distance}")
    
    # スレッド開始位置のデータを表示
    print(f"\nスレッド開始位置のデータ:")
    for i in range(20):
        if thread_start + i + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[thread_start + i:thread_start + i + 2])[0]
            print(f"  +{i:2d}: {binary_data[thread_start + i]:02X} {binary_data[thread_start + i + 1]:02X} = {val}")
    
    # SIMDロジックをデバッグ
    debug_simd_logic(binary_data, thread_start, thread_start + 200, distance)
    
    # もう一つの可能性：前の行の検証が失敗した場合
    print(f"\n=== 別の可能性: 前の反復の影響 ===")
    
    # スレッド開始前の位置を探る
    prev_start = thread_start - 20
    print(f"前の検索位置からシミュレート (0x{prev_start:08X}):")
    
    # 前の位置から順に進む
    pos = prev_start
    while pos < recorded_pos:
        # read_uint16_simd16_liteの結果
        found_at = -1
        for i in range(16):
            test_pos = pos + i
            if test_pos + 2 > len(binary_data):
                continue
            val = struct.unpack('>H', binary_data[test_pos:test_pos+2])[0]
            if val == 8:
                found_at = test_pos
                break
        
        if found_at >= 0:
            print(f"  pos=0x{pos:08X}: 00 08を0x{found_at:08X}で発見")
            if found_at == thread_start:
                print(f"    → これはスレッド開始位置！")
                # 検証が失敗した場合、pos = candidate_pos + 1
                print(f"    → もし検証失敗なら、次は 0x{found_at + 1:08X}")
                pos = found_at + 1
            else:
                break
        else:
            print(f"  pos=0x{pos:08X}: 見つからず、+15")
            pos += 15
            
        if pos == recorded_pos:
            print(f"  → 記録された位置 0x{recorded_pos:08X} に到達！")
            break

if __name__ == "__main__":
    main()