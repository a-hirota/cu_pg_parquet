#!/usr/bin/env python3
"""
問題のある位置の実際のバイト値を詳細表示
"""

import struct
from pathlib import Path

def show_bytes_detail(data, pos, length=100, label=""):
    """指定位置からのバイト値を詳細表示"""
    print(f"\n{label}")
    print(f"位置: 0x{pos:08X} ({pos:,})")
    print(f"{'='*80}")
    
    # 16バイトずつ表示
    for i in range(0, length, 16):
        if pos + i >= len(data):
            break
            
        # 16進表示
        hex_part = ""
        ascii_part = ""
        
        for j in range(16):
            if pos + i + j < len(data):
                byte = data[pos + i + j]
                hex_part += f"{byte:02X} "
                # ASCII表示
                if 32 <= byte <= 126:
                    ascii_part += chr(byte)
                else:
                    ascii_part += "."
            else:
                hex_part += "   "
                ascii_part += " "
                
        print(f"0x{pos + i:08X}: {hex_part} |{ascii_part}|")
    
    print()

def main():
    # バイナリファイルを読み込み
    bin_file = Path("test_data_large.bin") if Path("test_data_large.bin").exists() else Path("test_data.bin")
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    
    print("問題のある位置の実際のバイト値\n")
    
    # ケース1の詳細
    print("="*80)
    print("ケース1（11バイトずれ）")
    print("="*80)
    
    thread_start_1 = 66002323  # 0x03EF1D93
    recorded_pos_1 = 0x03EF1D9E
    
    # スレッド開始位置
    show_bytes_detail(binary_data, thread_start_1, 48, 
                     f"スレッド開始位置")
    
    # 記録された位置
    show_bytes_detail(binary_data, recorded_pos_1, 48,
                     f"記録された位置（+11バイト）")
    
    # 値の解釈
    print("値の解釈:")
    print(f"  スレッド開始 (0x{thread_start_1:08X}):")
    for i in range(0, 16, 2):
        if thread_start_1 + i + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[thread_start_1 + i:thread_start_1 + i + 2])[0]
            print(f"    +{i:2d}: 0x{val:04X} ({val:5d}) {'← 00 08なら行ヘッダ候補' if val == 8 else ''}")
    
    print(f"\n  記録位置 (0x{recorded_pos_1:08X}):")
    val = struct.unpack('>H', binary_data[recorded_pos_1:recorded_pos_1 + 2])[0]
    print(f"    値: 0x{val:04X} ({val}) ← これが行ヘッダとして記録された")
    
    # ケース2の詳細
    print("\n" + "="*80)
    print("ケース2（15バイトずれ）")
    print("="*80)
    
    thread_start_2 = 200575315  # 0x0BF48953
    recorded_pos_2 = 0x0BF48962
    
    # スレッド開始位置
    show_bytes_detail(binary_data, thread_start_2, 48,
                     f"スレッド開始位置")
    
    # 記録された位置
    show_bytes_detail(binary_data, recorded_pos_2, 48,
                     f"記録された位置（+15バイト）")
    
    # 値の解釈
    print("値の解釈:")
    print(f"  スレッド開始 (0x{thread_start_2:08X}):")
    for i in range(0, 16, 2):
        if thread_start_2 + i + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[thread_start_2 + i:thread_start_2 + i + 2])[0]
            print(f"    +{i:2d}: 0x{val:04X} ({val:5d}) {'← 00 08発見！' if val == 8 else ''}")
    
    print(f"\n  記録位置 (0x{recorded_pos_2:08X}):")
    val = struct.unpack('>H', binary_data[recorded_pos_2:recorded_pos_2 + 2])[0]
    print(f"    値: 0x{val:04X} ({val}) ← これが行ヘッダとして記録された")
    
    # 実際の行ヘッダを探す
    print("\n" + "="*80)
    print("実際の行ヘッダ位置の推定")
    print("="*80)
    
    for case_num, (start, recorded) in enumerate([(thread_start_1, recorded_pos_1), 
                                                   (thread_start_2, recorded_pos_2)], 1):
        print(f"\nケース{case_num}:")
        # 前後200バイトで00 08を探す
        found = []
        for i in range(max(0, recorded - 200), min(len(binary_data) - 2, recorded + 200)):
            if binary_data[i] == 0x00 and binary_data[i+1] == 0x08:
                # 次の4バイトが妥当なフィールド長か確認
                if i + 6 <= len(binary_data):
                    field_len = struct.unpack('>i', binary_data[i+2:i+6])[0]
                    if field_len == 8:  # int64の場合
                        distance = i - recorded
                        found.append((i, distance, field_len))
        
        if found:
            print(f"  00 08パターン（最初のフィールドが8バイト）:")
            for pos, dist, flen in found[:3]:
                direction = "前" if dist < 0 else "後"
                print(f"    0x{pos:08X} (記録位置の{abs(dist)}バイト{direction})")

if __name__ == "__main__":
    main()