#!/usr/bin/env python3
"""
両方のケースの詳細分析
"""

import struct
from pathlib import Path

def analyze_case(binary_data, thread_start, recorded_pos, case_name):
    """各ケースを詳細分析"""
    print(f"\n{'='*70}")
    print(f"{case_name}")
    print(f"{'='*70}")
    print(f"スレッド開始: 0x{thread_start:08X}")
    print(f"記録された位置: 0x{recorded_pos:08X}")
    print(f"差: {recorded_pos - thread_start} バイト")
    
    # スレッド開始位置から20バイトをチェック
    print(f"\nスレッド開始位置からの00 08検索:")
    found_positions = []
    
    for i in range(20):
        if thread_start + i + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[thread_start + i:thread_start + i + 2])[0]
            if val == 8:
                found_positions.append(i)
                print(f"  +{i:2d}: 00 08 発見！")
    
    if not found_positions:
        print("  00 08 が見つかりません")
        
        # 記録された位置の値を確認
        if recorded_pos + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[recorded_pos:recorded_pos+2])[0]
            print(f"\n記録された位置の値: 0x{val:04X} ({val})")
            
            # SIMDロジックをシミュレート
            print(f"\nSIMDロジックシミュレーション:")
            pos = thread_start
            iteration = 0
            while pos < recorded_pos + 10 and iteration < 10:
                # 16バイト検索
                found = False
                for j in range(16):
                    if pos + j + 2 <= len(binary_data):
                        test_val = struct.unpack('>H', binary_data[pos + j:pos + j + 2])[0]
                        if test_val == 8:
                            print(f"  反復{iteration}: pos=0x{pos:08X} → +{j}で00 08発見 (0x{pos + j:08X})")
                            found = True
                            break
                
                if not found:
                    print(f"  反復{iteration}: pos=0x{pos:08X} → 見つからず、+15")
                    pos += 15
                    if pos == recorded_pos:
                        print(f"    → 記録された位置に到達！")
                        break
                else:
                    break
                    
                iteration += 1

def main():
    # バイナリファイルを読み込み
    bin_file = Path("test_data_large.bin") if Path("test_data_large.bin").exists() else Path("test_data.bin")
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    
    # ケース1
    analyze_case(
        binary_data,
        thread_start=66002323,
        recorded_pos=0x3EF1D9E,
        case_name="ケース1（問題行1）"
    )
    
    # ケース2
    analyze_case(
        binary_data,
        thread_start=200575315,
        recorded_pos=0xBF48962,
        case_name="ケース2（問題行2）"
    )

if __name__ == "__main__":
    main()