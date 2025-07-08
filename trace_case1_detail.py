#!/usr/bin/env python3
"""
ケース1の11バイトずれを詳細追跡
"""

import struct
from pathlib import Path

def trace_simd_search(data, start_pos, target_pos):
    """SIMD検索を詳細にトレース"""
    pos = start_pos
    iteration = 0
    
    print(f"開始位置: 0x{start_pos:08X}")
    print(f"目標位置: 0x{target_pos:08X}")
    print(f"差: {target_pos - start_pos} バイト\n")
    
    while pos <= target_pos + 20 and iteration < 10:
        print(f"反復 {iteration}: pos = 0x{pos:08X}")
        
        # 16バイト検索をシミュレート
        found_at = -1
        for i in range(16):
            if pos + i + 2 > len(data):
                continue
                
            val = struct.unpack('>H', data[pos + i:pos + i + 2])[0]
            if i < 8 or val == 8:  # 最初の8個は詳細表示
                print(f"  +{i:2d} (0x{pos + i:08X}): {data[pos + i]:02X} {data[pos + i + 1]:02X} = {val}", end="")
                if pos + i == target_pos:
                    print(" <-- 記録された位置", end="")
                if val == 8:
                    print(" ★00 08発見★", end="")
                    if found_at == -1:
                        found_at = pos + i
                print()
        
        if found_at >= 0:
            print(f"  → 00 08を0x{found_at:08X}で発見")
            # ここで検証が行われる
            break
        else:
            print(f"  → 見つからず、pos += 15")
            next_pos = pos + 15
            
            # 次の位置が目標を超える場合
            if pos < target_pos and next_pos > target_pos:
                print(f"  ⚠️ 次の位置0x{next_pos:08X}は目標を超える")
                print(f"  ⚠️ 何か別のロジックが働いている可能性")
            
            pos = next_pos
        
        iteration += 1
        print()

def main():
    # バイナリファイルを読み込み
    bin_file = Path("test_data_large.bin") if Path("test_data_large.bin").exists() else Path("test_data.bin")
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    
    print("=== ケース1の詳細トレース ===\n")
    
    # 実際のスレッド境界より少し前から始めてみる
    actual_thread_start = 0x03EF1D93
    recorded_pos = 0x03EF1D9E
    
    # スレッド開始の少し前を確認
    print("スレッド開始前の状況:")
    for offset in [-20, -15, -10, -5, 0]:
        pos = actual_thread_start + offset
        if pos >= 0 and pos + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[pos:pos+2])[0]
            print(f"  {offset:+3d} (0x{pos:08X}): {binary_data[pos]:02X} {binary_data[pos+1]:02X} = {val}")
    
    print("\n" + "="*60 + "\n")
    
    # 別の可能性：スレッドが実際にはもう少し前から始まっている？
    possible_starts = [
        actual_thread_start - 4,  # 11 + 4 = 15
        actual_thread_start,
        actual_thread_start + 1,
    ]
    
    for start in possible_starts:
        print(f"\n仮説: スレッドが0x{start:08X}から開始した場合")
        trace_simd_search(binary_data, start, recorded_pos)
        print("\n" + "-"*60)

if __name__ == "__main__":
    main()