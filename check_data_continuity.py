#!/usr/bin/env python3
"""データの連続性を詳細確認"""

import pandas as pd
import numpy as np

def check_data_continuity():
    """チャンク0内のデータ連続性を確認"""
    print("=== チャンク0のデータ連続性確認 ===\n")
    
    df = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # _row_positionでソート
    df_sorted = df.sort_values('_row_position').reset_index(drop=True)
    
    # 位置の差分を計算
    positions = df_sorted['_row_position'].values
    diffs = np.diff(positions)
    
    # 通常の行サイズ（中央値）
    normal_size = np.median(diffs)
    print(f"通常の行サイズ（中央値）: {normal_size:.0f} bytes")
    
    # 大きなギャップ（通常の100倍以上）を検出
    large_gaps = np.where(diffs > normal_size * 100)[0]
    
    print(f"\n大きなギャップ（>100倍）の数: {len(large_gaps)}")
    
    if len(large_gaps) > 0:
        # 最大のギャップを詳細分析
        max_gap_idx = large_gaps[np.argmax(diffs[large_gaps])]
        max_gap_size = diffs[max_gap_idx]
        
        print(f"\n最大のギャップ:")
        print(f"  サイズ: {max_gap_size:,} bytes ({max_gap_size/(1024*1024):.1f}MB)")
        
        # ギャップ前後の行
        row_before = df_sorted.iloc[max_gap_idx]
        row_after = df_sorted.iloc[max_gap_idx + 1]
        
        print(f"\n  ギャップ前の行:")
        print(f"    インデックス: {max_gap_idx}")
        print(f"    c_custkey: {row_before['c_custkey']}")
        print(f"    位置: 0x{row_before['_row_position']:08X} ({row_before['_row_position']/(1024*1024):.1f}MB)")
        
        print(f"\n  ギャップ後の行:")
        print(f"    インデックス: {max_gap_idx + 1}")
        print(f"    c_custkey: {row_after['c_custkey']}")
        print(f"    位置: 0x{row_after['_row_position']:08X} ({row_after['_row_position']/(1024*1024):.1f}MB)")
        
        # このギャップ内に欠落キーがあるか確認
        if row_before['c_custkey'] < 3483451 < row_after['c_custkey']:
            print(f"\n  ⚠️ 欠落キー3483451はこのギャップ内にあります！")
        
        # ギャップ前後の詳細データ
        print(f"\n  ギャップ前後の10行:")
        start_idx = max(0, max_gap_idx - 5)
        end_idx = min(len(df_sorted), max_gap_idx + 6)
        
        for i in range(start_idx, end_idx):
            row = df_sorted.iloc[i]
            marker = ""
            if i == max_gap_idx:
                marker = " ← ギャップ前"
            elif i == max_gap_idx + 1:
                marker = " ← ギャップ後"
            
            print(f"    [{i:6d}] key={row['c_custkey']:8d}, pos=0x{row['_row_position']:08X}{marker}")
    
    # c_custkeyの連続性も確認
    print("\n\n=== c_custkeyの連続性確認 ===")
    
    # c_custkeyでソート
    df_key_sorted = df.sort_values('c_custkey').reset_index(drop=True)
    keys = df_key_sorted['c_custkey'].values
    key_diffs = np.diff(keys)
    
    # キーのギャップ（差が2以上）
    key_gaps = np.where(key_diffs >= 2)[0]
    
    print(f"キーのギャップ（差≥2）の数: {len(key_gaps)}")
    
    if len(key_gaps) > 0:
        # 欠落キー3483451付近を確認
        for gap_idx in key_gaps[:10]:  # 最初の10個
            key_before = keys[gap_idx]
            key_after = keys[gap_idx + 1]
            gap_size = key_after - key_before
            
            if key_before <= 3483451 <= key_after:
                print(f"\n⚠️ 欠落キー3483451を含むギャップ:")
                print(f"  {key_before} → {key_after} (ギャップ: {gap_size})")
                
                # これらのキーの位置情報
                row_before = df_key_sorted.iloc[gap_idx]
                row_after = df_key_sorted.iloc[gap_idx + 1]
                
                print(f"  前のキーの位置: 0x{row_before['_row_position']:08X} ({row_before['_row_position']/(1024*1024):.1f}MB)")
                print(f"  後のキーの位置: 0x{row_after['_row_position']:08X} ({row_after['_row_position']/(1024*1024):.1f}MB)")
                print(f"  位置の差: {row_after['_row_position'] - row_before['_row_position']:,} bytes")

if __name__ == "__main__":
    check_data_continuity()