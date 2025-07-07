#!/usr/bin/env python3
"""
異常に大きいorderkeyの詳細分析
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_abnormal_keys():
    """異常なキーを持つ行を詳細分析"""
    
    output_dir = Path("/home/ubuntu/gpupgparser/output")
    parquet_files = sorted(output_dir.glob("chunk_*_queue.parquet"))
    
    print("=== 異常なorderkeyの分析 ===\n")
    
    abnormal_count = 0
    
    for file in parquet_files:
        df = pd.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
        
        # 異常値（10億以上）を検出
        abnormal = df[df['lo_orderkey'] > 1e9]
        
        if len(abnormal) > 0:
            print(f"\n{file.name}:")
            print(f"  異常な行数: {len(abnormal):,}")
            print(f"  全体の割合: {len(abnormal)/len(df)*100:.4f}%")
            
            # 詳細表示（最大10行）
            print("\n  詳細（最大10行）:")
            for idx, (_, row) in enumerate(abnormal.head(10).iterrows()):
                orderkey = int(row['lo_orderkey'])
                linenumber = int(row['lo_linenumber'])
                
                # 16進数表示
                print(f"    {idx+1}. lo_orderkey={orderkey:,} (0x{orderkey:016X}), lo_linenumber={linenumber}")
                
                # バイト解析
                orderkey_bytes = orderkey.to_bytes(8, 'big', signed=False)
                print(f"       バイト表現: {' '.join(f'{b:02X}' for b in orderkey_bytes)}")
                
                # 4バイト整数として解釈した場合
                if orderkey <= 0xFFFFFFFF:
                    print(f"       32bit値: {orderkey}")
                else:
                    # 上位4バイトと下位4バイトを確認
                    high_32 = orderkey >> 32
                    low_32 = orderkey & 0xFFFFFFFF
                    print(f"       上位32bit: {high_32} (0x{high_32:08X})")
                    print(f"       下位32bit: {low_32} (0x{low_32:08X})")
                    
                    # もし下位32bitが妥当な範囲なら表示
                    if low_32 < 200_000_000:
                        print(f"       → 下位32bitが妥当な範囲: {low_32:,}")
            
            abnormal_count += len(abnormal)
    
    print(f"\n\n=== サマリー ===")
    print(f"異常なorderkeyを持つ総行数: {abnormal_count:,}")
    
    # 正常な範囲の確認
    print("\n正常なorderkeyの範囲を確認:")
    normal_min = float('inf')
    normal_max = 0
    
    for file in parquet_files[:3]:  # 最初の3ファイル
        df = pd.read_parquet(file, columns=['lo_orderkey'])
        normal = df[df['lo_orderkey'] < 1e9]['lo_orderkey']
        if len(normal) > 0:
            normal_min = min(normal_min, normal.min())
            normal_max = max(normal_max, normal.max())
    
    print(f"  正常範囲: {normal_min:,} - {normal_max:,}")

def main():
    analyze_abnormal_keys()

if __name__ == "__main__":
    main()