#!/usr/bin/env python3
"""チャンク境界と欠落データの関係を確認"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_chunk_boundaries():
    """各チャンクの境界を確認"""
    print("=== チャンク境界の確認 ===\n")
    
    # 全チャンクファイルを確認
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    chunk_info = []
    total_rows = 0
    
    for i, pf in enumerate(parquet_files):
        print(f"\n{pf.name}:")
        df = pd.read_parquet(pf, columns=['c_custkey', '_row_position'])
        
        total_rows += len(df)
        
        # 基本情報
        min_key = df['c_custkey'].min()
        max_key = df['c_custkey'].max()
        min_pos = df['_row_position'].min()
        max_pos = df['_row_position'].max()
        
        chunk_info.append({
            'chunk': i,
            'file': pf.name,
            'rows': len(df),
            'min_key': min_key,
            'max_key': max_key,
            'min_pos': min_pos,
            'max_pos': max_pos,
            'min_pos_mb': min_pos / (1024*1024),
            'max_pos_mb': max_pos / (1024*1024)
        })
        
        print(f"  行数: {len(df):,}")
        print(f"  c_custkey範囲: {min_key:,} - {max_key:,}")
        print(f"  位置範囲: 0x{min_pos:08X} - 0x{max_pos:08X}")
        print(f"  位置範囲(MB): {min_pos/(1024*1024):.1f}MB - {max_pos/(1024*1024):.1f}MB")
    
    print(f"\n総行数: {total_rows:,}")
    
    # チャンク間のギャップを確認
    print("\n\n=== チャンク間のギャップ ===")
    for i in range(len(chunk_info) - 1):
        curr = chunk_info[i]
        next = chunk_info[i + 1]
        
        # キーのギャップ
        key_gap = next['min_key'] - curr['max_key']
        # 位置のギャップ  
        pos_gap = next['min_pos'] - curr['max_pos']
        
        print(f"\nチャンク{i} → チャンク{i+1}:")
        print(f"  キーギャップ: {curr['max_key']} → {next['min_key']} (差: {key_gap})")
        print(f"  位置ギャップ: 0x{curr['max_pos']:08X} → 0x{next['min_pos']:08X}")
        print(f"  位置ギャップ(MB): {curr['max_pos_mb']:.1f}MB → {next['min_pos_mb']:.1f}MB")
        print(f"  ギャップサイズ: {pos_gap:,} bytes ({pos_gap/(1024*1024):.1f}MB)")
        
        # 欠落キー3483451がこのギャップに含まれるか
        if curr['max_key'] < 3483451 < next['min_key']:
            print(f"  ⚠️ 欠落キー3483451がこのギャップに含まれています！")
    
    # 128MB〜549MBの範囲を確認
    print("\n\n=== 128MB〜549MBの範囲 ===")
    mb_128 = 128 * 1024 * 1024
    mb_549 = 549 * 1024 * 1024
    
    for info in chunk_info:
        # このチャンクが128MB〜549MBの範囲と重なるか
        if info['max_pos'] >= mb_128 and info['min_pos'] <= mb_549:
            print(f"\n{info['file']}:")
            print(f"  このチャンクは128MB〜549MBの範囲と重なっています")
            print(f"  チャンク範囲: {info['min_pos_mb']:.1f}MB - {info['max_pos_mb']:.1f}MB")
            
            # このチャンク内で128MB〜549MBの範囲のデータを確認
            df = pd.read_parquet(f"output/{info['file']}")
            in_range = df[(df['_row_position'] >= mb_128) & (df['_row_position'] <= mb_549)]
            print(f"  128MB〜549MB範囲内の行数: {len(in_range):,}")
            
            if len(in_range) > 0:
                print(f"  最小キー: {in_range['c_custkey'].min()}")
                print(f"  最大キー: {in_range['c_custkey'].max()}")

def check_missing_in_chunk():
    """欠落データがどのチャンクに属するか確認"""
    print("\n\n=== 欠落データの詳細確認 ===")
    
    # キー3483450（欠落直前）と3483452（欠落直後）を探す
    found_before = False
    found_after = False
    
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        
        # 3483450を含むか
        if 3483450 in df['c_custkey'].values:
            row = df[df['c_custkey'] == 3483450].iloc[0]
            print(f"\nキー3483450が見つかりました: {pf.name}")
            print(f"  位置: 0x{row['_row_position']:08X} ({row['_row_position']/(1024*1024):.1f}MB)")
            print(f"  thread_id: {row['_thread_id']}")
            found_before = True
        
        # 3483452を含むか
        if 3483452 in df['c_custkey'].values:
            row = df[df['c_custkey'] == 3483452].iloc[0]
            print(f"\nキー3483452が見つかりました: {pf.name}")
            print(f"  位置: 0x{row['_row_position']:08X} ({row['_row_position']/(1024*1024):.1f}MB)")
            print(f"  thread_id: {row['_thread_id']}")
            found_after = True
    
    if found_before and found_after:
        print("\n⚠️ 欠落の前後のキーが見つかりました。欠落は処理中に発生しています。")

if __name__ == "__main__":
    check_chunk_boundaries()
    check_missing_in_chunk()