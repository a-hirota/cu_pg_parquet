#!/usr/bin/env python3
"""14個の欠落キーの位置パターンを分析"""

import pandas as pd
import numpy as np

def find_positions():
    """欠落キーの前後の位置を確認"""
    print("=== 14個の欠落キーの位置分析 ===\n")
    
    # 欠落キー
    missing_keys = [476029, 1227856, 2731633, 3483451, 4235332, 4987296, 5739177,
                   6491094, 7243028, 7994887, 8746794, 9498603, 10250384, 11754161]
    
    # チャンクファイル
    chunks = {
        0: pd.read_parquet("output/customer_chunk_0_queue.parquet"),
        1: pd.read_parquet("output/customer_chunk_1_queue.parquet")
    }
    
    # Decimal型をintに変換
    for chunk in chunks.values():
        chunk['c_custkey'] = chunk['c_custkey'].astype('int64')
    
    # 各欠落キーの前後を確認
    for missing_key in missing_keys:
        # どちらのチャンクに属するか
        chunk_id = 0 if missing_key < 6015105 else 1
        df = chunks[chunk_id]
        
        # 前後のキーを探す
        before_key = missing_key - 1
        after_key = missing_key + 1
        
        before_row = df[df['c_custkey'] == before_key]
        after_row = df[df['c_custkey'] == after_key]
        
        print(f"\n欠落キー {missing_key} (チャンク{chunk_id}):")
        
        if len(before_row) > 0:
            pos = before_row.iloc[0]['_row_position']
            print(f"  前のキー({before_key}): 位置 0x{pos:08X} ({pos/(1024*1024):.1f}MB)")
        
        if len(after_row) > 0:
            pos = after_row.iloc[0]['_row_position']
            print(f"  後のキー({after_key}): 位置 0x{pos:08X} ({pos/(1024*1024):.1f}MB)")
            
        if len(before_row) > 0 and len(after_row) > 0:
            gap = after_row.iloc[0]['_row_position'] - before_row.iloc[0]['_row_position']
            print(f"  位置ギャップ: {gap:,} bytes ({gap/(1024*1024):.1f}MB)")
    
    # 位置のパターンを分析
    print("\n\n=== 位置パターンの分析 ===")
    
    # チャンク0の欠落位置
    chunk0_missing = [k for k in missing_keys if k < 6015105]
    positions = []
    
    for key in chunk0_missing:
        after_row = chunks[0][chunks[0]['c_custkey'] == key + 1]
        if len(after_row) > 0:
            positions.append(after_row.iloc[0]['_row_position'])
    
    if positions:
        positions = np.array(positions)
        diffs = np.diff(positions)
        
        print(f"\nチャンク0の欠落位置:")
        for i, pos in enumerate(positions):
            print(f"  {chunk0_missing[i]}: {pos/(1024*1024):.1f}MB")
        
        print(f"\n位置間隔:")
        for i, diff in enumerate(diffs):
            print(f"  {chunk0_missing[i]}→{chunk0_missing[i+1]}: {diff/(1024*1024):.1f}MB")
        
        # 128MBごとの可能性をチェック
        mb_128 = 128 * 1024 * 1024
        for i, pos in enumerate(positions):
            multiple = pos / mb_128
            print(f"\n{chunk0_missing[i]}の位置: {multiple:.3f} × 128MB")

if __name__ == "__main__":
    find_positions()