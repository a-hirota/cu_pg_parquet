#!/usr/bin/env python3
"""欠落キーが実際に存在するか確認"""

import pandas as pd
import numpy as np

def check_keys_exist():
    """欠落と思われるキーが実際に存在するか確認"""
    print("=== 欠落キーの存在確認 ===\n")
    
    # 「欠落」とされたキー
    missing_keys = [476029, 1227856, 2731633, 3483451, 4235332, 4987296, 5739177,
                   6491094, 7243028, 7994887, 8746794, 9498603, 10250384, 11754161]
    
    # 両方のチャンクを読み込み
    print("Parquetファイルを読み込み中...")
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    chunk1 = pd.read_parquet("output/customer_chunk_1_queue.parquet")
    
    # Decimal型をintに変換
    chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
    chunk1['c_custkey'] = chunk1['c_custkey'].astype('int64')
    
    # 全データを結合
    all_data = pd.concat([chunk0, chunk1], ignore_index=True)
    print(f"総行数: {len(all_data):,}")
    
    # 各キーの存在を確認
    print("\n各キーの存在確認:")
    for key in missing_keys:
        rows = all_data[all_data['c_custkey'] == key]
        if len(rows) > 0:
            print(f"\n✓ c_custkey={key} は存在します！")
            for idx, row in rows.iterrows():
                print(f"  位置: 0x{row['_row_position']:08X} ({row['_row_position']/(1024*1024):.1f}MB)")
                print(f"  チャンク: {0 if row['_row_position'] in chunk0['_row_position'].values else 1}")
        else:
            print(f"\n✗ c_custkey={key} は本当に存在しません")
    
    # 全体のキーの重複を確認
    print("\n\n=== キーの重複確認 ===")
    duplicates = all_data[all_data.duplicated(subset=['c_custkey'], keep=False)]
    if len(duplicates) > 0:
        print(f"重複キー数: {len(duplicates.groupby('c_custkey'))}")
        print("\n重複例（最初の5個）:")
        for key, group in list(duplicates.groupby('c_custkey'))[:5]:
            print(f"\nc_custkey={key}:")
            for idx, row in group.iterrows():
                print(f"  位置: 0x{row['_row_position']:08X} ({row['_row_position']/(1024*1024):.1f}MB)")
    else:
        print("重複キーはありません")
    
    # ユニークキー数を確認
    unique_keys = set(all_data['c_custkey'])
    print(f"\n\n=== 最終確認 ===")
    print(f"ユニークキー数: {len(unique_keys):,}")
    print(f"期待値（12,030,000）との差: {12030000 - len(unique_keys):,}")
    
    # 本当に欠落しているキーを探す
    actual_missing = []
    for i in range(1, 12030001):
        if i not in unique_keys:
            actual_missing.append(i)
    
    print(f"\n本当に欠落しているキー数: {len(actual_missing)}")
    if actual_missing:
        print("欠落キー:")
        for key in actual_missing[:20]:
            print(f"  {key}")

def analyze_position_order():
    """_row_positionの順序を分析"""
    print("\n\n=== _row_positionの順序分析 ===")
    
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
    
    # _row_positionでソート
    chunk0_sorted = chunk0.sort_values('_row_position')
    
    # c_custkeyの順序を確認
    print("_row_positionでソートした時のc_custkeyの順序（最初の20行）:")
    for i, row in chunk0_sorted.head(20).iterrows():
        print(f"  pos=0x{row['_row_position']:08X} ({row['_row_position']/(1024*1024):6.1f}MB): c_custkey={row['c_custkey']}")
    
    # 位置が飛んでいる箇所を探す
    positions = chunk0_sorted['_row_position'].values
    diffs = np.diff(positions)
    large_gaps = np.where(diffs > 1000000)[0]  # 1MB以上のギャップ
    
    print(f"\n\n1MB以上のギャップ: {len(large_gaps)}個")
    if len(large_gaps) > 0:
        print("\n最初の5個のギャップ:")
        for idx in large_gaps[:5]:
            before = chunk0_sorted.iloc[idx]
            after = chunk0_sorted.iloc[idx + 1]
            gap = after['_row_position'] - before['_row_position']
            
            print(f"\n  ギャップ {gap/(1024*1024):.1f}MB:")
            print(f"    前: pos=0x{before['_row_position']:08X}, c_custkey={before['c_custkey']}")
            print(f"    後: pos=0x{after['_row_position']:08X}, c_custkey={after['c_custkey']}")

if __name__ == "__main__":
    check_keys_exist()
    analyze_position_order()