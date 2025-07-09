#!/usr/bin/env python3
"""チャンクの混在問題を分析"""

import pandas as pd
import numpy as np

def analyze_chunks():
    """両方のチャンクを分析"""
    print("=== チャンク混在問題の分析 ===\n")
    
    # 両方のチャンクを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    chunk1 = pd.read_parquet("output/customer_chunk_1_queue.parquet")
    
    # Decimal型をintに変換
    chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
    chunk1['c_custkey'] = chunk1['c_custkey'].astype('int64')
    
    print(f"チャンク0: {len(chunk0):,}行")
    print(f"  c_custkey範囲: {chunk0['c_custkey'].min():,} - {chunk0['c_custkey'].max():,}")
    
    print(f"\nチャンク1: {len(chunk1):,}行") 
    print(f"  c_custkey範囲: {chunk1['c_custkey'].min():,} - {chunk1['c_custkey'].max():,}")
    
    # 重複を確認
    keys0 = set(chunk0['c_custkey'])
    keys1 = set(chunk1['c_custkey'])
    
    overlap = keys0 & keys1
    print(f"\n重複キー数: {len(overlap):,}")
    
    if overlap:
        print(f"重複キーの例（最初の10個）:")
        for key in sorted(overlap)[:10]:
            print(f"  {key}")
    
    # 本来のチャンク境界（6015105）
    boundary = 6015105
    
    # チャンク0に含まれる境界以降のキー
    chunk0_beyond = chunk0[chunk0['c_custkey'] >= boundary]
    print(f"\nチャンク0に含まれる境界({boundary})以降のキー: {len(chunk0_beyond):,}個")
    
    # チャンク1に含まれる境界未満のキー
    chunk1_before = chunk1[chunk1['c_custkey'] < boundary]
    print(f"チャンク1に含まれる境界({boundary})未満のキー: {len(chunk1_before):,}個")
    
    # 全体のキー数を確認
    all_keys = keys0 | keys1
    print(f"\n全ユニークキー数: {len(all_keys):,}")
    print(f"期待値（12,030,000）との差: {12030000 - len(all_keys):,}")
    
    # 実際の欠落を確認
    all_keys_sorted = sorted(all_keys)
    actual_missing = []
    
    for i in range(1, 12030001):
        if i not in all_keys:
            actual_missing.append(i)
    
    print(f"\n実際の欠落キー数: {len(actual_missing)}")
    
    if actual_missing:
        print(f"実際の欠落キー（最初の20個）:")
        for i, key in enumerate(actual_missing[:20]):
            print(f"  {i+1:2d}. {key}")
    
    # 15個の欠落キーを特定
    if len(actual_missing) <= 20:
        print(f"\n✓ 実際の欠落は{len(actual_missing)}個のみです！")
        print("欠落キー:")
        for key in actual_missing:
            # どちらのチャンクに属するべきか
            expected_chunk = 0 if key < boundary else 1
            print(f"  {key} (本来チャンク{expected_chunk})")

if __name__ == "__main__":
    analyze_chunks()