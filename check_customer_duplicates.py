#!/usr/bin/env python3
"""
customerテーブルの重複を詳細分析
"""

import cudf
import numpy as np

print("=== customerテーブル重複分析 ===\n")

# Parquetファイル読み込み (cuDFを使用)
df = cudf.read_parquet("/home/ubuntu/gpupgparser/output/chunk_0_queue.parquet")
print(f"Parquet総行数: {len(df):,}")

# c_custkeyの重複をチェック
if 'c_custkey' in df.columns:
    unique_keys = df['c_custkey'].nunique()
    total_keys = len(df)
    duplicate_count = total_keys - unique_keys
    
    print(f"\nc_custkey分析:")
    print(f"  総キー数: {total_keys:,}")
    print(f"  ユニークキー数: {unique_keys:,}")
    print(f"  重複キー数: {duplicate_count:,}")
    print(f"  重複率: {duplicate_count / total_keys * 100:.1f}%")
    
    # GPU検出行数との差を計算
    gpu_detected = 6_015_118
    buffer_size = 4_638_072
    difference = gpu_detected - buffer_size
    
    print(f"\nGPU検出との比較:")
    print(f"  GPU検出行数: {gpu_detected:,}")
    print(f"  バッファサイズ: {buffer_size:,}")
    print(f"  差分: {difference:,}")
    print(f"  Parquet内の重複数: {duplicate_count:,}")
    
    if abs(difference - duplicate_count) < 100:  # 誤差100行以内
        print(f"\n✅ 差分({difference:,})と重複数({duplicate_count:,})がほぼ一致！")
        print("   → GPUは重複行も含めて検出している")
    
    # 重複キーの詳細 (cuDFではto_pandasが必要な場合がある)
    try:
        duplicated_mask = df.duplicated(subset=['c_custkey'], keep=False)
        duplicated_df = df[duplicated_mask]
        
        if len(duplicated_df) > 0:
            print(f"\n重複している行の例（最初の10行）:")
            print(duplicated_df.head(10)[['c_custkey', 'c_name']].to_pandas())
            
            # 最も重複が多いキー
            dup_counts = df[duplicated_mask]['c_custkey'].value_counts()
            print(f"\n最も重複が多いキーTOP5:")
            for i in range(min(5, len(dup_counts))):
                key = dup_counts.index[i].item()
                count = dup_counts.iloc[i].item()
                print(f"  c_custkey={key}: {count}回")
    except Exception as e:
        print(f"\n重複詳細分析でエラー: {e}")

# 推定値の修正案
print(f"\n=== バッファサイズ推定の修正案 ===")
print(f"現在の推定: {buffer_size:,}行 (estimated_rows × 105%)")
print(f"必要サイズ: {gpu_detected:,}行 (実際のGPU検出数)")
safety_factor = gpu_detected / (buffer_size / 1.05) 
print(f"推奨安全係数: {safety_factor:.2f} (現在の1.05の代わりに)")