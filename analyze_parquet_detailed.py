#!/usr/bin/env python3
"""
Parquetファイルの詳細分析（PostgreSQL不要）
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_all_parquet_files():
    """全Parquetファイルを分析"""
    
    output_dir = Path("/home/ubuntu/gpupgparser/output")
    parquet_files = sorted(output_dir.glob("chunk_*_queue.parquet"))
    
    print(f"見つかったParquetファイル数: {len(parquet_files)}")
    print("="*80)
    
    total_rows = 0
    total_unique = 0
    all_stats = []
    
    # 各ファイルの統計情報を収集
    for i, file in enumerate(parquet_files):
        print(f"\n{file.name}:")
        df = pd.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
        
        row_count = len(df)
        unique_count = df.drop_duplicates(['lo_orderkey', 'lo_linenumber']).shape[0]
        duplicate_count = row_count - unique_count
        
        min_orderkey = df['lo_orderkey'].min()
        max_orderkey = df['lo_orderkey'].max()
        min_linenumber = df['lo_linenumber'].min()
        max_linenumber = df['lo_linenumber'].max()
        
        # 異常に大きいorderkeyをチェック
        large_keys = df[df['lo_orderkey'] > 1e12]
        
        print(f"  行数: {row_count:,}")
        print(f"  ユニークキー数: {unique_count:,}")
        print(f"  重複数: {duplicate_count:,}")
        print(f"  lo_orderkey範囲: {min_orderkey:,} - {max_orderkey:,}")
        print(f"  lo_linenumber範囲: {min_linenumber} - {max_linenumber}")
        
        if len(large_keys) > 0:
            print(f"  ⚠️ 異常に大きいorderkey検出: {len(large_keys)}行")
            print(f"     例: {large_keys['lo_orderkey'].iloc[0]:,}")
        
        total_rows += row_count
        total_unique += unique_count
        
        all_stats.append({
            'file': file.name,
            'rows': row_count,
            'unique': unique_count,
            'duplicates': duplicate_count,
            'min_orderkey': min_orderkey,
            'max_orderkey': max_orderkey,
            'large_keys': len(large_keys)
        })
        
        # 重複キーの詳細
        if duplicate_count > 0:
            duplicates = df[df.duplicated(['lo_orderkey', 'lo_linenumber'], keep=False)]
            dup_groups = duplicates.groupby(['lo_orderkey', 'lo_linenumber']).size()
            print(f"  重複パターン:")
            print(f"    最大重複数: {dup_groups.max()}")
            print(f"    重複グループ数: {len(dup_groups)}")
            
            # 重複例を表示
            if len(dup_groups) > 0:
                example_key = dup_groups.idxmax()
                example_rows = df[(df['lo_orderkey'] == example_key[0]) & 
                                (df['lo_linenumber'] == example_key[1])]
                print(f"    重複例: lo_orderkey={example_key[0]}, lo_linenumber={example_key[1]} → {len(example_rows)}回")
    
    # サマリー
    print("\n" + "="*80)
    print("=== 全体サマリー ===")
    print(f"総ファイル数: {len(parquet_files)}")
    print(f"総行数: {total_rows:,}")
    print(f"総ユニークキー数: {total_unique:,}")
    print(f"総重複数: {total_rows - total_unique:,}")
    print(f"平均行数/ファイル: {total_rows / len(parquet_files):,.0f}")
    
    # 各ファイルの行数の分布
    row_counts = [s['rows'] for s in all_stats]
    print(f"\n行数分布:")
    print(f"  最小: {min(row_counts):,}")
    print(f"  最大: {max(row_counts):,}")
    print(f"  標準偏差: {np.std(row_counts):,.0f}")
    
    # 異常値のあるファイル
    files_with_large_keys = [s for s in all_stats if s['large_keys'] > 0]
    if files_with_large_keys:
        print(f"\n⚠️ 異常に大きいorderkeyを含むファイル: {len(files_with_large_keys)}個")
        for s in files_with_large_keys:
            print(f"  {s['file']}: {s['large_keys']}行")
    
    # 期待値との比較（246M行の場合）
    expected_total = 246_012_324
    coverage = total_rows / expected_total * 100
    print(f"\n期待値との比較:")
    print(f"  期待総行数: {expected_total:,}")
    print(f"  実際の総行数: {total_rows:,}")
    print(f"  カバー率: {coverage:.2f}%")
    print(f"  不足行数: {expected_total - total_rows:,}")

def main():
    analyze_all_parquet_files()

if __name__ == "__main__":
    main()