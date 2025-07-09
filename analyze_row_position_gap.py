#!/usr/bin/env python3
"""_row_positionのギャップを詳細分析"""

import pandas as pd
import numpy as np

def analyze_position_gaps():
    """row_positionのギャップを分析"""
    print("=== row_positionギャップ分析 ===\n")
    
    # Parquetファイルを読み込み
    df = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # row_positionでソート
    df_sorted = df.sort_values('_row_position')
    
    # 隣接する行の間のギャップを計算
    positions = df_sorted['_row_position'].values
    gaps = np.diff(positions)
    
    # 通常の行サイズを推定（中央値）
    normal_gap = np.median(gaps)
    print(f"通常の行間隔（中央値）: {normal_gap:.0f} bytes")
    
    # 大きなギャップを検出（通常の10倍以上）
    large_gaps_mask = gaps > normal_gap * 10
    large_gaps_indices = np.where(large_gaps_mask)[0]
    
    print(f"\n大きなギャップ数: {len(large_gaps_indices)}")
    
    if len(large_gaps_indices) > 0:
        print("\n最大のギャップTop 5:")
        # ギャップサイズでソート
        sorted_indices = large_gaps_indices[np.argsort(gaps[large_gaps_indices])[::-1]][:5]
        
        for idx in sorted_indices:
            gap_size = gaps[idx]
            pos_before = positions[idx]
            pos_after = positions[idx + 1]
            
            # 対応する行データ
            row_before = df_sorted.iloc[idx]
            row_after = df_sorted.iloc[idx + 1]
            
            print(f"\nギャップ #{idx}:")
            print(f"  サイズ: {gap_size:,} bytes ({gap_size/1024/1024:.1f}MB)")
            print(f"  前の行: c_custkey={row_before['c_custkey']}, pos=0x{pos_before:08X} ({pos_before/1024/1024:.1f}MB)")
            print(f"  後の行: c_custkey={row_after['c_custkey']}, pos=0x{pos_after:08X} ({pos_after/1024/1024:.1f}MB)")
            
            # 256MB境界との関係
            mb_256 = 256 * 1024 * 1024
            if pos_before < mb_256 < pos_after:
                print(f"  ⚠️ 256MB境界（0x{mb_256:08X}）を跨いでいます")
            
            # 他の重要な境界
            for mb in [128, 256, 384, 512]:
                boundary = mb * 1024 * 1024
                if pos_before < boundary < pos_after:
                    print(f"  → {mb}MB境界を跨いでいます")
    
    # 欠落キー3483451の位置を確認
    print("\n\n=== 欠落キー3483451付近の詳細 ===")
    
    # キー3483450と3483452を探す
    key_before = df[df['c_custkey'] == 3483450]
    key_after = df[df['c_custkey'] == 3483452]
    
    if len(key_before) > 0 and len(key_after) > 0:
        pos_before = key_before['_row_position'].iloc[0]
        pos_after = key_after['_row_position'].iloc[0]
        gap = pos_after - pos_before
        
        print(f"キー3483450: pos=0x{pos_before:08X} ({pos_before/1024/1024:.1f}MB)")
        print(f"キー3483452: pos=0x{pos_after:08X} ({pos_after/1024/1024:.1f}MB)")
        print(f"ギャップ: {gap:,} bytes ({gap/1024/1024:.1f}MB)")
        
        # thread_strideとの関係
        thread_stride = 1150  # 実際の値
        skipped_threads = gap // thread_stride
        print(f"\nthread_stride({thread_stride})で計算:")
        print(f"  スキップされたスレッド数: {skipped_threads:,.0f}")
    
    # 全体のカバレッジ確認
    print("\n\n=== データカバレッジ ===")
    min_pos = df['_row_position'].min()
    max_pos = df['_row_position'].max()
    total_range = max_pos - min_pos
    
    print(f"最小位置: 0x{min_pos:08X} ({min_pos/1024/1024:.1f}MB)")
    print(f"最大位置: 0x{max_pos:08X} ({max_pos/1024/1024:.1f}MB)")
    print(f"範囲: {total_range:,} bytes ({total_range/1024/1024:.1f}MB)")
    
    # 実際にカバーされているバイト数
    actual_coverage = len(df) * normal_gap
    coverage_rate = actual_coverage / total_range * 100
    print(f"\nカバー率: {coverage_rate:.1f}%")

if __name__ == "__main__":
    analyze_position_gaps()