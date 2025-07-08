#!/usr/bin/env python3
"""
実際に存在するブロックIDを確認
"""

import cudf
from pathlib import Path
import numpy as np

def analyze_actual_blocks():
    """実際に存在するブロックIDを分析"""
    
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    all_blocks = set()
    
    for pf in parquet_files:
        print(f"\n{pf.name}を分析中...")
        df = cudf.read_parquet(pf, columns=['_thread_id'])
        
        # スレッドIDからブロックIDを計算
        thread_ids = df['_thread_id'].to_pandas().values
        block_ids = thread_ids // 256
        unique_blocks = np.unique(block_ids)
        
        print(f"  ユニークブロック数: {len(unique_blocks):,}")
        print(f"  ブロックID範囲: {unique_blocks[0]:,} - {unique_blocks[-1]:,}")
        
        all_blocks.update(unique_blocks)
    
    # 全ブロックの分析
    all_blocks = sorted(all_blocks)
    print(f"\n\n全体のブロック分析:")
    print(f"総ユニークブロック数: {len(all_blocks):,}")
    print(f"ブロックID範囲: {all_blocks[0]:,} - {all_blocks[-1]:,}")
    
    # 大きなギャップを探す
    block_diffs = np.diff(all_blocks)
    large_gaps = np.where(block_diffs > 1000)[0]
    
    if len(large_gaps) > 0:
        print(f"\n大きなギャップ（>1000）: {len(large_gaps)}箇所")
        for i in large_gaps[:10]:  # 最初の10個
            gap_size = block_diffs[i]
            block_before = all_blocks[i]
            block_after = all_blocks[i+1]
            print(f"\n  Block {block_before:,} → {block_after:,}")
            print(f"    ギャップ: {gap_size:,} ブロック")
            print(f"    16進数: 0x{block_before:06X} → 0x{block_after:06X}")
            
            # スキップされたブロックの範囲を表示
            first_skipped = block_before + 1
            last_skipped = block_after - 1
            print(f"    スキップ範囲: Block {first_skipped:,} - {last_skipped:,}")
            
            # 特徴的なブロックIDをチェック
            if block_before == 4095:
                print(f"    → 0xFFF (12ビット境界)")
            if first_skipped == 4096:
                print(f"    → 0x1000から開始")

def check_specific_blocks():
    """特定のブロックが存在するか確認"""
    print("\n\n特定ブロックの存在確認:")
    print("="*80)
    
    # チェックしたいブロックID
    check_blocks = [4095, 4096, 4097, 5461, 5462, 12504, 12505]
    
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    for block_id in check_blocks:
        thread_range = (block_id * 256, (block_id + 1) * 256 - 1)
        found = False
        
        for pf in parquet_files:
            df = cudf.read_parquet(pf, columns=['_thread_id', 'c_custkey'])
            
            # このブロックのスレッドを探す
            mask = ((df['_thread_id'] >= thread_range[0]) & 
                   (df['_thread_id'] <= thread_range[1]))
            
            if mask.sum() > 0:
                found = True
                block_data = df[mask]
                num_threads = block_data['_thread_id'].nunique()
                num_rows = len(block_data)
                
                print(f"\nBlock {block_id} (Thread {thread_range[0]}-{thread_range[1]}):")
                print(f"  ファイル: {pf.name}")
                print(f"  スレッド数: {num_threads}")
                print(f"  行数: {num_rows}")
                
                # いくつかのスレッドIDを表示
                unique_threads = block_data['_thread_id'].unique().to_pandas()[:5]
                print(f"  スレッドID例: {list(unique_threads)}")
        
        if not found:
            print(f"\nBlock {block_id}: 見つかりません")

if __name__ == "__main__":
    analyze_actual_blocks()
    check_specific_blocks()