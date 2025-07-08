#!/usr/bin/env python3
"""
スレッド境界での欠落パターンを分析
"""

import cudf
from pathlib import Path
import numpy as np

def analyze_thread_boundaries():
    """スレッド境界パターンを分析"""
    
    # 欠落キーとその前後のスレッド情報
    missing_patterns = [
        (476029, 1048575, 3201426),
        (1227856, 1747626, 3606663),
        (1979731, 2097152, 3809255),
        (2731633, 699049, 2998833),
        (3483451, 1398101, 3404032),
        (4235332, 2446677, 4214529),
        (4987296, 349524, 2796205),
        # 以下は重複あり
        (6491094, 716296, 717145),
        (7243028, 716296, 717145),
        (7994887, 716296, 717145),
        (9498603, 716296, 717145),
        (10250384, 717145, 719349),
        (11002286, 717145, 719349),
        (11754161, 719349, 719434),
    ]
    
    print("欠落パターンの分析:")
    print("="*80)
    
    # スレッドIDの差を計算
    for missing_key, thread_before, thread_after in missing_patterns:
        thread_diff = thread_after - thread_before
        print(f"\n欠落キー: {missing_key}")
        print(f"  前スレッド: {thread_before:8d} (0x{thread_before:08X})")
        print(f"  後スレッド: {thread_after:8d} (0x{thread_after:08X})")
        print(f"  スレッド差: {thread_diff:8d}")
        
        # 特定のパターンを確認
        if thread_before & 0xFFFFF == 0xFFFFF:
            print(f"  → 前スレッドが境界値! (下位20ビットが全て1)")
        if thread_before & 0x3FFFFF == 0x3FFFFF:
            print(f"  → 前スレッドが大きな境界値! (下位22ビットが全て1)")
        if thread_after & 0xFFFF == 0:
            print(f"  → 後スレッドが境界値! (下位16ビットが0)")
    
    # スレッドストライドの計算
    print("\n\nスレッドストライドの推定:")
    print("="*80)
    
    # チャンクごとの情報を読み込み
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    for pf in parquet_files:
        print(f"\n{pf.name}:")
        df = cudf.read_parquet(pf, columns=['_thread_id', '_thread_start_pos', '_thread_end_pos'])
        
        # ユニークなスレッドIDを取得
        unique_threads = df['_thread_id'].unique().to_pandas()
        unique_threads = np.sort(unique_threads)
        
        print(f"  総スレッド数: {len(unique_threads):,}")
        print(f"  スレッドID範囲: {unique_threads[0]:,} - {unique_threads[-1]:,}")
        
        # スレッドストライドを計算（隣接スレッドのstart_posの差）
        thread_info = df.groupby('_thread_id').agg({
            '_thread_start_pos': 'first',
            '_thread_end_pos': 'first'
        }).to_pandas()
        
        # ソートして隣接スレッドの差を計算
        thread_info = thread_info.sort_index()
        if len(thread_info) > 1:
            strides = np.diff(thread_info['_thread_start_pos'])
            unique_strides = np.unique(strides)
            
            print(f"  ストライドパターン:")
            for stride in unique_strides[:5]:  # 最初の5パターンのみ
                count = np.sum(strides == stride)
                print(f"    {stride:6d} bytes: {count:5d} 回")
    
    # 境界値のスレッドを詳しく分析
    print("\n\n境界値スレッドの詳細:")
    print("="*80)
    
    # 問題のあるスレッドIDのリスト
    problem_threads = set()
    for _, thread_before, thread_after in missing_patterns:
        problem_threads.add(thread_before)
        problem_threads.add(thread_after)
    
    for pf in parquet_files:
        df = cudf.read_parquet(pf)
        
        for thread_id in sorted(problem_threads):
            thread_data = df[df['_thread_id'] == thread_id]
            if len(thread_data) > 0:
                print(f"\nスレッド {thread_id} (0x{thread_id:08X}):")
                print(f"  検出行数: {len(thread_data)}")
                
                # スレッドの開始/終了位置
                start_pos = int(thread_data['_thread_start_pos'].iloc[0])
                end_pos = int(thread_data['_thread_end_pos'].iloc[0])
                thread_size = end_pos - start_pos
                
                print(f"  開始位置: 0x{start_pos:08X}")
                print(f"  終了位置: 0x{end_pos:08X}")
                print(f"  サイズ: {thread_size} bytes")
                
                # 特別なパターンをチェック
                if thread_size == 192:
                    print(f"  → 標準ストライド (192 bytes)")
                elif thread_size < 192:
                    print(f"  → 小さいストライド! (通常より{192-thread_size}バイト小さい)")
                
                # 最後の行の位置を確認
                if len(thread_data) > 0:
                    last_row_pos = int(thread_data['_row_position'].iloc[-1])
                    gap = end_pos - last_row_pos
                    print(f"  最後の行位置: 0x{last_row_pos:08X}")
                    print(f"  終了位置までの距離: {gap} bytes")
                    
                    if gap > 100:
                        print(f"  → 大きなギャップ! 次の行を見逃している可能性")

if __name__ == "__main__":
    analyze_thread_boundaries()