#!/usr/bin/env python3
"""欠落thread_idと前後のthread_idの境界分析"""

import pandas as pd

def analyze_boundaries():
    """欠落thread_idと前後のthread_idの境界を分析"""
    print("=== 欠落thread_idと前後のthread_idの境界分析 ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # すべてのユニークなthread情報を取得
    unique_threads = chunk0[['_thread_id', '_thread_start_pos', '_thread_end_pos']].drop_duplicates('_thread_id')
    unique_threads = unique_threads.sort_values('_thread_id')
    
    # 64MB単位の境界
    mb_64 = 64 * 1024 * 1024
    
    print("欠落thread_id一覧と前後のthread情報:\n")
    print("="*120)
    
    for missing_tid in missing_threads:
        print(f"\n欠落thread_id: {missing_tid}")
        print("-"*120)
        
        # 前後のthread_id
        before_tid = missing_tid - 1
        after_tid = missing_tid + 1
        
        # 前のthread情報を取得
        before_info = unique_threads[unique_threads['_thread_id'] == before_tid]
        if len(before_info) > 0:
            before_row = before_info.iloc[0]
            before_start = int(before_row['_thread_start_pos'])
            before_end = int(before_row['_thread_end_pos'])
            
            print(f"\n前thread (id={before_tid}):")
            print(f"  start_pos: 0x{before_start:08X} ({before_start/(1024*1024):8.4f}MB)")
            print(f"  end_pos:   0x{before_end:08X} ({before_end/(1024*1024):8.4f}MB)")
            
            # どの64MB境界を跨いでいるか確認
            before_start_block = before_start // mb_64
            before_end_block = before_end // mb_64
            
            if before_start_block != before_end_block:
                boundary = (before_end_block) * mb_64
                print(f"  ⚠️ {boundary/(1024*1024):.0f}MB境界を跨ぐ！")
                print(f"     境界前: {boundary - before_start}バイト")
                print(f"     境界後: {before_end - boundary}バイト")
        
        # 後のthread情報を取得
        after_info = unique_threads[unique_threads['_thread_id'] == after_tid]
        if len(after_info) > 0:
            after_row = after_info.iloc[0]
            after_start = int(after_row['_thread_start_pos'])
            after_end = int(after_row['_thread_end_pos'])
            
            print(f"\n後thread (id={after_tid}):")
            print(f"  start_pos: 0x{after_start:08X} ({after_start/(1024*1024):8.4f}MB)")
            print(f"  end_pos:   0x{after_end:08X} ({after_end/(1024*1024):8.4f}MB)")
            
            # どの64MBブロックに属するか
            after_start_block = after_start // mb_64
            
            # 64MBブロックの最初のthreadか確認
            block_start = after_start_block * mb_64
            distance_from_block_start = after_start - block_start
            
            print(f"  64MBブロック: {after_start_block * 64}MB〜{(after_start_block + 1) * 64}MB")
            print(f"  ブロック開始からの距離: {distance_from_block_start}バイト")
            
            if distance_from_block_start < 1024:  # 1KB以内なら「最初」と判定
                print(f"  ✓ このブロックの最初のthread")
    
    # 各64MBブロックの最初のthreadを確認
    print("\n\n=== 各64MBブロックの最初のthread ===")
    print("="*120)
    
    # 各ブロックの最初のthreadを探す
    blocks_to_check = [2, 3, 4, 6, 7]  # 128MB, 192MB, 256MB, 384MB, 448MB
    
    for block_idx in blocks_to_check:
        block_start = block_idx * mb_64
        block_end = (block_idx + 1) * mb_64
        
        print(f"\n{block_start/(1024*1024):.0f}MB〜{block_end/(1024*1024):.0f}MBブロック:")
        
        # このブロック内のthreadを探す
        block_threads = unique_threads[
            (unique_threads['_thread_start_pos'] >= block_start) &
            (unique_threads['_thread_start_pos'] < block_end)
        ].sort_values('_thread_start_pos')
        
        if len(block_threads) > 0:
            # 最初の5つのthreadを表示
            print("  最初のthread:")
            for i, (_, row) in enumerate(block_threads.head(5).iterrows()):
                tid = int(row['_thread_id'])
                start = int(row['_thread_start_pos'])
                distance = start - block_start
                
                mark = ""
                if tid in missing_threads:
                    mark = " *** 欠落 ***"
                elif tid == 1048577:  # 192MBブロックの最初
                    mark = " ← 注目"
                elif tid == 1398103:  # 256MBブロックの最初
                    mark = " ← 注目"
                
                print(f"    {i+1}. thread_id {tid}: ブロック開始から+{distance}バイト{mark}")
    
    # まとめ
    print("\n\n=== まとめ ===")
    print("="*120)
    
    print("\n境界を跨ぐthread_idと欠落thread_idの関係:")
    print("\n| 境界 | 境界を跨ぐthread | 欠落thread | 欠落後の最初のthread |")
    print("|------|------------------|------------|---------------------|")
    print("| 192MB | 1048575 | 1048576 | 1048577（192MBブロックの最初） |")
    print("| 256MB | 1398101 | 1398102 | 1398103（256MBブロックの最初） |")
    print("| 384MB | 2097152 | 2097153 | 2097154（384MBブロック内） |")
    print("| 448MB | 2446678 | 2446679 | 2446680（448MBブロック内） |")

if __name__ == "__main__":
    analyze_boundaries()