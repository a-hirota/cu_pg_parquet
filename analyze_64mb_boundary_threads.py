#!/usr/bin/env python3
"""64MB境界での欠落thread_idを詳細分析"""

import pandas as pd
import numpy as np

def analyze_boundary():
    """64MB境界での欠落thread_idを詳細分析"""
    print("=== 64MB境界での欠落thread_id分析 ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    # すべてのユニークなthread情報を取得
    unique_threads = chunk0[['_thread_id', '_thread_start_pos', '_thread_end_pos']].drop_duplicates('_thread_id')
    unique_threads = unique_threads.sort_values('_thread_id')
    
    # 64MB境界
    mb_64 = 64 * 1024 * 1024
    
    # 各欠落thread_idの詳細分析
    for missing_tid in missing_threads:
        print(f"\n{'='*60}")
        print(f"欠落thread_id: {missing_tid}")
        print(f"{'='*60}")
        
        # 前後のthread情報を取得
        before_tid = missing_tid - 1
        after_tid = missing_tid + 1
        
        before_info = unique_threads[unique_threads['_thread_id'] == before_tid]
        after_info = unique_threads[unique_threads['_thread_id'] == after_tid]
        
        if len(before_info) > 0 and len(after_info) > 0:
            before_row = before_info.iloc[0]
            after_row = after_info.iloc[0]
            
            # 位置情報
            before_start = int(before_row['_thread_start_pos'])
            before_end = int(before_row['_thread_end_pos'])
            after_start = int(after_row['_thread_start_pos'])
            after_end = int(after_row['_thread_end_pos'])
            
            print(f"\n前thread (id={before_tid}):")
            print(f"  位置: 0x{before_start:08X} - 0x{before_end:08X}")
            print(f"  MB: {before_start/(1024*1024):.2f}MB - {before_end/(1024*1024):.2f}MB")
            
            print(f"\n後thread (id={after_tid}):")
            print(f"  位置: 0x{after_start:08X} - 0x{after_end:08X}")
            print(f"  MB: {after_start/(1024*1024):.2f}MB - {after_end/(1024*1024):.2f}MB")
            
            # ギャップ分析
            gap_size = after_start - before_end
            print(f"\nギャップ:")
            print(f"  サイズ: {gap_size}バイト")
            print(f"  位置: 0x{before_end:08X} - 0x{after_start:08X}")
            
            # 64MB境界との関係
            # before_endが属する64MBブロック
            before_block = before_end // mb_64
            # after_startが属する64MBブロック  
            after_block = after_start // mb_64
            
            if before_block != after_block:
                print(f"\n⚠️ 64MB境界を跨ぐ:")
                print(f"  前thread終了: {before_block * 64}MBブロック内")
                print(f"  後thread開始: {after_block * 64}MBブロック内")
                
                # 境界位置
                boundary = after_block * mb_64
                print(f"  境界位置: 0x{boundary:08X} ({boundary/(1024*1024):.0f}MB)")
                
                # 境界前後の状況
                before_boundary_gap = boundary - before_end
                after_boundary_gap = after_start - boundary
                
                print(f"\n境界前後のギャップ:")
                print(f"  境界前: {before_boundary_gap}バイト (0x{before_boundary_gap:X})")
                print(f"  境界後: {after_boundary_gap}バイト (0x{after_boundary_gap:X})")
                
                # 欠落thread_idが処理すべきだった位置を推定
                expected_start = before_end
                expected_end = expected_start + 192  # 固定サイズ
                
                print(f"\n欠落thread_idの推定位置:")
                print(f"  開始: 0x{expected_start:08X} ({expected_start/(1024*1024):.2f}MB)")
                print(f"  終了: 0x{expected_end:08X} ({expected_end/(1024*1024):.2f}MB)")
                
                if expected_start < boundary <= expected_end:
                    print(f"  ⚠️ 64MB境界がthread処理範囲内を通過！")
    
    # 64MB境界での全体的なパターンを確認
    print(f"\n\n{'='*60}")
    print("64MB境界でのthread_id分布パターン")
    print(f"{'='*60}\n")
    
    # 各64MB境界付近のthread_idを確認
    boundaries_to_check = [2, 3, 4, 6, 7]  # 128MB, 192MB, 256MB, 384MB, 448MB
    
    for boundary_idx in boundaries_to_check:
        boundary_pos = boundary_idx * mb_64
        
        print(f"\n{boundary_pos/(1024*1024):.0f}MB境界:")
        
        # この境界前後のthreadを探す
        near_boundary = unique_threads[
            (unique_threads['_thread_start_pos'] > boundary_pos - 1024) &
            (unique_threads['_thread_start_pos'] < boundary_pos + 1024)
        ].sort_values('_thread_start_pos')
        
        if len(near_boundary) > 0:
            for _, row in near_boundary.iterrows():
                tid = int(row['_thread_id'])
                start = int(row['_thread_start_pos'])
                distance = start - boundary_pos
                
                mark = ""
                if tid in missing_threads:
                    mark = " *** 欠落 ***"
                elif tid in [m-1 for m in missing_threads]:
                    mark = " (欠落の前)"
                elif tid in [m+1 for m in missing_threads]:
                    mark = " (欠落の後)"
                
                print(f"  thread_id {tid}: 境界から {distance:+6d}バイト{mark}")
        else:
            # 境界位置にthreadがない場合、最も近いthreadを探す
            before_boundary = unique_threads[unique_threads['_thread_end_pos'] <= boundary_pos].tail(1)
            after_boundary = unique_threads[unique_threads['_thread_start_pos'] >= boundary_pos].head(1)
            
            if len(before_boundary) > 0 and len(after_boundary) > 0:
                before_tid = int(before_boundary.iloc[0]['_thread_id'])
                after_tid = int(after_boundary.iloc[0]['_thread_id'])
                
                print(f"  境界前の最後: thread_id {before_tid}")
                print(f"  境界後の最初: thread_id {after_tid}")
                
                if after_tid - before_tid > 1:
                    print(f"  ⚠️ thread_idギャップ: {after_tid - before_tid}")
                    for tid in range(before_tid + 1, after_tid):
                        if tid in missing_threads:
                            print(f"    欠落: thread_id {tid}")

if __name__ == "__main__":
    analyze_boundary()