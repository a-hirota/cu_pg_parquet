#!/usr/bin/env python3
"""欠落thread_idの予定位置を計算"""

import pandas as pd
import numpy as np

def calculate_positions():
    """欠落thread_idの予定start_pos、end_posを計算"""
    print("=== 欠落thread_idの予定位置計算 ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # thread_stride（固定値）
    thread_stride = 1150  # バイト
    
    print(f"thread_stride: {thread_stride}バイト\n")
    
    # 各欠落thread_idの予定位置を計算
    print("欠落thread_idの予定位置:")
    print("-" * 80)
    
    for tid in missing_threads:
        # 開始位置 = thread_id * thread_stride
        start_pos = tid * thread_stride
        
        # 終了位置 = start_pos + thread_stride
        end_pos = start_pos + thread_stride
        
        print(f"\nthread_id {tid}:")
        print(f"  予定start_pos: 0x{start_pos:08X} ({start_pos:,}バイト, {start_pos/(1024*1024):.2f}MB)")
        print(f"  予定end_pos:   0x{end_pos:08X} ({end_pos:,}バイト, {end_pos/(1024*1024):.2f}MB)")
        print(f"  処理範囲サイズ: {thread_stride}バイト")
        
        # 64MB境界との関係を確認
        mb_64 = 64 * 1024 * 1024
        if start_pos % mb_64 == 0:
            print(f"  ⚠️ start_posが64MB境界に一致: {start_pos // mb_64} * 64MB")
        elif end_pos % mb_64 == 0:
            print(f"  ⚠️ end_posが64MB境界に一致: {end_pos // mb_64} * 64MB")
        
        # 256MB境界との関係を確認
        mb_256 = 256 * 1024 * 1024
        if start_pos % mb_256 == 0:
            print(f"  ⚠️ start_posが256MB境界に一致: {start_pos // mb_256} * 256MB")
        elif end_pos % mb_256 == 0:
            print(f"  ⚠️ end_posが256MB境界に一致: {end_pos // mb_256} * 256MB")
    
    # 実際のデータと比較
    print("\n\n=== 実際のデータとの比較 ===")
    
    # Parquetファイルから前後のthread_idのデータを取得
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    for tid in missing_threads:
        print(f"\nthread_id {tid}周辺:")
        
        # 前のthread_id
        before_tid = tid - 1
        if before_tid in chunk0['_thread_id'].values:
            before_rows = chunk0[chunk0['_thread_id'] == before_tid]
            if len(before_rows) > 0:
                row = before_rows.iloc[0]
                print(f"  前(thread_id={before_tid}):")
                print(f"    実際のstart_pos: 0x{row['_thread_start_pos']:08X}")
                print(f"    実際のend_pos:   0x{row['_thread_end_pos']:08X}")
                print(f"    予想のstart_pos: 0x{before_tid * thread_stride:08X}")
                print(f"    予想のend_pos:   0x{(before_tid + 1) * thread_stride:08X}")
        
        # 後のthread_id
        after_tid = tid + 1
        if after_tid in chunk0['_thread_id'].values:
            after_rows = chunk0[chunk0['_thread_id'] == after_tid]
            if len(after_rows) > 0:
                row = after_rows.iloc[0]
                print(f"  後(thread_id={after_tid}):")
                print(f"    実際のstart_pos: 0x{row['_thread_start_pos']:08X}")
                print(f"    実際のend_pos:   0x{row['_thread_end_pos']:08X}")
                print(f"    予想のstart_pos: 0x{after_tid * thread_stride:08X}")
                print(f"    予想のend_pos:   0x{(after_tid + 1) * thread_stride:08X}")
    
    # 欠落thread_idが処理すべきだったデータサイズを推定
    print("\n\n=== 欠落データサイズの推定 ===")
    
    # 平均的な行サイズを計算（周辺のthread_idから）
    sample_threads = list(range(1048570, 1048580)) + list(range(1398095, 1398110))
    
    row_sizes = []
    for tid in sample_threads:
        if tid in chunk0['_thread_id'].values:
            tid_rows = chunk0[chunk0['_thread_id'] == tid]
            if len(tid_rows) > 0:
                # このthread_idが処理した行数
                num_rows = len(tid_rows)
                # 1行あたりのサイズ（thread_stride / 行数）
                if num_rows > 0:
                    row_size = thread_stride / num_rows
                    row_sizes.append(row_size)
    
    if row_sizes:
        avg_row_size = np.mean(row_sizes)
        print(f"平均行サイズ: {avg_row_size:.1f}バイト")
        
        for tid in missing_threads:
            expected_rows = thread_stride / avg_row_size
            print(f"\nthread_id {tid}:")
            print(f"  予想処理行数: {expected_rows:.1f}行")
            print(f"  データサイズ: {thread_stride}バイト")

if __name__ == "__main__":
    calculate_positions()