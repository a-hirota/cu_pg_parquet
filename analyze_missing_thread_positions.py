#!/usr/bin/env python3
"""欠落thread_idの計算上の位置と前後のthread_idの位置を分析"""

import pandas as pd

def analyze_positions():
    """欠落thread_idの位置を詳細分析"""
    print("=== 欠落thread_idの位置分析 ===\n")
    
    # 欠落thread_id
    missing_threads = [1048576, 1398102, 2097153, 2446679]
    
    # thread_stride（仮定値）
    thread_stride = 1150  # バイト
    
    # Parquetファイルを読み込み
    chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    
    print(f"thread_stride（仮定）: {thread_stride}バイト\n")
    print("=" * 100)
    
    for missing_tid in missing_threads:
        print(f"\n欠落thread_id: {missing_tid} (0x{missing_tid:06X})")
        print("-" * 80)
        
        # 欠落thread_idの計算上の位置
        calc_start = missing_tid * thread_stride
        calc_end = calc_start + thread_stride
        
        print(f"\n【計算上の位置】")
        print(f"  start_pos: 0x{calc_start:08X} ({calc_start:,}バイト, {calc_start/(1024*1024):7.2f}MB)")
        print(f"  end_pos:   0x{calc_end:08X} ({calc_end:,}バイト, {calc_end/(1024*1024):7.2f}MB)")
        print(f"  サイズ:    {thread_stride}バイト")
        
        # 前後のthread_idの実際の位置
        print(f"\n【周辺thread_idの実際の位置】")
        
        # 前後10個のthread_idを確認
        found_before = False
        found_after = False
        
        for offset in range(-10, 11):
            if offset == 0:
                continue
                
            tid = missing_tid + offset
            
            if tid in chunk0['_thread_id'].values:
                rows = chunk0[chunk0['_thread_id'] == tid]
                if len(rows) > 0:
                    row = rows.iloc[0]
                    actual_start = row['_thread_start_pos']
                    actual_end = row['_thread_end_pos']
                    actual_size = actual_end - actual_start
                    
                    # 計算上の位置
                    calc_tid_start = tid * thread_stride
                    calc_tid_end = calc_tid_start + thread_stride
                    
                    # 差分
                    start_diff = actual_start - calc_tid_start
                    
                    print(f"\n  Thread {tid} (offset={offset:+3d}):")
                    print(f"    実際:  start=0x{actual_start:08X} ({actual_start/(1024*1024):7.2f}MB), "
                          f"end=0x{actual_end:08X}, size={actual_size}バイト")
                    print(f"    計算:  start=0x{calc_tid_start:08X} ({calc_tid_start/(1024*1024):7.2f}MB), "
                          f"end=0x{calc_tid_end:08X}, size={thread_stride}バイト")
                    print(f"    差分:  {start_diff:+,}バイト ({start_diff/(1024*1024):+.2f}MB)")
                    
                    # 直前・直後を特別にマーク
                    if offset == -1:
                        found_before = True
                        print(f"    ⬆️ 直前のthread")
                    elif offset == 1:
                        found_after = True
                        print(f"    ⬇️ 直後のthread")
        
        # ギャップサイズの計算
        if found_before and found_after:
            # 直前と直後のthread_idの実際の位置を取得
            before_tid = missing_tid - 1
            after_tid = missing_tid + 1
            
            before_row = chunk0[chunk0['_thread_id'] == before_tid].iloc[0]
            after_row = chunk0[chunk0['_thread_id'] == after_tid].iloc[0]
            
            gap_start = before_row['_thread_end_pos']
            gap_end = after_row['_thread_start_pos']
            gap_size = gap_end - gap_start
            
            print(f"\n【ギャップ分析】")
            print(f"  前thread終了位置: 0x{gap_start:08X}")
            print(f"  後thread開始位置: 0x{gap_end:08X}")
            print(f"  ギャップサイズ:   {gap_size:,}バイト ({gap_size/(1024*1024):.2f}MB)")
            
            # 欠落thread_idの計算上の位置がギャップ内にあるか確認
            if gap_start <= calc_start < gap_end:
                print(f"  ✓ 計算上の開始位置はギャップ内")
            else:
                print(f"  ✗ 計算上の開始位置はギャップ外")
        
        # 境界との関係
        print(f"\n【境界との関係】")
        mb_64 = 64 * 1024 * 1024
        mb_256 = 256 * 1024 * 1024
        
        # 計算上の位置
        if calc_start % mb_256 == 0:
            print(f"  計算上のstart_pos: 256MB境界に一致 ({calc_start // mb_256} * 256MB)")
        elif calc_start % mb_64 == 0:
            print(f"  計算上のstart_pos: 64MB境界に一致 ({calc_start // mb_64} * 64MB)")
        
        # 実際の周辺位置を確認
        for offset in [-1, 1]:
            tid = missing_tid + offset
            if tid in chunk0['_thread_id'].values:
                row = chunk0[chunk0['_thread_id'] == tid].iloc[0]
                pos = row['_thread_start_pos']
                
                if pos % mb_256 == 0:
                    print(f"  Thread {tid}のstart_pos: 256MB境界に一致 ({pos // mb_256} * 256MB)")
                elif pos % mb_64 == 0:
                    print(f"  Thread {tid}のstart_pos: 64MB境界に一致 ({pos // mb_64} * 64MB)")

if __name__ == "__main__":
    analyze_positions()