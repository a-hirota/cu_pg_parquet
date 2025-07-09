#!/usr/bin/env python3
"""実際のthread_stride（1150バイト）での問題を分析"""

import numpy as np

def analyze_actual_stride():
    """実際のパラメータでスレッド割り当てを分析"""
    print("=== 実際のthread_strideでの分析 ===\n")
    
    # 実際のパラメータ
    chunk_size = 8724152320  # チャンク0のサイズ（8320MB）
    header_size = 11
    thread_stride = 1150  # 実際の値
    threads_per_block = 256
    grid_x = 29634
    grid_y = 1
    total_threads = grid_x * grid_y * threads_per_block
    
    print(f"チャンクサイズ: {chunk_size:,} bytes ({chunk_size/1024/1024:.1f}MB)")
    print(f"グリッド: ({grid_x}, {grid_y})")
    print(f"総スレッド数: {total_threads:,}")
    print(f"スレッドストライド: {thread_stride:,} bytes")
    
    # 最後のスレッドの処理範囲
    last_thread_id = total_threads - 1
    last_start = header_size + last_thread_id * thread_stride
    last_end = header_size + (last_thread_id + 1) * thread_stride
    
    print(f"\n最後のスレッド (Thread {last_thread_id}):")
    print(f"  開始位置: {last_start:,} bytes ({last_start/1024/1024:.1f}MB)")
    print(f"  終了位置: {last_end:,} bytes ({last_end/1024/1024:.1f}MB)")
    print(f"  チャンクサイズとの差: {chunk_size - last_start:,} bytes")
    
    # 全データをカバーできているか確認
    coverage = (last_end - header_size) / chunk_size * 100
    print(f"\nカバー率: {coverage:.1f}%")
    
    # 256MB境界付近のスレッド
    print("\n=== 256MB境界付近のスレッド ===")
    mb_256 = 256 * 1024 * 1024
    
    for boundary_mb in [256, 512, 768, 1024]:
        boundary = boundary_mb * 1024 * 1024
        thread_at_boundary = (boundary - header_size) // thread_stride
        
        print(f"\n{boundary_mb}MB境界 (0x{boundary:08X}):")
        for tid in range(thread_at_boundary - 2, thread_at_boundary + 3):
            if tid >= 0 and tid < total_threads:
                start = header_size + tid * thread_stride
                print(f"  Thread {tid}: 0x{start:08X} ({start/1024/1024:.2f}MB)")

def find_missing_rows_location():
    """15行がどこで欠落しているか特定"""
    print("\n\n=== 15行欠落位置の特定 ===\n")
    
    # 観測データ
    expected_rows = 12030000
    detected_rows = 12029985
    missing_rows = expected_rows - detected_rows
    
    # チャンクごとの期待行数（均等分割と仮定）
    rows_per_chunk = expected_rows // 16
    
    print(f"期待行数: {expected_rows:,}")
    print(f"検出行数: {detected_rows:,}")
    print(f"欠落行数: {missing_rows}")
    print(f"チャンクあたり期待行数: {rows_per_chunk:,}")
    
    # 平均行サイズから欠落データサイズを推定
    avg_row_size = 1150
    missing_data_size = missing_rows * avg_row_size
    
    print(f"\n欠落データサイズ（推定）: {missing_data_size:,} bytes ({missing_data_size/1024:.1f}KB)")
    
    # スレッドストライドとの関係
    thread_stride = 1150
    missing_threads = missing_data_size // thread_stride
    
    print(f"欠落に相当するスレッド数: {missing_threads:.1f}")

def check_thread_1398101_existence():
    """Thread 1398101が実際に実行されるか確認"""
    print("\n\n=== Thread 1398101の存在確認 ===\n")
    
    # グリッド設定
    threads_per_block = 256
    grid_x = 29634
    grid_y = 1
    total_threads = grid_x * grid_y * threads_per_block
    
    print(f"総スレッド数: {total_threads:,}")
    
    # Thread 1398101は存在するか？
    if 1398101 < total_threads:
        print(f"Thread 1398101: 存在する ✓")
    else:
        print(f"Thread 1398101: 存在しない ✗")
    
    # Thread 3404032は存在するか？
    if 3404032 < total_threads:
        print(f"Thread 3404032: 存在する ✓")
    else:
        print(f"Thread 3404032: 存在しない ✗")
    
    print(f"\n実際の最大スレッドID: {total_threads - 1}")

def analyze_thread_detection_pattern():
    """どのスレッドが行を検出したかのパターンを分析"""
    print("\n\n=== スレッド検出パターン分析 ===\n")
    
    # 仮定：各スレッドは平均1行を検出
    # 12,029,985行を検出したスレッドがある
    
    detected_rows = 12029985
    total_threads = 7586304
    
    avg_rows_per_thread = detected_rows / total_threads
    print(f"平均行数/スレッド: {avg_rows_per_thread:.2f}")
    
    # もしThread 1398102-3404031が動作していない場合
    skip_start = 1398102
    skip_end = 3404031
    skipped_threads = skip_end - skip_start + 1
    
    working_threads = total_threads - skipped_threads
    adjusted_avg = detected_rows / working_threads
    
    print(f"\nスキップされたスレッド数: {skipped_threads:,}")
    print(f"動作したスレッド数: {working_threads:,}")
    print(f"調整後の平均行数/スレッド: {adjusted_avg:.2f}")

if __name__ == "__main__":
    analyze_actual_stride()
    find_missing_rows_location()
    check_thread_1398101_existence()
    analyze_thread_detection_pattern()