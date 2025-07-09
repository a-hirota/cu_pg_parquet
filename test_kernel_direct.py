#!/usr/bin/env python3
"""CUDAカーネルを直接呼び出して256MB境界問題を確認"""

import cupy as cp
import numpy as np
from numba import cuda, uint64, int32
import sys
sys.path.append('/home/ubuntu/gpupgparser')

# カーネルをインポート
from src.cuda_kernels.postgres_binary_parser import parse_rows_and_fields_lite

def test_256mb_boundary():
    """256MB境界でのカーネル動作を確認"""
    print("=== 256MB境界カーネルテスト ===\n")
    
    # パラメータ
    header_size = 11
    ncols = 17
    thread_stride = 1150
    max_rows = 2000000
    
    # 257MBのテストデータ（256MB境界を跨ぐ）
    data_size = 257 * 1024 * 1024
    data = cp.zeros(data_size, dtype=cp.uint8)
    
    # PostgreSQLバイナリヘッダを設定
    header = np.array([
        0x50, 0x47, 0x43, 0x4F, 0x50, 0x59,  # PGCOPY
        0x0A, 0xFF, 0x0D, 0x0A, 0x00          # \\n\\377\\r\\n\\0
    ], dtype=np.uint8)
    data[:11] = cp.asarray(header)
    
    # テスト行を作成（実際のPostgreSQL形式）
    row_data = []
    row_data.extend([0x00, 0x11])  # 17列
    for i in range(17):
        row_data.extend([0x00, 0x00, 0x00, 0x04])  # 長さ4
        row_data.extend([0x54, 0x45, 0x53, 0x54])  # "TEST"
    row_bytes = bytes(row_data)
    row_size = len(row_bytes)
    
    print(f"データサイズ: {data_size:,} bytes ({data_size/1024/1024:.1f}MB)")
    print(f"行サイズ: {row_size} bytes")
    
    # データに行を書き込む
    pos = header_size
    row_count = 0
    while pos + row_size < data_size:
        data[pos:pos+row_size] = cp.asarray(list(row_bytes), dtype=cp.uint8)
        pos += row_size
        row_count += 1
    
    print(f"書き込んだ行数: {row_count:,}")
    
    # 256MB境界の行を計算
    mb_256 = 256 * 1024 * 1024
    boundary_row = (mb_256 - header_size) // row_size
    print(f"256MB境界付近の行: {boundary_row}")
    
    # 出力配列の準備
    row_positions = cuda.device_array(max_rows, np.uint64)
    field_offsets = cuda.device_array((max_rows, ncols), np.uint32)
    field_lengths = cuda.device_array((max_rows, ncols), np.int32)
    detected_row_count = cuda.to_device(np.array([0], dtype=np.int32))
    fixed_field_lengths = cuda.to_device(np.array([4] * ncols, dtype=np.int32))
    
    # グリッド設定
    threads_per_block = 256
    estimated_rows = data_size // 1150  # 推定行サイズ
    target_threads = estimated_rows
    
    blocks_x = min((target_threads + threads_per_block - 1) // threads_per_block, 65535)
    blocks_y = (target_threads + blocks_x * threads_per_block - 1) // (blocks_x * threads_per_block)
    if blocks_y > 65535:
        blocks_y = 65535
    
    actual_threads = blocks_x * blocks_y * threads_per_block
    thread_stride = (data_size + actual_threads - 1) // actual_threads
    
    print(f"\nカーネル設定:")
    print(f"  グリッド: ({blocks_x}, {blocks_y})")
    print(f"  総スレッド数: {actual_threads:,}")
    print(f"  スレッドストライド: {thread_stride} bytes")
    
    # カーネル実行
    grid_2d = (blocks_x, blocks_y)
    parse_rows_and_fields_lite[grid_2d, threads_per_block](
        data, header_size, ncols,
        row_positions, field_offsets, field_lengths, detected_row_count,
        thread_stride, max_rows, fixed_field_lengths
    )
    cuda.synchronize()
    
    # 結果を取得
    detected_count = int(detected_row_count.copy_to_host()[0])
    print(f"\n検出された行数: {detected_count:,}")
    print(f"期待との差: {row_count - detected_count}")
    
    if detected_count > 0:
        # 行位置を確認
        positions = row_positions.copy_to_host()[:detected_count]
        
        # 256MB境界付近の行を探す
        print("\n256MB境界付近の検出行:")
        for i in range(detected_count):
            if abs(positions[i] - mb_256) < 10000:  # 10KB以内
                start_idx = max(0, i-3)
                end_idx = min(detected_count, i+4)
                for j in range(start_idx, end_idx):
                    marker = " ← 256MB境界付近" if abs(positions[j] - mb_256) < 1000 else ""
                    print(f"  行{j}: 0x{positions[j]:08X} ({positions[j]/1024/1024:.3f}MB){marker}")
                break
        
        # ギャップを検出
        print("\nギャップ検出:")
        gaps = []
        for i in range(1, detected_count):
            expected_pos = positions[i-1] + row_size
            actual_pos = positions[i]
            gap = actual_pos - expected_pos
            if gap > row_size * 2:  # 2行分以上のギャップ
                gaps.append((i-1, i, gap))
        
        if gaps:
            for prev_idx, next_idx, gap_size in gaps[:5]:
                print(f"\n  行{prev_idx}と行{next_idx}の間:")
                print(f"    前の行: 0x{positions[prev_idx]:08X} ({positions[prev_idx]/1024/1024:.3f}MB)")
                print(f"    次の行: 0x{positions[next_idx]:08X} ({positions[next_idx]/1024/1024:.3f}MB)")
                print(f"    ギャップ: {gap_size:,} bytes ({gap_size/row_size:.1f}行分)")
        else:
            print("  ギャップは検出されませんでした")

if __name__ == "__main__":
    test_256mb_boundary()