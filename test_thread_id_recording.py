#!/usr/bin/env python3
"""thread_id記録が修正されたか確認"""

import sys
import os
import cupy as cp
import numpy as np
sys.path.append('/home/ubuntu/gpupgparser')

from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2_lite

def test_thread_id_recording():
    """thread_idが正しく記録されるかテスト"""
    print("=== thread_id記録テスト ===\n")
    
    # PostgreSQL COPY BINARYヘッダ
    header = bytes([
        0x50, 0x47, 0x43, 0x4F, 0x50, 0x59,  # PGCOPY
        0x0A, 0xFF, 0x0D, 0x0A, 0x00,        # \n\377\r\n\0
        0x00, 0x00, 0x00, 0x00,              # flags
        0x00, 0x00, 0x00, 0x00               # header extension
    ])
    
    # 簡単なテスト行（3列）
    test_row = bytearray()
    test_row.extend(b'\x00\x03')  # 3列
    
    # 各フィールド
    for i in range(3):
        test_row.extend(b'\x00\x00\x00\x04')  # 長さ4
        test_row.extend(b'TEST')  # データ
    
    # 10MBのテストデータを作成
    data_size = 10 * 1024 * 1024
    data = bytearray(header)
    row_size = len(test_row)
    
    num_rows = (data_size - len(header)) // row_size
    print(f"テストデータ: {data_size:,} bytes ({data_size/1024/1024:.1f}MB)")
    print(f"行サイズ: {row_size} bytes")
    print(f"生成行数: {num_rows:,}")
    
    # データ生成
    for i in range(num_rows):
        data.extend(test_row)
    
    # GPUメモリに転送
    data_gpu = cp.asarray(data, dtype=cp.uint8)
    
    # カラム定義（必要な属性を追加）
    Column = type('Column', (), {
        'name': '',
        'data_type': '',
        'pg_oid': 0,
        'elem_size': 0,
        'is_nullable': True,
        'type_modifier': -1
    })
    
    columns = [
        type('Column', (), {
            'name': 'col1', 
            'data_type': 'int32', 
            'pg_oid': 23,
            'elem_size': 4,
            'is_nullable': True,
            'type_modifier': -1
        })(),
        type('Column', (), {
            'name': 'col2', 
            'data_type': 'int32', 
            'pg_oid': 23,
            'elem_size': 4,
            'is_nullable': True,
            'type_modifier': -1
        })(),
        type('Column', (), {
            'name': 'col3', 
            'data_type': 'int32', 
            'pg_oid': 23,
            'elem_size': 4,
            'is_nullable': True,
            'type_modifier': -1
        })()
    ]
    
    # パース実行（test_mode=True）
    print("\nパース実行中...")
    result = parse_binary_chunk_gpu_ultra_fast_v2_lite(
        data_gpu,
        columns=columns,
        debug=True,
        test_mode=True
    )
    
    print(f"\n結果:")
    
    # test_modeではタプルが返される
    if isinstance(result, tuple):
        row_positions, field_offsets, field_lengths, thread_ids, thread_start_pos, thread_end_pos = result
        print(f"  検出行数: {len(row_positions):,}")
        
        # thread_idsの分析
        if thread_ids is not None:
            # CPUに転送して分析
            thread_ids_cpu = cp.asnumpy(cp.asarray(thread_ids))
            
            print(f"\nthread_id統計:")
            print(f"  サイズ: {len(thread_ids_cpu)}")
            print(f"  ユニーク値: {len(np.unique(thread_ids_cpu))}")
            print(f"  最小値: {np.min(thread_ids_cpu)}")
            print(f"  最大値: {np.max(thread_ids_cpu)}")
            
            # 値の分布
            unique, counts = np.unique(thread_ids_cpu, return_counts=True)
            print(f"\n  thread_id分布（上位10）:")
            for tid, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    Thread {tid}: {count}行")
            
            # 0以外の値があるか
            non_zero = thread_ids_cpu[thread_ids_cpu != 0]
            if len(non_zero) > 0:
                print(f"\n  ✓ thread_id != 0 の行数: {len(non_zero)}")
            else:
                print(f"\n  ✗ すべてのthread_idが0です")
        else:
            print("\n✗ thread_idsがNoneです")
        
        # thread_start_positionsも確認
        if thread_start_pos is not None:
            print("\n✓ thread_start_positionsも返されています")
        if thread_end_pos is not None:
            print("✓ thread_end_positionsも返されています")
    else:
        print("\n✗ 結果がタプルではありません")

if __name__ == "__main__":
    test_thread_id_recording()