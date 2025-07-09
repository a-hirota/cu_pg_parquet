#!/usr/bin/env python3
"""128MB-549MB範囲のデバッグ"""

import sys
import os
import cupy as cp
import numpy as np
import kvikio
sys.path.append('/home/ubuntu/gpupgparser')

from src.cuda_kernels.postgres_binary_parser_debug import parse_with_range_debug

def debug_range():
    """128MB-549MB範囲をデバッグ"""
    print("=== 128MB-549MB範囲デバッグ ===\n")
    
    # customerデータファイルを読み込み
    data_file = "/dev/shm/customer_chunk_0.bin"
    
    if not os.path.exists(data_file):
        print(f"エラー: {data_file}が存在しません")
        print("先にベンチマークを実行してデータファイルを生成してください")
        return
    
    file_size = os.path.getsize(data_file)
    print(f"データファイル: {data_file}")
    print(f"ファイルサイズ: {file_size:,} bytes ({file_size/(1024*1024*1024):.1f}GB)")
    
    # GPUメモリに直接読み込み
    print("\nGPUメモリに読み込み中...")
    data_gpu = cp.empty(file_size, dtype=cp.uint8)
    
    with kvikio.CuFile(data_file, "r") as f:
        f.read(data_gpu)
    
    print("読み込み完了")
    
    # カラム定義（customerテーブル）
    Column = type('Column', (), {
        'name': '',
        'data_type': '',
        'pg_oid': 0,
        'elem_size': 0,
        'is_nullable': True,
        'type_modifier': -1
    })
    
    columns = [
        type('Column', (), {'name': 'c_custkey', 'data_type': 'int32', 'pg_oid': 23, 'elem_size': 4, 'is_nullable': True, 'type_modifier': -1})(),
        type('Column', (), {'name': 'c_name', 'data_type': 'string', 'pg_oid': 1043, 'elem_size': -1, 'is_nullable': True, 'type_modifier': -1})(),
        type('Column', (), {'name': 'c_address', 'data_type': 'string', 'pg_oid': 1043, 'elem_size': -1, 'is_nullable': True, 'type_modifier': -1})(),
        type('Column', (), {'name': 'c_city', 'data_type': 'string', 'pg_oid': 1043, 'elem_size': -1, 'is_nullable': True, 'type_modifier': -1})(),
        type('Column', (), {'name': 'c_nation', 'data_type': 'string', 'pg_oid': 1043, 'elem_size': -1, 'is_nullable': True, 'type_modifier': -1})(),
        type('Column', (), {'name': 'c_region', 'data_type': 'string', 'pg_oid': 1043, 'elem_size': -1, 'is_nullable': True, 'type_modifier': -1})(),
        type('Column', (), {'name': 'c_phone', 'data_type': 'string', 'pg_oid': 1043, 'elem_size': -1, 'is_nullable': True, 'type_modifier': -1})(),
        type('Column', (), {'name': 'c_mktsegment', 'data_type': 'string', 'pg_oid': 1043, 'elem_size': -1, 'is_nullable': True, 'type_modifier': -1})(),
    ]
    
    # デバッグ実行
    result = parse_with_range_debug(data_gpu, columns, mb_start=128, mb_end=549)
    
    # thread_idsを分析
    if result['thread_ids'] is not None:
        thread_ids = cp.asnumpy(cp.asarray(result['thread_ids']))
        unique_threads = np.unique(thread_ids)
        
        print(f"\n全体のthread_id統計:")
        print(f"  ユニークスレッド数: {len(unique_threads):,}")
        print(f"  最小thread_id: {np.min(unique_threads)}")
        print(f"  最大thread_id: {np.max(unique_threads)}")
        
        # thread_idのギャップを探す
        thread_diffs = np.diff(unique_threads)
        large_gaps = np.where(thread_diffs > 1000)[0]
        
        if len(large_gaps) > 0:
            print(f"\n大きなthread_idギャップ（>1000）:")
            for gap_idx in large_gaps[:5]:
                prev_thread = unique_threads[gap_idx]
                next_thread = unique_threads[gap_idx + 1]
                gap_size = next_thread - prev_thread
                print(f"  Thread {prev_thread} → {next_thread} (ギャップ: {gap_size:,})")

if __name__ == "__main__":
    debug_range()