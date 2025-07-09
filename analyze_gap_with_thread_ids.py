#!/usr/bin/env python3
"""修正されたthread_idを使用してギャップを分析"""

import pandas as pd
import numpy as np
import subprocess
import os

def run_with_thread_ids():
    """thread_id記録を有効にしてベンチマークを実行"""
    print("=== thread_id記録付きベンチマーク実行 ===\n")
    
    # 既存のファイルをバックアップ
    if os.path.exists("output/customer_chunk_0_queue.parquet"):
        os.rename("output/customer_chunk_0_queue.parquet", 
                 "output/customer_chunk_0_queue_backup.parquet")
    
    # 環境変数設定
    env = os.environ.copy()
    env.update({
        'RUST_LOG': 'info',
        'RUST_PARALLEL_CONNECTIONS': '16',
        'GPUPGPARSER_TEST_MODE': '1',  # test_mode有効
        'GPUPGPARSER_DEBUG': '1',
        'TABLE_NAME': 'customer',
        'TOTAL_CHUNKS': '2',
        'CHUNK_ID': '0'
    })
    
    print("Rustプログラムを実行中...")
    cmd = ["/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk", "customer"]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("エラー:")
        print(result.stderr)
        return False
    
    print("実行完了")
    
    # GPUパーサーを実行
    print("\nGPUパーサーを実行中...")
    gpu_cmd = [
        "python", "/home/ubuntu/gpupgparser/docs/benchmark/benchmark_rust_gpu_direct.py",
        "--table", "customer",
        "--chunks", "1",
        "--parallel", "16"
    ]
    
    gpu_result = subprocess.run(gpu_cmd, env=env, capture_output=True, text=True)
    
    if gpu_result.returncode != 0:
        print("GPUパーサーエラー:")
        print(gpu_result.stderr)
    
    return True

def analyze_thread_gaps():
    """thread_idを使用してギャップを分析"""
    print("\n\n=== thread_idギャップ分析 ===\n")
    
    parquet_file = "output/customer_chunk_0_queue.parquet"
    if not os.path.exists(parquet_file):
        print(f"エラー: {parquet_file}が存在しません")
        return
    
    df = pd.read_parquet(parquet_file)
    
    print(f"総行数: {len(df):,}")
    print(f"列: {list(df.columns)}")
    
    if '_thread_id' not in df.columns:
        print("\n⚠️ _thread_id列が存在しません")
        return
    
    # thread_id統計
    print(f"\nthread_id統計:")
    print(f"  ユニーク値: {df['_thread_id'].nunique():,}")
    print(f"  最小値: {df['_thread_id'].min()}")
    print(f"  最大値: {df['_thread_id'].max()}")
    
    # thread_idでソート
    df_sorted = df.sort_values('_thread_id')
    
    # ユニークなthread_idを取得
    unique_threads = sorted(df['_thread_id'].unique())
    
    # ギャップを検出
    print("\n大きなthread_idギャップ:")
    gaps = []
    for i in range(1, len(unique_threads)):
        gap = unique_threads[i] - unique_threads[i-1]
        if gap > 1000:  # 1000以上のギャップ
            gaps.append((unique_threads[i-1], unique_threads[i], gap))
    
    if gaps:
        for prev, next, gap_size in gaps[:5]:
            print(f"\n  Thread {prev} → Thread {next} (ギャップ: {gap_size:,})")
            
            # これらのスレッドの詳細
            prev_data = df[df['_thread_id'] == prev]
            next_data = df[df['_thread_id'] == next]
            
            if len(prev_data) > 0:
                print(f"    Thread {prev}:")
                print(f"      行数: {len(prev_data)}")
                print(f"      開始位置: 0x{prev_data['_thread_start_pos'].iloc[0]:08X} ({prev_data['_thread_start_pos'].iloc[0]/(1024*1024):.1f}MB)")
                print(f"      終了位置: 0x{prev_data['_thread_end_pos'].iloc[0]:08X} ({prev_data['_thread_end_pos'].iloc[0]/(1024*1024):.1f}MB)")
                print(f"      最大c_custkey: {prev_data['c_custkey'].max()}")
            
            if len(next_data) > 0:
                print(f"    Thread {next}:")
                print(f"      行数: {len(next_data)}")
                print(f"      開始位置: 0x{next_data['_thread_start_pos'].iloc[0]:08X} ({next_data['_thread_start_pos'].iloc[0]/(1024*1024):.1f}MB)")
                print(f"      終了位置: 0x{next_data['_thread_end_pos'].iloc[0]:08X} ({next_data['_thread_end_pos'].iloc[0]/(1024*1024):.1f}MB)")
                print(f"      最小c_custkey: {next_data['c_custkey'].min()}")
    else:
        print("  大きなギャップは見つかりませんでした")
    
    # 欠落キー3483451の周辺
    print("\n\n=== 欠落キー3483451の周辺 ===")
    
    # c_custkeyでソートして確認
    df_key_sorted = df.sort_values('c_custkey')
    
    # 3483450と3483452を探す
    key_before = df_key_sorted[df_key_sorted['c_custkey'] == 3483450]
    key_after = df_key_sorted[df_key_sorted['c_custkey'] == 3483452]
    
    if len(key_before) > 0:
        print(f"\nキー3483450 (欠落直前):")
        row = key_before.iloc[0]
        print(f"  thread_id: {row['_thread_id']}")
        print(f"  位置: 0x{row['_row_position']:08X} ({row['_row_position']/(1024*1024):.1f}MB)")
        print(f"  スレッド範囲: 0x{row['_thread_start_pos']:08X}-0x{row['_thread_end_pos']:08X}")
    
    if len(key_after) > 0:
        print(f"\nキー3483452 (欠落直後):")
        row = key_after.iloc[0]
        print(f"  thread_id: {row['_thread_id']}")
        print(f"  位置: 0x{row['_row_position']:08X} ({row['_row_position']/(1024*1024):.1f}MB)")
        print(f"  スレッド範囲: 0x{row['_thread_start_pos']:08X}-0x{row['_thread_end_pos']:08X}")
    
    if len(key_before) > 0 and len(key_after) > 0:
        thread_before = key_before.iloc[0]['_thread_id']
        thread_after = key_after.iloc[0]['_thread_id']
        
        if thread_before != thread_after:
            print(f"\n⚠️ 異なるスレッドで処理されています:")
            print(f"  Thread {thread_before} → Thread {thread_after}")
            print(f"  スレッドギャップ: {thread_after - thread_before}")

if __name__ == "__main__":
    # 既存のParquetファイルを分析
    analyze_thread_gaps()