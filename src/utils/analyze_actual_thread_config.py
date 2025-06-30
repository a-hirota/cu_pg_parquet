#!/usr/bin/env python3
"""
実際のGPU設定とスレッドあたりの処理行数を計算
"""

import numpy as np
import sys
sys.path.append('/home/ubuntu/gpupgparser')

from src.cuda_kernels.postgres_binary_parser import (
    get_device_properties, calculate_optimal_grid_sm_aware, 
    estimate_row_size_from_columns
)
from src.types import ColumnMeta, INT32, INT64, DECIMAL128, DATE32, UTF8

def analyze_actual_configuration():
    """実際の設定に基づいて487行欠損を分析"""
    
    # lineorderテーブルのカラム定義（実際の定義）
    columns = [
        ColumnMeta(name='lo_orderkey', pg_oid=23, arrow_id=INT32, elem_size=4, pg_typmod=-1),
        ColumnMeta(name='lo_linenumber', pg_oid=20, arrow_id=INT64, elem_size=8, pg_typmod=-1),
        ColumnMeta(name='lo_custkey', pg_oid=23, arrow_id=INT32, elem_size=4, pg_typmod=-1),
        ColumnMeta(name='lo_partkey', pg_oid=23, arrow_id=INT32, elem_size=4, pg_typmod=-1),
        ColumnMeta(name='lo_suppkey', pg_oid=23, arrow_id=INT32, elem_size=4, pg_typmod=-1),
        ColumnMeta(name='lo_orderdate', pg_oid=1082, arrow_id=DATE32, elem_size=4, pg_typmod=-1),
        ColumnMeta(name='lo_orderpriority', pg_oid=1042, arrow_id=UTF8, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_shippriority', pg_oid=1042, arrow_id=UTF8, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_quantity', pg_oid=1700, arrow_id=DECIMAL128, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_extendedprice', pg_oid=1700, arrow_id=DECIMAL128, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_ordtotalprice', pg_oid=1700, arrow_id=DECIMAL128, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_discount', pg_oid=1700, arrow_id=DECIMAL128, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_revenue', pg_oid=1700, arrow_id=DECIMAL128, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_supplycost', pg_oid=1700, arrow_id=DECIMAL128, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_tax', pg_oid=1700, arrow_id=DECIMAL128, elem_size=-1, pg_typmod=-1),
        ColumnMeta(name='lo_commitdate', pg_oid=1082, arrow_id=DATE32, elem_size=4, pg_typmod=-1),
        ColumnMeta(name='lo_shipmode', pg_oid=1042, arrow_id=UTF8, elem_size=-1, pg_typmod=-1)
    ]
    
    # 実際のパラメータ
    total_rows = 246011837  # 総行数
    missing_rows = 487      # 欠損行数
    num_chunks = 32         # チャンク数
    workers_per_chunk = 16  # ワーカー数/チャンク
    
    # 各チャンクのサイズ（約8GB）
    chunk_size = 8 * 1024**3  # 8GB
    header_size = 19
    data_size_per_chunk = chunk_size - header_size
    
    print("=== 実際の設定 ===")
    print(f"総行数: {total_rows:,}")
    print(f"欠損行数: {missing_rows:,}")
    print(f"チャンク数: {num_chunks}")
    print(f"ワーカー数/チャンク: {workers_per_chunk}")
    print(f"チャンクサイズ: {chunk_size / 1024**3:.1f} GB")
    
    # 推定行サイズ
    estimated_row_size = estimate_row_size_from_columns(columns)
    print(f"\n推定行サイズ: {estimated_row_size} バイト")
    
    # GPUグリッド設定を計算
    blocks_x, blocks_y, threads_per_block = calculate_optimal_grid_sm_aware(
        data_size_per_chunk, estimated_row_size
    )
    
    total_threads = blocks_x * blocks_y * threads_per_block
    print(f"\nGPUグリッド設定:")
    print(f"  blocks: ({blocks_x}, {blocks_y})")
    print(f"  threads_per_block: {threads_per_block}")
    print(f"  総スレッド数: {total_threads:,}")
    
    # thread_strideの計算
    thread_stride = (data_size_per_chunk + total_threads - 1) // total_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size
    
    # MAX_ROWS_PER_THREADによる制限
    MAX_ROWS_PER_THREAD = 200
    max_thread_stride = estimated_row_size * MAX_ROWS_PER_THREAD
    if thread_stride > max_thread_stride:
        thread_stride_limited = max_thread_stride
        print(f"\n⚠️ thread_stride制限発動:")
        print(f"  元の値: {thread_stride:,} バイト")
        print(f"  制限後: {thread_stride_limited:,} バイト")
        thread_stride = thread_stride_limited
    else:
        print(f"\nthread_stride: {thread_stride:,} バイト（制限なし）")
    
    # スレッドあたりの処理行数
    rows_per_thread = thread_stride / estimated_row_size
    print(f"スレッドあたり処理行数: {rows_per_thread:.1f}")
    
    # ローカル配列サイズとの比較
    LOCAL_ARRAY_SIZE = 256
    print(f"\nローカル配列サイズ: {LOCAL_ARRAY_SIZE} 行")
    
    if rows_per_thread > LOCAL_ARRAY_SIZE:
        print(f"⚠️ 問題: スレッドあたりの行数（{rows_per_thread:.1f}）が配列サイズ（{LOCAL_ARRAY_SIZE}）を超過!")
        excess_rows = rows_per_thread - LOCAL_ARRAY_SIZE
        print(f"  超過行数/スレッド: {excess_rows:.1f}")
        
        # 全体での影響
        potential_missing = excess_rows * total_threads * num_chunks * workers_per_chunk
        print(f"  潜在的な欠損行数: {potential_missing:,.0f}")
    
    # カーネルコードの確認
    print("\n=== カーネルコードの問題箇所 ===")
    print("\nparse_rows_and_fields_lite（行327-434）:")
    print("1. ローカル配列が256行まで:")
    print("   local_positions = cuda.local.array(256, uint64)")
    print("\n2. 行処理ループ（行403-417）:")
    print("   if is_valid:")
    print("       if local_count < 256:  # ← 256行を超えるとスキップ")
    print("           local_positions[local_count] = uint64(candidate_pos)")
    print("           local_count += 1")
    print("\n問題: 256行を超える行は処理されずに無視される！")
    
    # 詳細な計算
    print("\n=== 詳細分析 ===")
    
    # チャンクあたりの行数
    rows_per_chunk = total_rows // num_chunks
    rows_per_worker = rows_per_chunk // workers_per_chunk
    
    print(f"\n行数の分布:")
    print(f"  行/チャンク: {rows_per_chunk:,}")
    print(f"  行/ワーカー: {rows_per_worker:,}")
    
    # ワーカーあたりのデータサイズ
    data_per_worker = rows_per_worker * estimated_row_size
    print(f"\nデータサイズ/ワーカー: {data_per_worker:,} バイト ({data_per_worker/1024**2:.1f} MB)")
    
    # ワーカーあたりのスレッド数
    threads_per_worker = data_per_worker / thread_stride
    print(f"スレッド数/ワーカー: {threads_per_worker:.1f}")
    
    # 境界スレッドの分析
    print("\n=== 境界スレッドの分析 ===")
    
    # 余りの行数
    remainder_rows = rows_per_worker % rows_per_thread
    print(f"余り行数/ワーカー: {remainder_rows:.1f}")
    
    # 487行の分布
    missing_per_chunk = missing_rows / num_chunks
    missing_per_worker = missing_per_chunk / workers_per_chunk
    
    print(f"\n欠損行の分布:")
    print(f"  欠損/チャンク: {missing_per_chunk:.2f}")
    print(f"  欠損/ワーカー: {missing_per_worker:.4f}")
    
    # 可能性のあるシナリオ
    print("\n=== 可能性のあるシナリオ ===")
    
    # シナリオ1: 特定のスレッドで256行制限
    if rows_per_thread > MAX_ROWS_PER_THREAD:
        threads_hitting_limit = missing_rows / (rows_per_thread - MAX_ROWS_PER_THREAD)
        print(f"\nシナリオ1: {MAX_ROWS_PER_THREAD}行制限に達したスレッド")
        print(f"  影響スレッド数: {threads_hitting_limit:.0f}")
    
    # シナリオ2: ローカル配列256行制限
    if rows_per_thread > LOCAL_ARRAY_SIZE:
        threads_hitting_array_limit = missing_rows / (rows_per_thread - LOCAL_ARRAY_SIZE)
        print(f"\nシナリオ2: ローカル配列256行制限に達したスレッド")
        print(f"  影響スレッド数: {threads_hitting_array_limit:.0f}")
    
    # シナリオ3: 境界処理の問題
    print(f"\nシナリオ3: チャンク境界での処理漏れ")
    print(f"  各チャンクの最後で約{missing_per_chunk:.1f}行が処理されない可能性")

if __name__ == "__main__":
    # デバイスプロパティを表示
    props = get_device_properties()
    print("=== GPU デバイス情報 ===")
    for key, value in props.items():
        print(f"{key}: {value}")
    print()
    
    analyze_actual_configuration()