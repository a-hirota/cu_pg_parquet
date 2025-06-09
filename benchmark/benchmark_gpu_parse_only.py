"""
GPUパース専用ベンチマーク - Ultra Fast Parser最適化確認
========================================================

元のmemory-coalescingブランチと同等のGPU割り当て効率を確認するためのベンチマーク
- GPU自動最適化機能の動作確認
- パース専用でのGPU使用効率測定
- デバッグ情報によるthreads/blocks設定の詳細確認
"""

import os
import time
import numpy as np
import psycopg
from numba import cuda
import argparse

from src.metadata import fetch_column_meta
from src.build_buf_from_postgres import parse_binary_chunk_gpu, detect_pg_header_size

# Ultra Fast版パーサーをインポート
try:
    from src.cuda_kernels.ultra_fast_parser import parse_binary_chunk_gpu_ultra_fast_v2
    ULTRA_FAST_AVAILABLE = True
except ImportError as e:
    ULTRA_FAST_AVAILABLE = False
    print(f"⚠️ Ultra Fast Parser not available: {e}")

TABLE_NAME = "lineorder"

def run_gpu_parse_benchmark(limit_rows=1000000, use_ultra_fast=True, debug=False):
    """GPUパース専用ベンチマーク"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    print(f"=== GPUパース専用ベンチマーク ===")
    print(f"テーブル: {tbl}")
    print(f"行数制限: {limit_rows:,}")
    print(f"Ultra Fast Parser: {'有効' if use_ultra_fast and ULTRA_FAST_AVAILABLE else '無効'}")
    print("-" * 40)

    # データ取得
    conn = psycopg.connect(dsn)
    try:
        print("1. メタデータ取得中...")
        start_meta = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {tbl}")
        meta_time = time.time() - start_meta
        print(f"   完了: {meta_time:.4f}秒")
        
        ncols = len(columns)

        print("2. COPY BINARY実行中...")
        start_copy = time.time()
        copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
        copy_time = time.time() - start_copy
        print(f"   完了: {copy_time:.4f}秒, サイズ: {len(raw_host) / (1024*1024):.2f} MB")

    finally:
        conn.close()

    print("3. GPU転送中...")
    start_transfer = time.time()
    raw_dev = cuda.to_device(raw_host)
    transfer_time = time.time() - start_transfer
    print(f"   完了: {transfer_time:.4f}秒")

    # ヘッダーサイズ検出
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"   ヘッダーサイズ: {header_size} バイト")

    # === 従来版パーサーテスト ===
    print("\n4. 従来版パーサーテスト...")
    standard_start = time.time()
    field_offsets_std, field_lengths_std = parse_binary_chunk_gpu(
        raw_dev, ncols, threads_per_block=256, header_size=header_size
    )
    standard_time = time.time() - standard_start
    rows_std = field_offsets_std.shape[0]
    print(f"   従来版完了: {standard_time:.4f}秒, 行数: {rows_std:,}")

    # === Ultra Fast版パーサーテスト ===
    if use_ultra_fast and ULTRA_FAST_AVAILABLE:
        print("\n5. Ultra Fast版パーサーテスト...")
        print(f"   GPU自動最適化機能でthreads/blocks設定を動的決定中...")
        
        ultra_start = time.time()
        field_offsets_ultra, field_lengths_ultra = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=header_size, debug=debug
        )
        ultra_time = time.time() - ultra_start
        rows_ultra = field_offsets_ultra.shape[0]
        print(f"   Ultra Fast完了: {ultra_time:.4f}秒, 行数: {rows_ultra:,}")
        
        # 性能比較
        speedup = standard_time / ultra_time if ultra_time > 0 else 0
        print(f"\n=== GPU最適化性能比較 ===")
        print(f"従来版:      {standard_time:.4f}秒")
        print(f"Ultra Fast:  {ultra_time:.4f}秒")
        print(f"高速化倍率:  {speedup:.2f}x")
        print(f"削減時間:    {standard_time - ultra_time:.4f}秒")
        
        # 結果一致性確認
        if rows_std == rows_ultra:
            print(f"✅ 行数一致: {rows_std:,}")
        else:
            print(f"❌ 行数不一致: 従来版={rows_std:,}, Ultra Fast={rows_ultra:,}")
            
        # GPU使用効率分析
        if debug:
            print(f"\n=== GPU使用効率分析 ===")
            # ultra_fast_parserからGPU特性を取得
            from src.cuda_kernels.ultra_fast_parser import get_device_properties
            props = get_device_properties()
            sm_count = props.get('MULTIPROCESSOR_COUNT', 'Unknown')
            max_threads = props.get('MAX_THREADS_PER_BLOCK', 'Unknown')
            
            print(f"GPU SMコア数: {sm_count}")
            print(f"最大threads/block: {max_threads}")
            print(f"データサイズ: {len(raw_host) / (1024*1024):.2f} MB")
            
            # パース専用でのスループット計算
            data_throughput = (len(raw_host) / (1024*1024)) / ultra_time
            print(f"パース専用スループット: {data_throughput:.2f} MB/秒")
            
            # 理論最大スループット（参考）
            theoretical_max = sm_count * 1024 if isinstance(sm_count, int) else 0
            if theoretical_max > 0:
                efficiency = (rows_ultra / ultra_time) / theoretical_max * 100
                print(f"理論効率（参考）: {efficiency:.2f}%")
            
    else:
        print("\n5. Ultra Fast版パーサー: 利用不可")

    # 総合結果
    total_time = meta_time + copy_time + transfer_time + (ultra_time if use_ultra_fast and ULTRA_FAST_AVAILABLE else standard_time)
    print(f"\n=== 総合結果（パース専用） ===")
    print(f"メタデータ取得: {meta_time:.4f}秒")
    print(f"COPY BINARY:   {copy_time:.4f}秒")
    print(f"GPU転送:       {transfer_time:.4f}秒")
    if use_ultra_fast and ULTRA_FAST_AVAILABLE:
        print(f"GPUパース:     {ultra_time:.4f}秒 (Ultra Fast)")
    else:
        print(f"GPUパース:     {standard_time:.4f}秒 (従来版)")
    print(f"総時間:        {total_time:.4f}秒")
    print(f"処理データ:    {len(raw_host) / (1024*1024):.2f} MB")
    if use_ultra_fast and ULTRA_FAST_AVAILABLE:
        print(f"パーススループット: {len(raw_host) / (1024*1024) / ultra_time:.2f} MB/秒")
    else:
        print(f"パーススループット: {len(raw_host) / (1024*1024) / standard_time:.2f} MB/秒")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='GPUパース専用ベンチマーク - Ultra Fast Parser最適化確認')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    parser.add_argument('--no-ultra-fast', action='store_true', help='Ultra Fast版を無効化')
    parser.add_argument('--debug', action='store_true', help='デバッグ情報を詳細表示')
    
    args = parser.parse_args()
    
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        exit(1)
    
    run_gpu_parse_benchmark(
        limit_rows=args.rows,
        use_ultra_fast=not args.no_ultra_fast,
        debug=args.debug
    )

if __name__ == "__main__":
    main()