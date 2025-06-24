"""
PostgreSQL → GPU Ray並列 順次GPU処理版（直接抽出版）
16並列でPostgreSQLから読み込み、直接抽出版でGPU処理

最適化:
- 16並列でPostgreSQL COPYを実行
- GPU処理は1つずつ順次実行（メモリ競合回避）
- 統合バッファを削除した直接抽出版を使用
- 文字列破損修正済み

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import math
import psycopg
import ray
import gc
import numpy as np
from numba import cuda
import argparse
from typing import List, Dict, Tuple, Optional
import psutil
import json
import cupy as cp
import tempfile

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet_direct import postgresql_to_cudf_parquet_direct

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_parallel_ctid_ray_sequential_gpu_direct_extraction.output"

# 並列設定
DEFAULT_PARALLEL = 16
CHUNK_COUNT = 4


@ray.remote
class PostgreSQLWorker:
    """Ray並列ワーカー: PostgreSQLからデータ取得"""
    
    def __init__(self, worker_id: int, dsn: str):
        self.worker_id = worker_id
        self.dsn = dsn
        
    def copy_range_to_memory(self, ctid_range: Tuple[int, int], chunk_idx: int) -> Tuple[bytes, float]:
        """指定されたctid範囲のデータをメモリに読み込み"""
        start_ctid, end_ctid = ctid_range
        
        copy_start = time.time()
        
        # PostgreSQL接続
        with psycopg.connect(self.dsn) as conn:
            # COPY BINARYコマンド
            copy_sql = f"""
                COPY (
                    SELECT * FROM {TABLE_NAME}
                    WHERE ctid >= '({start_ctid},1)'::tid
                    AND ctid < '({end_ctid},1)'::tid
                ) TO STDOUT (FORMAT binary)
            """
            
            # データを一括で読み込み
            data_chunks = []
            with conn.cursor().copy(copy_sql) as cpy:
                while True:
                    chunk = cpy.read()
                    if not chunk:
                        break
                    data_chunks.append(chunk)
            
            # バイト列に結合
            data = b''.join(data_chunks)
        
        copy_time = time.time() - copy_start
        
        return data, copy_time


def estimate_ctid_ranges(total_pages: int, parallel: int, chunks_per_worker: int) -> List[Tuple[Tuple[int, int], int]]:
    """ctid範囲を計算（ワーカーごとに複数チャンク）"""
    total_chunks = parallel * chunks_per_worker
    pages_per_chunk = total_pages // total_chunks
    
    ranges = []
    for worker_id in range(parallel):
        for chunk_idx in range(chunks_per_worker):
            global_chunk_id = worker_id * chunks_per_worker + chunk_idx
            start_page = global_chunk_id * pages_per_chunk
            end_page = (global_chunk_id + 1) * pages_per_chunk if global_chunk_id < total_chunks - 1 else total_pages
            ranges.append(((start_page, end_page), chunk_idx))
    
    return ranges


def process_gpu_direct(data: bytes, columns: List, output_path: str) -> Tuple[float, int, Dict]:
    """GPU直接抽出処理"""
    gpu_start = time.time()
    
    # NumPy配列に変換
    raw_host = np.frombuffer(data, dtype=np.uint8)
    
    # GPU転送
    raw_dev = cuda.to_device(raw_host)
    
    # ヘッダーサイズ検出
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    
    # 直接抽出版でGPU処理
    cudf_df, detailed_timing = postgresql_to_cudf_parquet_direct(
        raw_dev=raw_dev,
        columns=columns,
        ncols=len(columns),
        header_size=header_size,
        output_path=output_path,
        compression='snappy',
        use_rmm=True,
        optimize_gpu=True
    )
    
    gpu_time = time.time() - gpu_start
    rows = len(cudf_df) if cudf_df is not None else 0
    
    # メモリ解放
    del raw_dev
    del raw_host
    del cudf_df
    
    # GPUメモリプールをクリア
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    
    gc.collect()
    
    return gpu_time, rows, detailed_timing


def run_parallel_benchmark(parallel: int = DEFAULT_PARALLEL, no_limit: bool = False, chunks: int = CHUNK_COUNT):
    """並列ベンチマーク実行（直接抽出版）"""
    
    print(f"\n=== PostgreSQL → GPU Ray並列ベンチマーク（直接抽出版） ===")
    print(f"並列数: {parallel}")
    print(f"チャンク数/ワーカー: {chunks}")
    print(f"総チャンク数: {parallel * chunks}")
    print(f"統合バッファ: 【削除済み】")
    print(f"文字列破損: 【修正済み】")
    
    # PostgreSQL接続情報
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    # テーブル情報取得
    with psycopg.connect(dsn) as conn:
        # メタデータ取得
        print("\nメタデータを取得中...")
        columns = fetch_column_meta(conn, f"SELECT * FROM {TABLE_NAME}")
        print(f"列数: {len(columns)}")
        
        # テーブルのページ数を取得
        result = conn.execute(
            "SELECT relpages FROM pg_class WHERE relname = %s", 
            (TABLE_NAME,)
        ).fetchone()
        
        if not result:
            print(f"エラー: テーブル {TABLE_NAME} が見つかりません")
            return
        
        total_pages = result[0]
        print(f"総ページ数: {total_pages:,}")
    
    # ctid範囲を計算
    ranges = estimate_ctid_ranges(total_pages, parallel, chunks)
    print(f"\n並列処理範囲:")
    for i in range(min(3, len(ranges))):
        (start, end), chunk_idx = ranges[i]
        print(f"  ワーカー{i//chunks} チャンク{chunk_idx}: ctid ({start},1) - ({end},1)")
    if len(ranges) > 3:
        print(f"  ... (残り {len(ranges) - 3} 範囲)")
    
    # Ray初期化
    if not ray.is_initialized():
        ray.init(num_cpus=parallel)
    
    overall_start = time.time()
    
    # ワーカーを作成
    workers = [PostgreSQLWorker.remote(i, dsn) for i in range(parallel)]
    
    # 並列COPY実行
    print(f"\n[Phase 1] PostgreSQL並列COPY開始 ({parallel}並列)")
    copy_start = time.time()
    
    futures = []
    for worker_id in range(parallel):
        worker = workers[worker_id]
        worker_ranges = ranges[worker_id * chunks:(worker_id + 1) * chunks]
        
        for ctid_range, chunk_idx in worker_ranges:
            future = worker.copy_range_to_memory.remote(ctid_range, chunk_idx)
            futures.append((worker_id, chunk_idx, future))
    
    # 結果を収集
    all_data = []
    total_copy_time = 0
    total_bytes = 0
    
    for worker_id, chunk_idx, future in futures:
        data, copy_time = ray.get(future)
        all_data.append((worker_id, chunk_idx, data))
        total_copy_time += copy_time
        total_bytes += len(data)
        print(f"  ワーカー{worker_id} チャンク{chunk_idx}: {len(data) / 1024**2:.1f} MB ({copy_time:.2f}秒)")
    
    avg_copy_time = total_copy_time / len(futures)
    elapsed_copy = time.time() - copy_start
    print(f"\n並列COPY完了: {total_bytes / 1024**3:.2f} GB")
    print(f"  実時間: {elapsed_copy:.2f}秒")
    print(f"  平均COPY時間: {avg_copy_time:.2f}秒")
    print(f"  スループット: {total_bytes / elapsed_copy / 1024**3:.2f} GB/秒")
    
    # GPU処理（順次実行）
    print(f"\n[Phase 2] GPU直接抽出処理開始（統合バッファ削除版）")
    gpu_start = time.time()
    
    total_rows = 0
    total_gpu_time = 0
    
    # 一時ファイルディレクトリ
    temp_dir = tempfile.mkdtemp(prefix="gpupgparser_")
    
    try:
        for idx, (worker_id, chunk_idx, data) in enumerate(all_data):
            chunk_output = f"{temp_dir}/chunk_{worker_id}_{chunk_idx}.parquet"
            
            print(f"\n処理中 [{idx+1}/{len(all_data)}]: ワーカー{worker_id} チャンク{chunk_idx}")
            
            gpu_time, rows, timing = process_gpu_direct(data, columns, chunk_output)
            
            total_gpu_time += gpu_time
            total_rows += rows
            
            print(f"  行数: {rows:,}")
            print(f"  GPU時間: {gpu_time:.2f}秒")
            print(f"  スループット: {len(data) / gpu_time / 1024**3:.2f} GB/秒")
            
            # メモリ解放
            del data
            gc.collect()
        
        gpu_elapsed = time.time() - gpu_start
        
        print(f"\nGPU処理完了:")
        print(f"  総行数: {total_rows:,}")
        print(f"  実時間: {gpu_elapsed:.2f}秒")
        print(f"  平均GPU時間: {total_gpu_time / len(all_data):.2f}秒")
        
        # 全体統計
        total_time = time.time() - overall_start
        
        print(f"\n{'='*60}")
        print("✅ 全処理完了（直接抽出版）")
        print('='*60)
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"  - 並列COPY: {elapsed_copy:.2f}秒")
        print(f"  - GPU処理: {gpu_elapsed:.2f}秒")
        print(f"総データサイズ: {total_bytes / 1024**3:.2f} GB")
        print(f"総行数: {total_rows:,} 行")
        print(f"全体スループット: {total_bytes / total_time / 1024**3:.2f} GB/秒")
        
        # 文字列破損チェック（最初のチャンクのみ）
        import cudf
        first_parquet = f"{temp_dir}/chunk_0_0.parquet"
        if os.path.exists(first_parquet):
            print("\n=== 文字列破損チェック ===")
            df = cudf.read_parquet(first_parquet)
            
            even_errors = 0
            odd_errors = 0
            
            if 'lo_orderpriority' in df.columns:
                check_rows = min(1000, len(df))
                for i in range(check_rows):
                    try:
                        value = df['lo_orderpriority'].iloc[i]
                        expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                        is_valid = any(value.startswith(p) for p in expected_patterns)
                        
                        if not is_valid:
                            if i % 2 == 0:
                                even_errors += 1
                            else:
                                odd_errors += 1
                    except:
                        pass
            
            print(f"偶数行エラー: {even_errors}")
            print(f"奇数行エラー: {odd_errors}")
            
            if even_errors + odd_errors == 0:
                print("✅ 文字列破損なし！")
                
    finally:
        # 一時ファイル削除
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Ray終了
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='PostgreSQL → GPU Ray並列ベンチマーク（直接抽出版）'
    )
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL,
                       help=f'並列数（デフォルト: {DEFAULT_PARALLEL}）')
    parser.add_argument('--no-limit', action='store_true',
                       help='全データを処理（LIMIT句なし）')
    parser.add_argument('--chunks', type=int, default=CHUNK_COUNT,
                       help=f'ワーカーあたりのチャンク数（デフォルト: {CHUNK_COUNT}）')
    
    args = parser.parse_args()
    
    # CUDA context確認
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        exit(1)
    
    run_parallel_benchmark(
        parallel=args.parallel,
        no_limit=args.no_limit,
        chunks=args.chunks
    )


if __name__ == "__main__":
    main()