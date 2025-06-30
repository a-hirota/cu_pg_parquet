#!/usr/bin/env python3
"""
GPU解析処理をデバッグ
"""

import os
import numpy as np
from numba import cuda
import rmm
import cupy as cp
import kvikio

from src.types import ColumnMeta
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet_direct
from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size
from src.readPostgres.metadata import fetch_column_meta
import psycopg

def main():
    # メタデータ取得
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
    conn.close()
    
    print(f"カラム数: {len(columns)}")
    
    # RMM初期化
    if not rmm.is_initialized():
        rmm.reinitialize(pool_allocator=True)
    
    # チャンクファイルを読み込み
    chunk_file = "/dev/shm/chunk_0.bin"
    file_size = os.path.getsize(chunk_file)
    print(f"ファイルサイズ: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
    
    # kvikioで読み込み
    gpu_buffer = rmm.DeviceBuffer(size=file_size)
    with kvikio.CuFile(chunk_file, "rb") as f:
        gpu_array = cp.asarray(gpu_buffer).view(dtype=cp.uint8)
        bytes_read = f.read(gpu_array)
    
    print(f"読み込みバイト数: {bytes_read:,}")
    
    # numba cuda配列に変換
    raw_dev = cuda.as_cuda_array(gpu_buffer).view(dtype=np.uint8)
    
    # ヘッダーサイズ検出
    header_sample = raw_dev[:128].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"ヘッダーサイズ: {header_size}")
    
    # GPU解析を実行（デバッグモード）
    print("\nGPU解析を実行中...")
    
    cudf_df, detailed_timing = postgresql_to_cudf_parquet_direct(
        raw_dev=raw_dev,
        columns=columns,
        ncols=len(columns),
        header_size=header_size,
        output_path=None,  # Parquetファイルは作成しない
        compression='snappy',
        use_rmm=True,
        optimize_gpu=True,
        verbose=True  # 詳細ログを有効化
    )
    
    if cudf_df is not None:
        print(f"\n解析結果:")
        print(f"行数: {len(cudf_df):,}")
        print(f"列数: {len(cudf_df.columns)}")
        
        # 最初の数行を表示
        print("\n最初の5行:")
        print(cudf_df.head())
    else:
        print("解析に失敗しました")
    
    # メモリクリーンアップ
    del raw_dev
    del gpu_buffer
    del gpu_array
    del cudf_df
    
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

if __name__ == "__main__":
    main()