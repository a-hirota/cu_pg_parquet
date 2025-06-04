"""
PostgreSQL → COPY BINARY → GPU Processing → Arrow RecordBatch → Parquet

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
PG_TABLE_PREFIX  : テーブルプレフィックス (optional)
"""

import os
import time
import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import cudf
from numba import cuda

from src.meta_fetch import fetch_column_meta, ColumnMeta
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v7_column_wise_integrated import decode_chunk_v7_column_wise_integrated

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_5m.output.parquet"

def run_benchmark():
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    print(f"ベンチマーク開始: テーブル={tbl}")
    start_total_time = time.time()
    conn = psycopg.connect(dsn)
    try:
        print("メタデータを取得中...")
        start_meta_time = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {tbl}")
        meta_time = time.time() - start_meta_time
        print(f"メタデータ取得完了 ({meta_time:.4f}秒)")
        
        ncols = len(columns)

        print("COPY BINARY を実行中...")
        start_copy_time = time.time()
        limit_rows = 1000000
        copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
        copy_time = time.time() - start_copy_time
        print(f"COPY BINARY 完了 ({copy_time:.4f}秒), データサイズ: {len(raw_host) / (1024*1024):.2f} MB")

    finally:
        conn.close()

    print("GPUにデータを転送中...")
    start_transfer_time = time.time()
    raw_dev = cuda.to_device(raw_host)
    transfer_time = time.time() - start_transfer_time
    print(f"GPU転送完了 ({transfer_time:.4f}秒)")

    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"ヘッダーサイズ: {header_size} バイト")

    print("GPUでパース中...")
    start_parse_time = time.time()
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev,
        ncols=ncols,
        header_size=header_size
    )
    parse_time = time.time() - start_parse_time
    rows = field_offsets_dev.shape[0]
    print(f"GPUパース完了 ({parse_time:.4f}秒), 行数: {rows}")

    print("GPUでデコード中...")
    start_decode_time = time.time()
    batch = decode_chunk_v7_column_wise_integrated(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    decode_time = time.time() - start_decode_time
    print(f"GPUデコード完了 ({decode_time:.4f}秒)")

    result_table = pa.Table.from_batches([batch])
    print(f"Arrow Table作成完了: {result_table.num_rows} 行, {result_table.num_columns} 列")

    print(f"Parquetファイル書き込み中: {OUTPUT_PARQUET_PATH}")
    start_write_time = time.time()
    pq.write_table(result_table, OUTPUT_PARQUET_PATH)
    write_time = time.time() - start_write_time
    print(f"Parquet書き込み完了 ({write_time:.4f}秒)")

    total_time = time.time() - start_total_time
    decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
    
    print(f"\nベンチマーク完了: 総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得: {meta_time:.4f} 秒")
    print(f"  COPY BINARY   : {copy_time:.4f} 秒")
    print(f"  GPU転送       : {transfer_time:.4f} 秒")
    print(f"  GPUパース     : {parse_time:.4f} 秒")
    print(f"  GPUデコード   : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み: {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数      : {rows:,} 行")
    print(f"  処理列数      : {len(columns)} 列")
    print(f"  Decimal列数   : {decimal_cols} 列")
    print(f"  データサイズ  : {len(raw_host) / (1024*1024):.2f} MB")
    total_cells = rows * len(columns)
    throughput = total_cells / decode_time if decode_time > 0 else 0
    print(f"  スループット  : {throughput:,.0f} cells/sec")
    print("----------------")


    print(f"\ncuDFでParquetファイルを読み込み中: {OUTPUT_PARQUET_PATH}")
    try:
        start_cudf_read_time = time.time()
        gdf = cudf.read_parquet(OUTPUT_PARQUET_PATH)
        cudf_read_time = time.time() - start_cudf_read_time
        print(f"cuDF読み込み完了 ({cudf_read_time:.4f}秒)")
        print("--- cuDF DataFrame Info ---")
        gdf.info()
        print("\n--- cuDF DataFrame Head ---")
        print(gdf.head())
        print("-------------------------")
        print("cuDF検証: 成功")
            
    except Exception as e:
        print(f"cuDF検証: 失敗 - {e}")

if __name__ == "__main__":
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        exit(1)
    run_benchmark()
