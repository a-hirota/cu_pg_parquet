"""
Benchmark: PostgreSQL → COPY BINARY → GPU 2‑pass → Arrow RecordBatch → Parquet

環境変数
--------
GPUPASER_PG_DSN  : psycopg2.connect 互換 DSN 文字列
PG_TABLE_PREFIX  : lineorder テーブルがスキーマ名付きの場合に指定 (optional)

処理内容
----------
lineorder テーブル (約500万行を想定) を
1. COPY BINARY で取得
2. GPU パイプライン decode_chunk で Arrow RecordBatch に変換
3. 処理時間を計測
4. 結果を Parquet ファイルに出力
5. 出力された Parquet ファイルを cuDF で読み込み、基本情報を表示して検証
"""

import os
import time
import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import cudf
from numba import cuda

# Import necessary functions from the correct modules using absolute paths from root
from src.meta_fetch import fetch_column_meta, ColumnMeta # Import ColumnMeta as well
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v2 import decode_chunk
# Remove CPU row start calculator import
# from test.test_single_row_pg_parser import calculate_row_starts_cpu


TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_5m.output.parquet" # 出力ファイルパス

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
        # -------------------------------
        # ColumnMeta
        # -------------------------------
        print("メタデータを取得中...")
        start_meta_time = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {tbl}")
        meta_time = time.time() - start_meta_time
        print(f"メタデータ取得完了 ({meta_time:.4f}秒)")
        ncols = len(columns)

        # -------------------------------
        # COPY BINARY chunk
        # -------------------------------
        print("COPY BINARY を実行中 (LIMIT 1,000,000)...") # Add limit info
        start_copy_time = time.time()
        # 注意: 全データをメモリに読み込むため、巨大テーブルではメモリ不足の可能性あり
        limit_rows = 1000000
        copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)" # Add LIMIT clause
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

    # -------------------------------
    # GPU 処理
    # -------------------------------
    print("GPUにデータを転送中...")
    start_transfer_time = time.time()
    raw_dev = cuda.to_device(raw_host)
    transfer_time = time.time() - start_transfer_time
    print(f"GPU転送完了 ({transfer_time:.4f}秒)")

    # ヘッダーサイズを検出
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"検出したヘッダーサイズ: {header_size} バイト")

    # GPU Parse (includes row counting, offset calculation, field parsing)
    print("GPUでパース中 (行数カウント、オフセット計算、フィールド解析)...")
    start_parse_time = time.time()
    # Call the updated wrapper which now handles row counting and offset calculation internally
    # Enable GPU row detection for better performance
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev,
        ncols=ncols,
        header_size=header_size,
        use_gpu_row_detection=True  # Ensure GPU row detection is used
        # rows and row_start_positions are no longer passed
    )
    parse_time = time.time() - start_parse_time
    # The row count is now determined inside parse_binary_chunk_gpu
    rows = field_offsets_dev.shape[0] # Get actual row count from output array shape
    print(f"GPUパース完了 ({parse_time:.4f}秒), 行数: {rows}")

    print("GPUでデコード中 (Pass 1 & 2)...")
    start_decode_time = time.time()
    # decode_chunk は単一の RecordBatch を返す想定
    batch = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    decode_time = time.time() - start_decode_time
    print(f"GPUデコード完了 ({decode_time:.4f}秒)")

    # Arrow Table に変換 (複数バッチの場合は結合が必要だが、ここでは単一バッチと仮定)
    result_table = pa.Table.from_batches([batch])
    print(f"Arrow Table 作成完了: {result_table.num_rows} 行, {result_table.num_columns} 列")

    # -------------------------------
    # Parquet 出力
    # -------------------------------
    print(f"Parquetファイル書き込み中: {OUTPUT_PARQUET_PATH}")
    start_write_time = time.time()
    pq.write_table(result_table, OUTPUT_PARQUET_PATH)
    write_time = time.time() - start_write_time
    print(f"Parquet書き込み完了 ({write_time:.4f}秒)")

    total_time = time.time() - start_total_time
    print(f"\nベンチマーク完了: 総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得: {meta_time:.4f} 秒")
    print(f"  COPY BINARY   : {copy_time:.4f} 秒")
    print(f"  GPU転送       : {transfer_time:.4f} 秒")
    # print(f"  行数計算(CPU) : {row_calc_time:.4f} 秒") # Removed CPU calculation time
    print(f"  GPUパース(含 行数/オフセット計算): {parse_time:.4f} 秒")
    print(f"  GPUデコード   : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み: {write_time:.4f} 秒")
    print("----------------")


    # -------------------------------
    # cuDF での検証
    # -------------------------------
    print(f"\ncuDFでParquetファイルを読み込んで検証中: {OUTPUT_PARQUET_PATH}")
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
        print("cuDFでの読み込み検証: 成功")
    except Exception as e:
        print(f"cuDFでの読み込み検証: 失敗 - {e}")

if __name__ == "__main__":
    # CUDAコンテキストの初期化を確認
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        exit(1)
    run_benchmark()
