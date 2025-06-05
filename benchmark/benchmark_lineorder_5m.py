"""
PostgreSQL → COPY BINARY → GPU Processing → Arrow RecordBatch → Parquet

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
PG_TABLE_PREFIX  : テーブルプレフィックス (optional)
USE_ZERO_COPY    : ZeroCopy機能を使用 (True/False, optional)
"""

import os
import time
import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import cudf
from numba import cuda
import argparse

from src.metadata import fetch_column_meta
from src.types import ColumnMeta
from src.build_buf_from_postgres import parse_binary_chunk_gpu, detect_pg_header_size
# 従来処理は削除済み（ZeroCopyのみ使用）
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_5m.output.parquet"

def run_benchmark(limit_rows=1000000):
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    # ZeroCopy統合処理のみをサポート
    print(f"ベンチマーク開始: テーブル={tbl}, 処理方式=ZeroCopy統合")
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

    # ZeroCopy統合処理のみを実行
    print("ZeroCopy統合処理中...")
    start_processing_time = time.time()
    
    try:
        cudf_df, detailed_timing = postgresql_to_cudf_parquet(
            raw_dev=raw_dev,
            columns=columns,
            ncols=ncols,
            header_size=header_size,
            output_path=OUTPUT_PARQUET_PATH,
            compression='snappy',
            use_rmm=True,
            optimize_gpu=True
        )
        
        processing_time = time.time() - start_processing_time
        rows = len(cudf_df)
        parse_time = detailed_timing.get('gpu_parsing', 0)
        decode_time = detailed_timing.get('cudf_creation', 0)
        write_time = detailed_timing.get('parquet_export', 0)
        
        print(f"ZeroCopy統合処理完了 ({processing_time:.4f}秒), 行数: {rows}")
        
    except Exception as e:
        print(f"ZeroCopy処理でエラー: {e}")
        print("処理を中断します。")
        return

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
        verification_df = cudf.read_parquet(OUTPUT_PARQUET_PATH)
        cudf_read_time = time.time() - start_cudf_read_time
        print(f"cuDF読み込み完了 ({cudf_read_time:.4f}秒)")
        
        print("--- cuDF DataFrame Info ---")
        verification_df.info()
        
        print(f"読み込み結果: {len(verification_df):,} 行 × {len(verification_df.columns)} 列")
        
        # データ型確認
        print("データ型:")
        for col_name, dtype in verification_df.dtypes.items():
            print(f"  {col_name}: {dtype}")
        
        print("\n--- cuDF DataFrame Head ---")
        # 全カラムを表示するための設定
        try:
            # cuDF 24.x以降の設定を試行
            with cudf.option_context('display.max_columns', None, 'display.width', None):
                print(verification_df.head())
        except Exception:
            try:
                # pandas互換の設定を試行
                import pandas as pd
                with pd.option_context('display.max_columns', None, 'display.width', None):
                    print(verification_df.head())
            except Exception:
                # フォールバック: 列を分割して表示
                n_cols = len(verification_df.columns)
                if n_cols > 10:
                    print("前半列:")
                    print(verification_df.iloc[:, :10].head())
                    print("後半列:")
                    print(verification_df.iloc[:, 10:].head())
                else:
                    print(verification_df.head())
        
        # 基本統計情報
        print("\n基本統計:")
        try:
            numeric_cols = verification_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:5]:  # 最初の5つの数値列のみ
                    col_data = verification_df[col]
                    if len(col_data) > 0:
                        print(f"  {col}: 平均={float(col_data.mean()):.2f}, 最小={float(col_data.min()):.2f}, 最大={float(col_data.max()):.2f}")
        except Exception as e:
            print(f"  統計情報エラー: {e}")
            
        print("-------------------------")
        print("cuDF検証: 成功")
            
    except Exception as e:
        print(f"cuDF検証: 失敗 - {e}")

def main():
    """メイン関数 - ZeroCopy版のみサポート"""
    parser = argparse.ArgumentParser(description='PostgreSQL → cuDF → Parquet ベンチマーク (ZeroCopy版)')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    
    args = parser.parse_args()
    
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        exit(1)
    
    run_benchmark(limit_rows=args.rows)

if __name__ == "__main__":
    main()
