"""
PostgreSQL → COPY BINARY → GPU Processing → Arrow RecordBatch → Parquet
個別バッファ版ベンチマーク

統合バッファを廃止し、個別バッファのみを使用する最適化版のベンチマーク
文字列処理で共有メモリを使用せず、直接グローバルメモリコピーによる高速化を実現

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
import argparse

from src.metadata import fetch_column_meta
from src.types import ColumnMeta
from src.build_buf_from_postgres import parse_binary_chunk_gpu, detect_pg_header_size
# 個別バッファ版を使用
from src.main_postgres_to_parquet_refactored import postgresql_to_cudf_parquet_individual

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_5m_individual.output.parquet"

def run_benchmark(limit_rows=1000000):
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    # 個別バッファ版ベンチマーク
    print(f"ベンチマーク開始: テーブル={tbl}, 処理方式=個別バッファ専用")
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

    # 個別バッファ版処理を実行
    print("個別バッファ処理中...")
    start_processing_time = time.time()
    
    try:
        cudf_df, detailed_timing = postgresql_to_cudf_parquet_individual(
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
        buffer_time = detailed_timing.get('individual_buffers', 0)
        decode_time = detailed_timing.get('cudf_creation', 0)
        write_time = detailed_timing.get('parquet_export', 0)
        
        print(f"個別バッファ処理完了 ({processing_time:.4f}秒), 行数: {rows}")
        
    except Exception as e:
        print(f"個別バッファ処理でエラー: {e}")
        print("処理を中断します。")
        return

    total_time = time.time() - start_total_time
    decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
    
    print(f"\nベンチマーク完了: 総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳（個別バッファ版） ---")
    print(f"  メタデータ取得     : {meta_time:.4f} 秒")
    print(f"  COPY BINARY       : {copy_time:.4f} 秒")
    print(f"  GPU転送           : {transfer_time:.4f} 秒")
    print(f"  GPUパース         : {parse_time:.4f} 秒")
    print(f"  個別バッファ作成   : {buffer_time:.4f} 秒")
    print(f"  cuDF作成          : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み   : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数          : {rows:,} 行")
    print(f"  処理列数          : {len(columns)} 列")
    print(f"  Decimal列数       : {decimal_cols} 列")
    print(f"  データサイズ      : {len(raw_host) / (1024*1024):.2f} MB")
    total_cells = rows * len(columns)
    
    # 個別バッファ処理のスループット計算
    if buffer_time > 0:
        buffer_throughput = total_cells / buffer_time
        print(f"  個別バッファスループット: {buffer_throughput:,.0f} cells/sec")
    
    if decode_time > 0:
        cudf_throughput = total_cells / decode_time
        print(f"  cuDF作成スループット    : {cudf_throughput:,.0f} cells/sec")
    
    overall_throughput = total_cells / processing_time if processing_time > 0 else 0
    print(f"  総合スループット        : {overall_throughput:,.0f} cells/sec")
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
        
        print("\n--- cuDF DataFrame Head (全列表示) ---")
        # pandas設定を使用して全列を強制表示
        import pandas as pd
        
        # DataFrameを文字列として表示し、全列を確実に表示
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        try:
            # cuDFをpandasに変換して全列表示
            pandas_df = verification_df.to_pandas()
            print(pandas_df.head())
        except Exception:
            # フォールバック: 列を一つずつ表示
            print("cuDF Head (列別表示):")
            for i, col_name in enumerate(verification_df.columns):
                print(f"  列{i+1:2d} {col_name:20s}: {verification_df[col_name].iloc[:3].to_pandas().tolist()}")
        
        # 設定をリセット
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
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
        print("cuDF検証: 成功（個別バッファ版）")
            
    except Exception as e:
        print(f"cuDF検証: 失敗 - {e}")

def main():
    """メイン関数 - 個別バッファ版ベンチマーク"""
    parser = argparse.ArgumentParser(description='PostgreSQL → cuDF → Parquet ベンチマーク (個別バッファ版)')
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