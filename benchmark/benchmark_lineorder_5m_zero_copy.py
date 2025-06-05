"""
PostgreSQL → COPY BINARY → GPU Processing → cuDF ZeroCopy → Direct GPU Parquet

cuDFゼロコピー処理とGPU直接Parquet書き出しのベンチマーク

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
PG_TABLE_PREFIX  : テーブルプレフィックス (optional)
"""

import os
import time
import numpy as np
import psycopg
import cudf
from numba import cuda

from src.metadata import fetch_column_meta
from src.types import ColumnMeta
from src.binary_parser import parse_binary_chunk_gpu, detect_pg_header_size
from src.cudf_zero_copy_processor import CuDFZeroCopyProcessor, decode_chunk_integrated_zero_copy

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_5m_zero_copy.output.parquet"

def run_zero_copy_benchmark():
    """cuDFゼロコピーベンチマーク実行"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    print(f"=== cuDF ZeroCopy ベンチマーク開始 ===")
    print(f"テーブル: {tbl}")
    print(f"出力: {OUTPUT_PARQUET_PATH}")
    start_total_time = time.time()
    
    # cuDFゼロコピープロセッサー初期化
    print("cuDF ZeroCopy プロセッサー初期化中...")
    processor_init_start = time.time()
    zero_copy_processor = CuDFZeroCopyProcessor(use_rmm=True)
    processor_init_time = time.time() - processor_init_start
    print(f"プロセッサー初期化完了 ({processor_init_time:.4f}秒)")
    
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

    print("=== cuDF ZeroCopy 統合処理開始 ===")
    start_zero_copy_time = time.time()
    
    try:
        # ゼロコピー統合デコード + GPU直接Parquet書き出し
        cudf_df = decode_chunk_integrated_zero_copy(
            raw_dev, 
            field_offsets_dev, 
            field_lengths_dev, 
            columns,
            OUTPUT_PARQUET_PATH,
            zero_copy_processor
        )
        
        zero_copy_time = time.time() - start_zero_copy_time
        print(f"=== cuDF ZeroCopy 統合処理完了 ({zero_copy_time:.4f}秒) ===")
        print(f"cuDF DataFrame作成完了: {len(cudf_df)} 行, {len(cudf_df.columns)} 列")
        print(f"GPU直接Parquet書き出し完了: {OUTPUT_PARQUET_PATH}")
        
    except Exception as e:
        print(f"cuDF ZeroCopy処理でエラー: {e}")
        # フォールバック処理
        print("従来方式でフォールバック中...")
        from src.column_processor import decode_chunk_integrated
        import pyarrow.parquet as pq
        
        batch = decode_chunk_integrated(raw_dev, field_offsets_dev, field_lengths_dev, columns)
        arrow_table = pa.Table.from_batches([batch])
        pq.write_table(arrow_table, OUTPUT_PARQUET_PATH.replace('.parquet', '_fallback.parquet'))
        
        zero_copy_time = time.time() - start_zero_copy_time
        print(f"フォールバック処理完了 ({zero_copy_time:.4f}秒)")

    total_time = time.time() - start_total_time
    decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
    
    print(f"\n=== ベンチマーク結果 ===")
    print(f"総実行時間: {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  プロセッサー初期化: {processor_init_time:.4f} 秒")
    print(f"  メタデータ取得    : {meta_time:.4f} 秒")
    print(f"  COPY BINARY      : {copy_time:.4f} 秒")
    print(f"  GPU転送          : {transfer_time:.4f} 秒")
    print(f"  GPUパース        : {parse_time:.4f} 秒")
    print(f"  ZeroCopy統合処理 : {zero_copy_time:.4f} 秒")
    print("--- パフォーマンス ---")
    print(f"  処理行数         : {rows:,} 行")
    print(f"  処理列数         : {len(columns)} 列")
    print(f"  Decimal列数      : {decimal_cols} 列")
    print(f"  データサイズ     : {len(raw_host) / (1024*1024):.2f} MB")
    
    total_cells = rows * len(columns)
    if zero_copy_time > 0:
        throughput = total_cells / zero_copy_time
        print(f"  スループット     : {throughput:,.0f} cells/sec")
    
    # データサイズあたりの処理速度
    data_mb = len(raw_host) / (1024*1024)
    if zero_copy_time > 0:
        mb_per_sec = data_mb / zero_copy_time
        print(f"  データ処理速度   : {mb_per_sec:.2f} MB/sec")
    
    print("=========================")

    # 結果検証
    print(f"\ncuDFでParquetファイルを読み込み検証中: {OUTPUT_PARQUET_PATH}")
    try:
        start_verify_time = time.time()
        verification_df = cudf.read_parquet(OUTPUT_PARQUET_PATH)
        verify_time = time.time() - start_verify_time
        
        print(f"検証用読み込み完了 ({verify_time:.4f}秒)")
        print("--- cuDF DataFrame 検証結果 ---")
        print(f"  行数: {len(verification_df):,}")
        print(f"  列数: {len(verification_df.columns)}")
        print("  データ型:")
        for col_name, dtype in verification_df.dtypes.items():
            print(f"    {col_name}: {dtype}")
        
        print("\n--- サンプルデータ (先頭5行) ---")
        try:
            # 全カラムを表示するための設定
            with cudf.option_context('display.max_columns', None, 'display.width', None):
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
        
        print("---------------------------")
        print("cuDF ZeroCopy 検証: 成功")
            
    except Exception as e:
        print(f"cuDF ZeroCopy 検証: 失敗 - {e}")


def run_comparison_benchmark():
    """従来方式との比較ベンチマーク"""
    print("\n" + "="*60)
    print("=== 従来方式 vs cuDF ZeroCopy 比較ベンチマーク ===")
    print("="*60)
    
    # 1. 従来方式
    print("\n[1/2] 従来方式実行中...")
    from benchmark.benchmark_lineorder_5m import run_benchmark as run_traditional
    traditional_start = time.time()
    run_traditional()
    traditional_time = time.time() - traditional_start
    
    # 2. ZeroCopy方式
    print("\n[2/2] cuDF ZeroCopy方式実行中...")
    zero_copy_start = time.time()
    run_zero_copy_benchmark()
    zero_copy_time = time.time() - zero_copy_start
    
    # 比較結果
    print("\n" + "="*60)
    print("=== 比較結果 ===")
    print(f"従来方式        : {traditional_time:.4f} 秒")
    print(f"ZeroCopy方式    : {zero_copy_time:.4f} 秒")
    if traditional_time > 0:
        speedup = traditional_time / zero_copy_time
        improvement = ((traditional_time - zero_copy_time) / traditional_time) * 100
        print(f"高速化倍率      : {speedup:.2f}x")
        print(f"性能向上        : {improvement:.1f}%")
    print("="*60)


if __name__ == "__main__":
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        exit(1)
    
    # 実行モード選択
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_comparison_benchmark()
    else:
        run_zero_copy_benchmark()