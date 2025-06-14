"""
RMM バージョン互換対応版
PostgreSQL → GPU直接コピー （RMM 21.x ～ 25.x 対応）

RMM 25.x で copy_from_host の API が変更されたため、
実行時にバージョンを判定して適切な API を使用します。

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import inspect
import psycopg
import rmm
import numpy as np
from numba import cuda
import argparse

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_rmm_compatible.output.parquet"

def get_rmm_copy_method():
    """RMM バージョンに応じた適切なコピー方法を返す"""
    try:
        # copy_from_host のシグネチャを確認
        sig = inspect.signature(rmm.DeviceBuffer.copy_from_host)
        if 'dst_offset' in sig.parameters:
            print("✅ RMM 旧API (21.x系) 検出")
            return "old_api"
        else:
            print("✅ RMM 新API (25.x系) 検出 - Numba CUDA Driver使用")
            return "numba_driver"
    except Exception as e:
        print(f"⚠️  RMM API検出エラー: {e} - Numba CUDA Driver使用")
        return "numba_driver"

def copy_chunk_to_gpu_buffer(dbuf, chunk, offset, method):
    """
    チャンクをGPUバッファの指定オフセットにコピー
    
    Args:
        dbuf: rmm.DeviceBuffer
        chunk: bytes (ホストデータ)
        offset: int (GPU バッファ内のオフセット)
        method: str ("old_api" | "numba_driver")
    """
    chunk_size = len(chunk)
    
    if method == "old_api":
        # RMM 21.x 系の旧API
        dbuf.copy_from_host(buffer=chunk, dst_offset=offset)
        
    elif method == "numba_driver":
        # RMM 25.x 系対応 - Numba CUDA Driver使用
        cuda.cudadrv.driver.memcpy_htod(
            int(dbuf.ptr) + offset,  # dst GPU ptr + オフセット
            chunk,                   # src host bytes
            chunk_size               # サイズ
        )
    else:
        raise ValueError(f"未対応のコピー方法: {method}")

def run_rmm_compatible_benchmark(limit_rows=1000000):
    """RMM バージョン互換対応版ベンチマーク"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    # RMM バージョン情報表示
    print(f"=== RMM バージョン互換対応版ベンチマーク ===")
    try:
        print(f"RMM バージョン: {rmm.__version__}")
    except AttributeError:
        print("RMM バージョン: 不明")
    
    copy_method = get_rmm_copy_method()
    
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}")
    
    # RMM 初期化
    try:
        if not rmm.is_initialized():
            rmm.reinitialize(pool_allocator=True, initial_pool_size=8*1024**3)
            print("✅ RMM メモリプール初期化完了")
    except Exception as e:
        print(f"❌ RMM初期化エラー: {e}")
        return

    start_total_time = time.time()
    
    # メタデータ取得
    conn = psycopg.connect(dsn)
    try:
        print("メタデータを取得中...")
        start_meta_time = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {TABLE_NAME}")
        meta_time = time.time() - start_meta_time
        print(f"メタデータ取得完了 ({meta_time:.4f}秒)")
        
        ncols = len(columns)

        # データサイズ推定
        rows_est = limit_rows
        row_bytes = 17*8 + 17*4  # 概算
        header_est = 19
        total_size_est = header_est + rows_est * row_bytes + 1024
        
        print(f"推定データサイズ: {total_size_est / (1024*1024):.2f} MB")
        
        # GPU バッファ確保
        print("GPU バッファを確保中...")
        dbuf = rmm.DeviceBuffer(size=total_size_est)
        print(f"GPU バッファ確保完了: {total_size_est / (1024*1024):.2f} MB")

        # COPY BINARY → GPU直接書き込み
        print("COPY BINARY → GPU直接書き込み実行中...")
        print(f"使用コピー方法: {copy_method}")
        start_copy_time = time.time()
        
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        
        offset = 0
        total_chunks = 0
        
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy:
                for chunk in copy:
                    if chunk:
                        chunk_size = len(chunk)
                        if offset + chunk_size > dbuf.size:
                            print(f"⚠️  警告: バッファサイズ不足")
                            break
                        
                        # RMM バージョン対応のコピー
                        copy_chunk_to_gpu_buffer(dbuf, chunk, offset, copy_method)
                        offset += chunk_size
                        total_chunks += 1
                        
                        # 進捗表示
                        if total_chunks % 1000 == 0:
                            print(f"  📊 チャンク {total_chunks:,} | {offset / (1024*1024):.2f} MB")
        
        copy_time = time.time() - start_copy_time
        actual_data_size = offset
        
        print(f"✅ COPY BINARY → GPU直接書き込み完了 ({copy_time:.4f}秒)")
        print(f"  処理チャンク数: {total_chunks:,}")
        print(f"  実際のデータサイズ: {actual_data_size / (1024*1024):.2f} MB")
        print(f"  スループット: {actual_data_size / (1024*1024) / copy_time:.2f} MB/sec")

    finally:
        conn.close()

    # GPU バッファをトリミング
    if actual_data_size < dbuf.size:
        print("GPU バッファをトリミング中...")
        trimmed_dbuf = rmm.DeviceBuffer(size=actual_data_size)
        # RMM 25.x では copy_from_device も引数が変更されている可能性があるので注意
        try:
            trimmed_dbuf.copy_from_device(dbuf, size=actual_data_size)
        except TypeError:
            # フォールバック: Numba CUDA Driver使用
            cuda.cudadrv.driver.memcpy_dtod(
                int(trimmed_dbuf.ptr),
                int(dbuf.ptr),
                actual_data_size
            )
        dbuf = trimmed_dbuf
        print(f"GPU バッファトリミング完了: {actual_data_size / (1024*1024):.2f} MB")

    # numba GPU アレイに変換
    print("GPU バッファを numba GPU アレイに変換中...")
    raw_dev = cuda.as_cuda_array(dbuf).view(dtype=np.uint8)
    print(f"GPU アレイ変換完了: {raw_dev.shape[0]:,} bytes")

    # ヘッダーサイズ検出
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"ヘッダーサイズ: {header_size} バイト")

    # GPU最適化処理
    print("GPU最適化処理中...")
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
        
        print(f"GPU最適化処理完了 ({processing_time:.4f}秒), 行数: {rows}")
        
    except Exception as e:
        print(f"GPU最適化処理でエラー: {e}")
        return

    total_time = time.time() - start_total_time
    decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
    
    print(f"\n=== RMM互換版ベンチマーク完了 ===")
    print(f"総時間 = {total_time:.4f} 秒")
    print(f"使用API: {copy_method}")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得       : {meta_time:.4f} 秒")
    print(f"  COPY→GPU直接書き込み: {copy_time:.4f} 秒")
    print(f"  GPUパース           : {parse_time:.4f} 秒")
    print(f"  GPUデコード         : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み     : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数      : {rows:,} 行")
    print(f"  処理列数      : {len(columns)} 列")
    print(f"  Decimal列数   : {decimal_cols} 列")
    print(f"  データサイズ  : {actual_data_size / (1024*1024):.2f} MB")
    print(f"  処理チャンク数: {total_chunks:,}")
    
    total_cells = rows * len(columns)
    throughput = total_cells / decode_time if decode_time > 0 else 0
    network_throughput = actual_data_size / (1024*1024) / copy_time
    print(f"  セル処理速度  : {throughput:,.0f} cells/sec")
    print(f"  ネットワーク速度: {network_throughput:.2f} MB/sec")
    print("--- 最適化効果 ---")
    print("  ✅ ファイル I/O: 完全ゼロ (直接GPU書き込み)")
    print("  ✅ ホストメモリ: 最小化 (チャンクサイズのみ)")
    print("  ✅ GPU転送: リアルタイム (ストリーミング)")
    print(f"  ✅ RMM API: {copy_method} 自動選択")
    print("=====================================")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU直接コピー（RMM互換版）')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    
    args = parser.parse_args()
    
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    run_rmm_compatible_benchmark(limit_rows=args.rows)

if __name__ == "__main__":
    main()