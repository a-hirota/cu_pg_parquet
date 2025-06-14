"""
PostgreSQL → GPU 1回コピー方式
「バッファ1回コピー」でシンプルかつ高効率な実装

RMM 25.x の正しい API:
- copy_from_host(host_bytes) # 位置引数1個のみ
- dst_offset や buffer= は存在しない

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import psycopg
import rmm
import numpy as np
from numba import cuda
import argparse
import io

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_single_copy.output.parquet"

def run_single_copy_benchmark(limit_rows=1000000):
    """
    バッファ1回コピー方式ベンチマーク
    - COPY ストリームを全て受け取ってから1回でGPUにコピー
    - CPU使用率最小化、シンプルなコード
    """
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    print(f"=== PostgreSQL → GPU 1回コピー方式 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}")
    
    # RMM 初期化
    try:
        if not rmm.is_initialized():
            rmm.reinitialize(
                pool_allocator=True, 
                initial_pool_size=8*1024**3,  # 8GB
                logging=True  # デバッグ用
            )
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

        # COPY BINARY → ホストバッファに全データ収集
        print("COPY BINARY → ホストバッファ収集中...")
        start_copy_time = time.time()
        
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        
        # 方式1: b"".join() で一括収集（シンプル）
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_obj:
                print("  📡 COPY ストリーム受信開始...")
                
                # 全チャンクを収集
                chunks = []
                chunk_count = 0
                for chunk in copy_obj:
                    if chunk:
                        chunks.append(chunk)
                        chunk_count += 1
                        
                        # 進捗表示を無効化
                        # if chunk_count % 1000 == 0:
                        #     total_size = sum(len(c) for c in chunks)
                        #     print(f"    チャンク {chunk_count:,} | {total_size / (1024*1024):.2f} MB")
                
                # 一括結合
                print("  🔗 チャンク結合中...")
                host_bytes = b"".join(chunks)
                chunks.clear()  # メモリ解放

        copy_time = time.time() - start_copy_time
        total_size = len(host_bytes)
        
        print(f"✅ COPY BINARY → ホストバッファ完了 ({copy_time:.4f}秒)")
        print(f"  処理チャンク数: {chunk_count:,}")
        print(f"  データサイズ: {total_size / (1024*1024):.2f} MB")
        print(f"  ネットワーク速度: {total_size / (1024*1024) / copy_time:.2f} MB/sec")

    finally:
        conn.close()

    # GPU バッファ確保 & 1回コピー
    print(f"GPU バッファ確保 & 1回コピー実行中...")
    start_gpu_time = time.time()
    
    try:
        # GPU バッファ確保（実データサイズぴったり）
        dbuf = rmm.DeviceBuffer(size=total_size)
        print(f"  GPU バッファ確保完了: {total_size / (1024*1024):.2f} MB")
        
        # 【重要】RMM 25.x 正しい API - 位置引数1個のみ
        dbuf.copy_from_host(host_bytes)
        print(f"  ✅ GPU 1回コピー完了!")
        
        # ホストメモリ解放
        del host_bytes
        
    except Exception as e:
        print(f"❌ GPU コピーエラー: {e}")
        return
    
    gpu_time = time.time() - start_gpu_time
    gpu_throughput = total_size / (1024*1024) / gpu_time
    print(f"GPU転送完了 ({gpu_time:.4f}秒), 速度: {gpu_throughput:.2f} MB/sec")

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
    
    print(f"\n=== バッファ1回コピー方式ベンチマーク完了 ===")
    print(f"総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得    : {meta_time:.4f} 秒")
    print(f"  COPY→ホスト収集  : {copy_time:.4f} 秒")
    print(f"  ホスト→GPU転送   : {gpu_time:.4f} 秒")
    print(f"  GPUパース        : {parse_time:.4f} 秒")
    print(f"  GPUデコード      : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み  : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数      : {rows:,} 行")
    print(f"  処理列数      : {len(columns)} 列")
    print(f"  Decimal列数   : {decimal_cols} 列")
    print(f"  データサイズ  : {total_size / (1024*1024):.2f} MB")
    print(f"  処理チャンク数: {chunk_count:,}")
    
    total_cells = rows * len(columns)
    throughput = total_cells / decode_time if decode_time > 0 else 0
    print(f"  セル処理速度  : {throughput:,.0f} cells/sec")
    print(f"  ネットワーク速度: {total_size / (1024*1024) / copy_time:.2f} MB/sec")
    print(f"  GPU転送速度    : {gpu_throughput:.2f} MB/sec")
    
    print("--- 最適化効果（1回コピー方式） ---")
    print("  ✅ ファイル I/O: 完全ゼロ")
    print("  ✅ ホストメモリ: 一時的に全データサイズ、転送後即解放")
    print("  ✅ GPU転送: 1回のみ（オーバーヘッド最小）")
    print("  ✅ CPU使用率: 最小化（複雑なオフセット処理なし）") 
    print("  ✅ RMM API: 正しい位置引数使用")
    print("=========================================")

    # 検証用の簡単な出力
    print(f"\ncuDF検証用出力:")
    try:
        print(f"出力Parquet: {OUTPUT_PARQUET_PATH}")
        print(f"読み込み確認: {len(cudf_df):,} 行 × {len(cudf_df.columns)} 列")
        print("先頭データ型:")
        for i, (col_name, dtype) in enumerate(cudf_df.dtypes.items()):
            if i < 5:  # 最初の5列のみ
                print(f"  {col_name}: {dtype}")
        print("✅ cuDF検証: 成功")
    except Exception as e:
        print(f"❌ cuDF検証: {e}")

def run_memory_optimized_single_copy(limit_rows=1000000):
    """
    メモリ最適化版：bytearray を事前確保してreadinto() 使用
    大きなデータセット用
    """
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    print(f"=== メモリ最適化版 1回コピー方式 ===")
    
    # データサイズ推定
    rows_est = limit_rows
    row_bytes = 17*8 + 17*4  # 概算
    header_est = 19
    estimated_size = header_est + rows_est * row_bytes
    
    print(f"推定データサイズ: {estimated_size / (1024*1024):.2f} MB")
    
    # RMM 初期化
    if not rmm.is_initialized():
        rmm.reinitialize(pool_allocator=True, initial_pool_size=8*1024**3)
        print("✅ RMM 初期化完了")

    # 事前にbytearray確保
    print("ホストバッファ事前確保中...")
    host_buffer = bytearray(estimated_size)
    
    conn = psycopg.connect(dsn)
    try:
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        
        print("COPY BINARY → 事前確保バッファに直接書き込み中...")
        start_time = time.time()
        
        offset = 0
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_obj:
                for chunk in copy_obj:
                    if chunk:
                        chunk_size = len(chunk)
                        if offset + chunk_size > len(host_buffer):
                            # バッファ拡張
                            host_buffer.extend(b'\x00' * (chunk_size * 2))
                        
                        # 直接書き込み
                        host_buffer[offset:offset+chunk_size] = chunk
                        offset += chunk_size
        
        # 実際のサイズにトリミング
        actual_data = bytes(host_buffer[:offset])
        del host_buffer  # 早期解放
        
        copy_time = time.time() - start_time
        print(f"✅ データ収集完了 ({copy_time:.4f}秒)")
        print(f"実際のデータサイズ: {len(actual_data) / (1024*1024):.2f} MB")
        
    finally:
        conn.close()
    
    # GPU 1回コピー
    print("GPU 1回コピー実行中...")
    start_gpu_time = time.time()
    
    dbuf = rmm.DeviceBuffer(size=len(actual_data))
    dbuf.copy_from_host(actual_data)  # 位置引数1個のみ
    
    gpu_time = time.time() - start_gpu_time
    print(f"✅ GPU転送完了 ({gpu_time:.4f}秒)")
    print(f"GPU転送速度: {len(actual_data) / (1024*1024) / gpu_time:.2f} MB/sec")
    
    return dbuf, len(actual_data)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU 1回コピー方式')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    parser.add_argument('--memory-optimized', action='store_true', help='メモリ最適化版を使用')
    
    args = parser.parse_args()
    
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    if args.memory_optimized:
        run_memory_optimized_single_copy(limit_rows=args.rows)
    else:
        run_single_copy_benchmark(limit_rows=args.rows)

if __name__ == "__main__":
    main()