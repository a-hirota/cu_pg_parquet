"""
PostgreSQL → Rust → GPU キューベース並列処理版
Producer-Consumerパターンで真の並列処理を実現
"""

import os
import time
import subprocess
import json
import numpy as np
from numba import cuda
import rmm
import cudf
import cupy as cp
import kvikio
from pathlib import Path
from typing import List, Dict, Any, Optional
import psycopg
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet_direct
from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size
from src.readPostgres.metadata import fetch_column_meta
import pyarrow.parquet as pq

TABLE_NAME = "lineorder"  # デフォルト値（実行時に上書きされる）
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
MAX_QUEUE_SIZE = 3  # キューの最大サイズ

# グローバル変数
chunk_stats = []
gpu_row_counts = {}  # GPU処理行数を保存（チャンクIDをキーとする辞書）
shutdown_flag = threading.Event()


def signal_handler(sig, frame):
    """Ctrl+Cハンドラー"""
    print("\n\n⚠️  処理を中断しています...")
    shutdown_flag.set()
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


def setup_rmm_pool():
    """RMMメモリプールを適切に設定"""
    try:
        if rmm.is_initialized():
            return
        
        # GPUメモリの90%を使用可能に設定
        gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        pool_size = int(gpu_memory * 0.9)
        
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=pool_size,
            maximum_pool_size=pool_size
        )
    except Exception as e:
        print(f"⚠️ RMM初期化警告: {e}")


def get_postgresql_metadata():
    """PostgreSQLからテーブルメタデータを取得"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    conn = psycopg.connect(dsn)
    try:
        print(f"PostgreSQLメタデータを取得中 (テーブル: {TABLE_NAME})...")
        columns = fetch_column_meta(conn, f"SELECT * FROM {TABLE_NAME}")
        print(f"✅ メタデータ取得完了: {len(columns)} 列")
        
        
        return columns
    finally:
        conn.close()


def cleanup_files(total_chunks=8):
    """ファイルをクリーンアップ"""
    files = [
        f"{OUTPUT_DIR}/{TABLE_NAME}_meta_0.json",
        f"{OUTPUT_DIR}/{TABLE_NAME}_data_0.ready"
    ] + [f"{OUTPUT_DIR}/chunk_{i}.bin" for i in range(total_chunks)]
    
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def rust_producer(chunk_queue: queue.Queue, total_chunks: int, stats_queue: queue.Queue):
    """Rust転送を実行するProducerスレッド"""
    for chunk_id in range(total_chunks):
        if shutdown_flag.is_set():
            break
            
        try:
            print(f"\n[Producer] チャンク {chunk_id + 1}/{total_chunks} Rust転送開始...")
            
            env = os.environ.copy()
            env['CHUNK_ID'] = str(chunk_id)
            env['TOTAL_CHUNKS'] = str(total_chunks)
            
            rust_start = time.time()
            process = subprocess.run(
                [RUST_BINARY],
                capture_output=True,
                text=True,
                env=env
            )
            
            if process.returncode != 0:
                print(f"❌ Rustエラー: {process.stderr}")
                continue
            
            # JSON結果を抽出
            output = process.stdout
            json_start = output.find("===CHUNK_RESULT_JSON===")
            json_end = output.find("===END_CHUNK_RESULT_JSON===")
            
            if json_start != -1 and json_end != -1:
                json_str = output[json_start + len("===CHUNK_RESULT_JSON==="):json_end].strip()
                result = json.loads(json_str)
                rust_time = result['elapsed_seconds']
                file_size = result['total_bytes']
                chunk_file = result['chunk_file']
            else:
                rust_time = time.time() - rust_start
                chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
                file_size = os.path.getsize(chunk_file)
            
            chunk_info = {
                'chunk_id': chunk_id,
                'chunk_file': chunk_file,
                'file_size': file_size,
                'rust_time': rust_time
            }
            
            print(f"[Producer] チャンク {chunk_id + 1} 転送完了 ({rust_time:.1f}秒, {file_size / 1024**3:.1f}GB)")
            
            # キューに追加（ブロッキング）
            chunk_queue.put(chunk_info)
            stats_queue.put(('rust_time', rust_time))
            
        except Exception as e:
            print(f"[Producer] エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # 終了シグナル
    chunk_queue.put(None)
    print("[Producer] 全チャンク転送完了")


def gpu_consumer(chunk_queue: queue.Queue, columns: List[ColumnMeta], consumer_id: int, stats_queue: queue.Queue):
    """GPU処理を実行するConsumerスレッド"""
    while not shutdown_flag.is_set():
        try:
            # キューからチャンクを取得（ブロッキング）
            chunk_info = chunk_queue.get(timeout=1)
            
            if chunk_info is None:  # 終了シグナル
                break
                
            chunk_id = chunk_info['chunk_id']
            chunk_file = chunk_info['chunk_file']
            file_size = chunk_info['file_size']
            
            print(f"[Consumer-{consumer_id}] チャンク {chunk_id + 1} GPU処理開始...")
            
            # GPUメモリクリア
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            gpu_start = time.time()
            
            # kvikio+RMMで直接GPU転送
            transfer_start = time.time()
            
            # RMM DeviceBufferを使用
            gpu_buffer = rmm.DeviceBuffer(size=file_size)
            
            # kvikioで直接読み込み
            with kvikio.CuFile(chunk_file, "rb") as f:
                gpu_array = cp.asarray(gpu_buffer).view(dtype=cp.uint8)
                bytes_read = f.read(gpu_array)
            
            if bytes_read != file_size:
                raise RuntimeError(f"読み込みサイズ不一致: {bytes_read} != {file_size}")
            
            # numba cuda配列に変換（ゼロコピー）
            raw_dev = cuda.as_cuda_array(gpu_buffer).view(dtype=np.uint8)
            
            transfer_time = time.time() - transfer_start
            
            # ヘッダーサイズ検出
            header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
            header_size = detect_pg_header_size(header_sample)
            
            # 直接抽出処理
            chunk_output = f"output/chunk_{chunk_id}_queue.parquet"
            
            cudf_df, detailed_timing = postgresql_to_cudf_parquet_direct(
                raw_dev=raw_dev,
                columns=columns,
                ncols=len(columns),
                header_size=header_size,
                output_path=chunk_output,
                compression='snappy',
                use_rmm=True,
                optimize_gpu=True,
                verbose=False
            )
            
            gpu_time = time.time() - gpu_start
            
            # 処理統計
            rows = len(cudf_df) if cudf_df is not None else 0
            
            # GPU処理行数を保存
            gpu_row_counts[chunk_id] = rows
            
            print(f"[Consumer-{consumer_id}] チャンク {chunk_id + 1} GPU処理完了 ({gpu_time:.1f}秒, {rows:,}行)")
            
            # テストモードで追加情報を表示
            if os.environ.get('GPUPGPARSER_TEST_MODE', '0') == '1':
                print(f"[CHUNK DEBUG] チャンク {chunk_id + 1}: ")
                print(f"  - ファイルサイズ: {file_size / 1024**2:.1f} MB")
                print(f"  - 検出行数: {rows:,}行")
                print(f"  - GPUパース時間: {detailed_timing.get('gpu_parsing', 0):.2f}秒")
                print(f"  - 行あたり: {file_size/rows if rows > 0 else 0:.1f} bytes/row")
            
            # 統計情報を送信
            stats_queue.put(('gpu_time', gpu_time))
            stats_queue.put(('transfer_time', transfer_time))
            stats_queue.put(('rows', rows))
            stats_queue.put(('size', file_size))
            
            # 詳細統計を保存
            chunk_stats.append({
                'chunk_id': chunk_id,
                'consumer_id': consumer_id,
                'rust_time': chunk_info['rust_time'],
                'gpu_time': gpu_time,
                'transfer_time': transfer_time,
                'parse_time': detailed_timing.get('gpu_parsing', 0),
                'string_time': detailed_timing.get('string_buffer_creation', 0),
                'write_time': detailed_timing.get('parquet_export', 0),
                'rows': rows,
                'size_gb': file_size / 1024**3
            })
            
            # メモリ解放
            del raw_dev
            del gpu_buffer
            del gpu_array
            if cudf_df is not None:
                del cudf_df
            
            mempool.free_all_blocks()
            
            # ガベージコレクション
            import gc
            gc.collect()
            
            # チャンクファイル削除（中間ファイルのみ）
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            # Parquetファイルは出力ファイルなので保持
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Consumer-{consumer_id}] エラー: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[Consumer-{consumer_id}] 終了")


def validate_parquet_output(file_path: str, num_rows: int = 5, gpu_rows: int = None) -> bool:
    """
    Parquetファイルの検証とサンプル表示
    
    Args:
        file_path: 検証するParquetファイルのパス
        num_rows: 表示する行数
        gpu_rows: GPU処理で検出した行数（比較用）
    
    Returns:
        検証成功の場合True
    """
    try:
        # PyArrowでParquetファイルを読み込む
        table = pq.read_table(file_path)
        
        print(f"\n📊 Parquetファイル検証: {os.path.basename(file_path)}")
        print(f"├─ 行数: {table.num_rows:,}", end="")
        if gpu_rows is not None:
            if table.num_rows == gpu_rows:
                print(f" ✅ OK (GPU処理行数と一致)")
            else:
                print(f" ❌ NG (GPU処理行数: {gpu_rows:,})")
        else:
            print()
        print(f"├─ 列数: {table.num_columns}")
        print(f"└─ ファイルサイズ: {os.path.getsize(file_path) / 1024**2:.2f} MB")
        
        # サンプルデータを表示
        print(f"\n📝 サンプルデータ（先頭{num_rows}行）:")
        print("─" * 80)
        df_sample = table.slice(0, num_rows).to_pandas()
        print(df_sample.to_string(index=False, max_colwidth=20))
        print("─" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Parquet検証エラー: {e}")
        return False


def run_parallel_pipeline(columns: List[ColumnMeta], total_chunks: int):
    """真の並列パイプライン実行"""
    # キューとスレッド管理
    chunk_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    stats_queue = queue.Queue()
    
    start_time = time.time()
    
    # Producerスレッド開始
    producer_thread = threading.Thread(
        target=rust_producer,
        args=(chunk_queue, total_chunks, stats_queue)
    )
    producer_thread.start()
    
    # Consumerスレッド開始（1つのみ - GPUメモリ制約）
    consumer_thread = threading.Thread(
        target=gpu_consumer,
        args=(chunk_queue, columns, 1, stats_queue)
    )
    consumer_thread.start()
    
    # 統計収集
    total_rust_time = 0
    total_gpu_time = 0
    total_transfer_time = 0
    total_rows = 0
    total_size = 0
    
    # スレッドの終了を待機しながら統計を収集
    producer_thread.join()
    consumer_thread.join()
    
    # 統計キューから結果を収集
    while not stats_queue.empty():
        stat_type, value = stats_queue.get()
        if stat_type == 'rust_time':
            total_rust_time += value
        elif stat_type == 'gpu_time':
            total_gpu_time += value
        elif stat_type == 'transfer_time':
            total_transfer_time += value
        elif stat_type == 'rows':
            total_rows += value
        elif stat_type == 'size':
            total_size += value
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'total_rust_time': total_rust_time,
        'total_gpu_time': total_gpu_time,
        'total_transfer_time': total_transfer_time,
        'total_rows': total_rows,
        'total_size': total_size,
        'processed_chunks': len(chunk_stats)
    }


def main(total_chunks=8, table_name=None):
    global TABLE_NAME
    if table_name:
        TABLE_NAME = table_name
    # kvikio設定確認
    is_compat = os.environ.get("KVIKIO_COMPAT_MODE", "").lower() in ["on", "1", "true"]
    
    # RMMメモリプール設定
    setup_rmm_pool()
    
    # CUDA context確認
    try:
        cuda.current_context()
    except Exception as e:
        print(f"❌ CUDA context エラー: {e}")
        return
    
    # クリーンアップ
    cleanup_files(total_chunks)
    
    # PostgreSQLからメタデータを取得
    columns = get_postgresql_metadata()
    
    try:
        # 並列パイプライン実行
        print("\n並列処理を開始します...")
        print("=" * 80)
        
        results = run_parallel_pipeline(columns, total_chunks)
        
        # 最終統計を構造化表示
        total_gb = results['total_size'] / 1024**3
        
        print(f"\n{'='*80}")
        print(f" ✅ 処理完了サマリー")
        print(f"{'='*80}")
        
        # 全体統計
        print(f"\n【全体統計】")
        print(f"├─ 総データサイズ: {total_gb:.2f} GB")
        print(f"├─ 総行数: {results['total_rows']:,} 行")
        print(f"├─ 総実行時間: {results['total_time']:.2f}秒")
        print(f"└─ 全体スループット: {total_gb / results['total_time']:.2f} GB/秒")
        
        # 処理時間内訳
        print(f"\n【処理時間内訳】")
        print(f"├─ Rust転送合計: {results['total_rust_time']:.2f}秒 ({total_gb / results['total_rust_time']:.2f} GB/秒)")
        print(f"└─ GPU処理合計: {results['total_gpu_time']:.2f}秒 ({total_gb / results['total_gpu_time']:.2f} GB/秒)")
        print(f"   └─ kvikio転送: {results['total_transfer_time']:.2f}秒")
        
        
        # チャンク毎の詳細統計テーブル
        if chunk_stats:
            print(f"\n【チャンク毎の処理時間】")
            print(f"┌{'─'*7}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*12}┐")
            print(f"│チャンク│ Rust転送 │kvikio転送│ GPUパース│ 文字列処理│ Parquet │   処理行数  │")
            print(f"├{'─'*7}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*12}┤")
            
            # チャンクIDでソート
            sorted_stats = sorted(chunk_stats, key=lambda x: x['chunk_id'])
            for stat in sorted_stats:
                print(f"│  {stat['chunk_id']:^3}  │ {stat['rust_time']:>6.2f}秒 │ {stat['transfer_time']:>6.2f}秒 │"
                      f" {stat['parse_time']:>6.2f}秒 │ {stat['string_time']:>6.2f}秒 │"
                      f" {stat['write_time']:>6.2f}秒 │{stat['rows']:>10,}行│")
            
            print(f"└{'─'*7}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*12}┘")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 性能測定完了後、サンプル検証を実行
        sample_parquet = "output/chunk_0_queue.parquet"
        if os.path.exists(sample_parquet):
            # chunk_0のGPU処理行数を取得
            gpu_rows_chunk0 = gpu_row_counts.get(0, None)
            validate_parquet_output(sample_parquet, num_rows=5, gpu_rows=gpu_rows_chunk0)
            # Parquetファイルは出力ファイルなので保持
        
        cleanup_files(total_chunks)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU キューベース並列処理版ベンチマーク')
    parser.add_argument('--table', type=str, default='lineorder', help='対象テーブル名')
    parser.add_argument('--parallel', type=int, default=16, help='並列接続数')
    parser.add_argument('--chunks', type=int, default=8, help='チャンク数')
    args = parser.parse_args()
    
    # 環境変数を設定
    os.environ['RUST_PARALLEL_CONNECTIONS'] = str(args.parallel)
    os.environ['TOTAL_CHUNKS'] = str(args.chunks)
    os.environ['TABLE_NAME'] = args.table  # Rust側にもテーブル名を伝える
    
    main(total_chunks=args.chunks, table_name=args.table)