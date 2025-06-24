"""
PostgreSQL → Rust → GPU リングバッファ版
並列処理によりRust転送とGPU処理を同時実行

改善内容:
1. リングバッファ方式で/dev/shmを利用
2. Rust転送とGPU処理の並列実行
3. プロデューサー・コンシューマーパターン
4. CPU待機時間の削減
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
from src.main_postgres_to_parquet_direct import postgresql_to_cudf_parquet_direct
from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size
from src.metadata import fetch_column_meta

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
RING_BUFFER_SIZE = 2  # 同時に保持する最大チャンク数（メモリ制約）

# グローバル変数
chunk_stats = []
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
            print("RMM既に初期化済み")
            return
        
        # GPUメモリの90%を使用可能に設定
        gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        pool_size = int(gpu_memory * 0.9)
        
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=pool_size,
            maximum_pool_size=pool_size
        )
        print(f"✅ RMMメモリプール初期化: {pool_size / 1024**3:.1f} GB")
    except Exception as e:
        print(f"⚠️ RMM初期化警告: {e}")


def get_postgresql_metadata():
    """PostgreSQLからテーブルメタデータを取得"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    conn = psycopg.connect(dsn)
    try:
        print("PostgreSQLメタデータを取得中...")
        columns = fetch_column_meta(conn, f"SELECT * FROM {TABLE_NAME}")
        print(f"✅ メタデータ取得完了: {len(columns)} 列")
        
        # デバッグ：Decimal列の情報を表示
        for col in columns:
            if col.arrow_id == 5:  # DECIMAL128
                print(f"  Decimal列 {col.name}: arrow_param={col.arrow_param}")
        
        return columns
    finally:
        conn.close()


def cleanup_files(total_chunks=8):
    """ファイルをクリーンアップ"""
    files = [
        f"{OUTPUT_DIR}/lineorder_meta_0.json",
        f"{OUTPUT_DIR}/lineorder_data_0.ready"
    ] + [f"{OUTPUT_DIR}/chunk_{i}.bin" for i in range(total_chunks)]
    
    for f in files:
        if os.path.exists(f):
            os.remove(f)


class RustProducer:
    """Rust転送を管理するプロデューサー"""
    
    def __init__(self, total_chunks: int, ring_buffer_size: int):
        self.total_chunks = total_chunks
        self.ring_buffer_size = ring_buffer_size
        self.completed_chunks = queue.Queue()
        self.active_chunks = set()
        self.lock = threading.Lock()
        
    def can_produce(self) -> bool:
        """新しいチャンクを生成可能か確認"""
        with self.lock:
            return len(self.active_chunks) < self.ring_buffer_size
    
    def produce_chunk(self, chunk_id: int) -> Optional[dict]:
        """1つのチャンクをRust転送"""
        if shutdown_flag.is_set():
            return None
            
        with self.lock:
            self.active_chunks.add(chunk_id)
        
        try:
            print(f"\n[Producer] チャンク {chunk_id + 1}/{self.total_chunks} Rust転送開始...", flush=True)
            
            env = os.environ.copy()
            env['CHUNK_ID'] = str(chunk_id)
            env['TOTAL_CHUNKS'] = str(self.total_chunks)
            
            rust_start = time.time()
            process = subprocess.run(
                [RUST_BINARY],
                capture_output=True,
                text=True,
                env=env
            )
            
            if process.returncode != 0:
                print(f"❌ Rustエラー: {process.stderr}")
                raise RuntimeError(f"チャンク{chunk_id}の転送失敗")
            
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
                columns_data = result.get('columns')
            else:
                rust_time = time.time() - rust_start
                chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
                file_size = os.path.getsize(chunk_file)
                columns_data = None
            
            chunk_info = {
                'chunk_id': chunk_id,
                'chunk_file': chunk_file,
                'file_size': file_size,
                'rust_time': rust_time,
                'columns': columns_data
            }
            
            print(f"[Producer] チャンク {chunk_id + 1} 転送完了 ({rust_time:.1f}秒, {file_size / 1024**3:.1f}GB)")
            
            # 完了キューに追加
            self.completed_chunks.put(chunk_info)
            return chunk_info
            
        except Exception as e:
            print(f"[Producer] エラー: {e}")
            with self.lock:
                self.active_chunks.discard(chunk_id)
            return None
    
    def mark_consumed(self, chunk_id: int):
        """チャンクが消費されたことをマーク"""
        with self.lock:
            self.active_chunks.discard(chunk_id)


class GPUConsumer:
    """GPU処理を管理するコンシューマー"""
    
    def __init__(self, columns: List[ColumnMeta], producer: RustProducer):
        self.columns = columns
        self.producer = producer
        self.processed_count = 0
        self.total_gpu_time = 0
        self.total_transfer_time = 0
        self.total_rows = 0
        self.total_size = 0
        
    def consume_chunk(self, chunk_info: dict) -> Optional[tuple]:
        """1つのチャンクをGPU処理"""
        if shutdown_flag.is_set():
            return None
            
        chunk_id = chunk_info['chunk_id']
        chunk_file = chunk_info['chunk_file']
        file_size = chunk_info['file_size']
        
        print(f"[Consumer] チャンク {chunk_id + 1} GPU処理開始...", flush=True)
        
        # GPUメモリクリア
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        gpu_start = time.time()
        
        try:
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
            process_start = time.time()
            chunk_output = f"benchmark/chunk_{chunk_id}_ringbuffer.parquet"
            
            cudf_df, detailed_timing = postgresql_to_cudf_parquet_direct(
                raw_dev=raw_dev,
                columns=self.columns,
                ncols=len(self.columns),
                header_size=header_size,
                output_path=chunk_output,
                compression='snappy',
                use_rmm=True,
                optimize_gpu=True,
                verbose=False
            )
            
            process_time = time.time() - process_start
            gpu_time = time.time() - gpu_start
            
            # 処理統計
            rows = len(cudf_df) if cudf_df is not None else 0
            parse_time = detailed_timing.get('gpu_parsing', 0)
            string_time = detailed_timing.get('string_buffer_creation', 0)
            write_time = detailed_timing.get('parquet_export', 0)
            
            print(f"[Consumer] チャンク {chunk_id + 1} GPU処理完了 ({gpu_time:.1f}秒, {rows:,}行)", flush=True)
            
            # 統計更新
            self.processed_count += 1
            self.total_gpu_time += gpu_time
            self.total_transfer_time += transfer_time
            self.total_rows += rows
            self.total_size += file_size
            
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
            
            # 詳細統計を保存
            chunk_stats.append({
                'chunk_id': chunk_id,
                'rust_time': chunk_info['rust_time'],
                'gpu_time': gpu_time,
                'transfer_time': transfer_time,
                'parse_time': parse_time,
                'string_time': string_time,
                'write_time': write_time,
                'rows': rows,
                'size_gb': file_size / 1024**3
            })
            
            return gpu_time, file_size, rows, detailed_timing, transfer_time
            
        except Exception as e:
            print(f"[Consumer] GPU処理エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # チャンクファイル削除
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            # Parquetファイル削除（検証省略）
            parquet_file = f"benchmark/chunk_{chunk_id}_ringbuffer.parquet"
            if os.path.exists(parquet_file):
                os.remove(parquet_file)
            # プロデューサーに消費完了を通知
            self.producer.mark_consumed(chunk_id)


def run_parallel_pipeline(columns: List[ColumnMeta], total_chunks: int):
    """並列パイプライン実行（簡易版：1つずつ処理）"""
    producer = RustProducer(total_chunks, RING_BUFFER_SIZE)
    consumer = GPUConsumer(columns, producer)
    
    # 統計情報
    total_rust_time = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:  # Producer + Consumer
        producer_future = None
        consumer_future = None
        next_chunk_id = 0
        processed_chunks = 0
        
        # 最初のチャンクを開始
        if next_chunk_id < total_chunks:
            producer_future = executor.submit(producer.produce_chunk, next_chunk_id)
            next_chunk_id += 1
        
        while processed_chunks < total_chunks and not shutdown_flag.is_set():
            # プロデューサーの完了を確認
            if producer_future and producer_future.done():
                chunk_info = producer_future.result()
                if chunk_info:
                    total_rust_time += chunk_info['rust_time']
                    
                    # 次のプロデューサーを開始
                    if next_chunk_id < total_chunks:
                        producer_future = executor.submit(producer.produce_chunk, next_chunk_id)
                        next_chunk_id += 1
                    else:
                        producer_future = None
                    
                    # コンシューマーを開始（前のが完了していれば）
                    if consumer_future is None or consumer_future.done():
                        consumer_future = executor.submit(consumer.consume_chunk, chunk_info)
                else:
                    producer_future = None
            
            # コンシューマーの完了を確認
            if consumer_future and consumer_future.done():
                result = consumer_future.result()
                if result:
                    processed_chunks += 1
                    print(f"[Main] チャンク処理完了 ({processed_chunks}/{total_chunks})")
                consumer_future = None
            
            # デバッグ情報
            if processed_chunks % 2 == 0:
                print(f"[Main] 進捗: {processed_chunks}/{total_chunks} 処理済み", flush=True)
            
            # 少し待機
            time.sleep(0.1)
        
        # 残りのタスクを待機
        if producer_future and not producer_future.done():
            producer_future.result()
        if consumer_future and not consumer_future.done():
            consumer_future.result()
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'total_rust_time': total_rust_time,
        'total_gpu_time': consumer.total_gpu_time,
        'total_transfer_time': consumer.total_transfer_time,
        'total_rows': consumer.total_rows,
        'total_size': consumer.total_size,
        'processed_chunks': processed_chunks
    }


def main(total_chunks=8):
    print("=== PostgreSQL → Rust → GPU リングバッファ版 ===")
    print(f"チャンク数: {total_chunks}")
    print(f"各チャンクサイズ: 約{52.86 / total_chunks:.1f} GB")
    print(f"リングバッファサイズ: {RING_BUFFER_SIZE}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print("\n改善内容:")
    print("  - リングバッファ方式で並列処理")
    print("  - Rust転送とGPU処理を同時実行")
    print("  - CPU待機時間を削減")
    
    # kvikio設定確認
    is_compat = os.environ.get("KVIKIO_COMPAT_MODE", "").lower() in ["on", "1", "true"]
    if is_compat:
        print("\n⚠️  kvikio互換モードで動作中")
    else:
        print("\n✅ kvikio GPUDirectモードの可能性")
    
    # RMMメモリプール設定
    setup_rmm_pool()
    
    # CUDA context確認
    try:
        cuda.current_context()
        print("\n✅ CUDA context OK")
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
        
        # 並列化による改善
        sequential_time = results['total_rust_time'] + results['total_gpu_time']
        parallel_speedup = sequential_time / results['total_time']
        print(f"\n【並列化効果】")
        print(f"├─ 逐次実行時間（推定）: {sequential_time:.2f}秒")
        print(f"├─ 並列実行時間: {results['total_time']:.2f}秒")
        print(f"└─ 高速化率: {parallel_speedup:.2f}倍")
        
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
        cleanup_files(total_chunks)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU リングバッファ版ベンチマーク')
    parser.add_argument('--table', type=str, default='lineorder', help='対象テーブル名')
    parser.add_argument('--parallel', type=int, default=16, help='並列接続数')
    parser.add_argument('--chunks', type=int, default=8, help='チャンク数')
    args = parser.parse_args()
    
    # 環境変数を設定
    os.environ['RUST_PARALLEL_CONNECTIONS'] = str(args.parallel)
    os.environ['TOTAL_CHUNKS'] = str(args.chunks)
    
    main(total_chunks=args.chunks)