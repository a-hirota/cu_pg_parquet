"""
PostgreSQL → GPU Ray並列 順次GPU処理版（直接GPU転送）
memoryviewから直接GPU転送により、中間バッファを排除

最適化:
- psycopg3のmemoryviewチャンクを直接GPU転送
- 中間バッファ（BytesIO、bytearray）を完全に排除
- GPU上でチャンクを結合
- 詳細なメモリメトリクス記録

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import math
import psycopg
import ray
import gc
import numpy as np
from numba import cuda
import argparse
from typing import List, Dict, Tuple, Optional
import psutil
import json
import cupy as cp

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_parallel_ctid_ray_sequential_gpu_direct.output"

# 並列設定
DEFAULT_PARALLEL = 16
CHUNK_COUNT = 4

# メトリクス記録用
class DetailedMetrics:
    def __init__(self, is_worker=False):
        self.is_worker = is_worker  # Ray Worker内かどうか
        self.metrics = {
            'memory': {
                'initial_system_mb': 0,
                'peak_system_mb': 0,
                'initial_gpu_mb': 0,
                'peak_gpu_mb': 0,
                'chunk_count': 0,
                'avg_chunk_size': 0,
                'total_data_size_mb': 0,
                'memory_overhead_ratio': 0
            },
            'performance': {
                'copy_time_sec': 0,
                'gpu_transfer_time_sec': 0,
                'gpu_transfer_count': 0,
                'gpu_processing_time_sec': 0,
                'total_time_sec': 0,
                'throughput_mb_sec': 0
            },
            'details': []
        }
    
    def log_memory_snapshot(self, stage: str):
        import psutil
        
        process = psutil.Process()
        system_memory_mb = process.memory_info().rss / (1024**2)
        
        # Ray Worker内ではGPUメモリを取得しない
        gpu_memory_mb = 0
        if not self.is_worker:
            try:
                mempool = cp.get_default_memory_pool()
                gpu_memory_mb = mempool.used_bytes() / (1024**2)
            except:
                gpu_memory_mb = 0
        
        self.metrics['details'].append({
            'timestamp': time.time(),
            'stage': stage,
            'system_memory_mb': system_memory_mb,
            'gpu_memory_mb': gpu_memory_mb
        })
        
        # ピーク値を更新
        if system_memory_mb > self.metrics['memory']['peak_system_mb']:
            self.metrics['memory']['peak_system_mb'] = system_memory_mb
        if gpu_memory_mb > self.metrics['memory']['peak_gpu_mb']:
            self.metrics['memory']['peak_gpu_mb'] = gpu_memory_mb
    
    def save_to_json(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)

@ray.remote  # CPUのみ、GPUはメインプロセスで使用
class PostgreSQLWorker:
    """Ray並列ワーカー: 直接GPU転送版"""
    
    def __init__(self, worker_id: int, dsn: str):
        self.worker_id = worker_id
        self.dsn = dsn
        
    def copy_data_to_memory_direct(
        self, 
        table_name: str,
        start_block: int, 
        end_block: int,
        chunk_idx: int,
        total_chunks: int,
        limit_rows: Optional[int] = None
    ) -> Dict[str, any]:
        """PostgreSQLからデータをCOPYしてCPUメモリに保持（GPU転送はメインプロセスで実行）"""
        
        # メトリクス初期化
        metrics = DetailedMetrics(is_worker=True)  # Ray Worker内なのでTrue
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        metrics.metrics['memory']['initial_system_mb'] = initial_memory
        metrics.log_memory_snapshot("start")
        
        # チャンクに応じてctid範囲を分割
        block_range = end_block - start_block
        chunk_block_size = block_range // total_chunks
        
        chunk_start_block = start_block + (chunk_idx * chunk_block_size)
        chunk_end_block = start_block + ((chunk_idx + 1) * chunk_block_size) if chunk_idx < total_chunks - 1 else end_block
        
        print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY開始 ctid範囲 ({chunk_start_block},{chunk_end_block})...")
        print(f"  初期メモリ使用量: {initial_memory:.2f} MB")
        
        start_time = time.time()
        
        try:
            # PostgreSQL接続
            conn = psycopg.connect(self.dsn)
            
            # CPUメモリへのチャンク保存用
            all_chunks = []  # memoryviewのまま保持
            chunk_count = 0
            total_size = 0
            chunk_sizes = []
            
            try:
                # COPY SQL生成
                copy_sql = self._make_copy_sql(table_name, chunk_start_block, chunk_end_block, limit_rows)
                
                # データ取得と直接GPU転送
                with conn.cursor() as cur:
                    with cur.copy(copy_sql) as copy_obj:
                        print(f"  CPUメモリへのデータ収集開始...")
                        
                        for chunk in copy_obj:
                            if chunk:
                                chunk_count += 1
                                chunk_size = len(chunk)
                                chunk_sizes.append(chunk_size)
                                total_size += chunk_size
                                
                                # memoryviewをbytesに変換して保存
                                if isinstance(chunk, memoryview):
                                    chunk_bytes = bytes(chunk)
                                else:
                                    chunk_bytes = chunk
                                
                                all_chunks.append(chunk_bytes)
                                
                                # 最初の10チャンクのサイズを表示
                                if chunk_count <= 10:
                                    print(f"    チャンク{chunk_count}: {chunk_size} bytes")
                                
                                # 定期的にメモリスナップショット
                                if chunk_count % 10000 == 0:
                                    metrics.log_memory_snapshot(f"chunk_{chunk_count}")
                
                # CPUでチャンクを結合
                print(f"  CPUでチャンク結合中... ({len(all_chunks)}個のチャンク)")
                concat_start = time.time()
                
                if all_chunks:
                    # bytearrayで効率的に結合
                    combined_data = bytearray()
                    for chunk in all_chunks:
                        combined_data.extend(chunk)
                    data_bytes = bytes(combined_data)
                    
                    # 中間バッファを即座に解放
                    del combined_data
                    del all_chunks
                else:
                    data_bytes = b''
                
                concat_time = time.time() - concat_start
                
                
                # メトリクス記録
                metrics.log_memory_snapshot("after_concat")
                copy_time = time.time() - start_time
                
                # チャンク統計
                if chunk_sizes:
                    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
                else:
                    avg_chunk_size = 0
                
                metrics.metrics['memory']['chunk_count'] = chunk_count
                metrics.metrics['memory']['avg_chunk_size'] = avg_chunk_size
                metrics.metrics['memory']['total_data_size_mb'] = total_size / (1024**2)
                metrics.metrics['performance']['copy_time_sec'] = copy_time
                
                # メモリオーバーヘッド計算
                current_memory = process.memory_info().rss / (1024**2)
                memory_used = current_memory - initial_memory
                if total_size > 0:
                    overhead_ratio = memory_used / (total_size / (1024**2))
                else:
                    overhead_ratio = 1.0
                metrics.metrics['memory']['memory_overhead_ratio'] = overhead_ratio
                
                print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY完了")
                print(f"  チャンク数: {chunk_count:,}")
                print(f"  平均チャンクサイズ: {avg_chunk_size:.0f} bytes")
                print(f"  総データサイズ: {total_size/(1024*1024):.1f} MB")
                print(f"  CPU結合時間: {concat_time:.2f}秒")
                print(f"  メモリ使用量: {current_memory:.2f} MB (増加: +{memory_used:.2f} MB)")
                print(f"  メモリオーバーヘッド率: {overhead_ratio:.2f}x")
                
                return {
                    'worker_id': self.worker_id,
                    'chunk_idx': chunk_idx,
                    'data': data_bytes,  # CPUで収集したデータ
                    'data_size': len(data_bytes),
                    'copy_time': copy_time,
                    'metrics': metrics.metrics,
                    'status': 'success'
                }
                
            finally:
                conn.close()
                # メモリ解放
                del all_chunks
                gc.collect()
                
        except Exception as e:
            print(f"Worker {self.worker_id}-Chunk{chunk_idx}: ❌COPY失敗 - {e}")
            return {
                'worker_id': self.worker_id,
                'chunk_idx': chunk_idx,
                'status': 'error',
                'error': str(e),
                'copy_time': time.time() - start_time,
                'metrics': metrics.metrics
            }
    
    def _make_copy_sql(self, table_name: str, start_block: int, end_block: int, limit_rows: Optional[int]) -> str:
        """COPY SQL生成"""
        sql = f"""
        COPY (
            SELECT * FROM {table_name}
            WHERE ctid >= '({start_block},1)'::tid
              AND ctid < '({end_block+1},1)'::tid
        ) TO STDOUT (FORMAT binary)"""
        
        return sql


def process_batch_on_gpu(
    batch_tasks: List[Dict],
    columns: List,
    gds_supported: bool,
    output_base: str,
    true_batch_mode: bool = False
) -> List[Dict]:
    """複数のメモリバッファを結合して1回のGPU処理で実行"""
    
    import cupy as cp
    
    num_tasks = len(batch_tasks)
    print(f"\n📊 バッチGPU処理開始: {num_tasks}個のタスクを統合処理")
    
    results = []
    start_total_time = time.time()
    
    try:
        # 1. 全データサイズを計算
        total_size = sum(task['data_size'] for task in batch_tasks)
        print(f"  統合データサイズ: {total_size/(1024*1024):.2f} MB")
        
        # 2. GPU配列を直接作成（ピンメモリ経由なし）
        start_gpu_transfer_time = time.time()
        gpu_array = cp.empty(total_size, dtype=cp.uint8)
        
        # 3. 各タスクのデータを直接GPU配列にコピー
        task_offsets = []
        current_offset = 0
        
        for task in batch_tasks:
            data_len = task['data_size']
            # データを直接GPU配列にコピー
            gpu_array[current_offset:current_offset + data_len] = cp.frombuffer(task['data'], dtype=cp.uint8)
            
            task_offsets.append({
                'worker_id': task['worker_id'],
                'chunk_idx': task['chunk_idx'],
                'offset': current_offset,
                'size': task['data_size']
            })
            current_offset += task['data_size']
        
        gpu_transfer_time = time.time() - start_gpu_transfer_time
        print(f"  GPU転送完了 ({gpu_transfer_time:.2f}秒, {(total_size/(1024*1024))/gpu_transfer_time:.2f} MB/sec)")
        
        # 以下、既存のGPU処理ロジックを使用
        start_gpu_processing_time = time.time()
        
        if true_batch_mode:
            # 真のバッチ処理モード
            print(f"\n  🚀 真のバッチ処理モード: {num_tasks}個のタスクを1回のGPU処理で実行")
            
            raw_dev = cuda.as_cuda_array(gpu_array)
            header_sample = gpu_array[:min(128, len(gpu_array))].get()
            header_size = detect_pg_header_size(header_sample)
            
            output_path = f"{output_base}_batch_integrated.parquet"
            
            try:
                cudf_df, detailed_timing = postgresql_to_cudf_parquet(
                    raw_dev=raw_dev,
                    columns=columns,
                    ncols=len(columns),
                    header_size=header_size,
                    output_path=output_path,
                    compression='snappy',
                    use_rmm=False,
                    optimize_gpu=True
                )
                
                print(f"  ✅ バッチ処理成功: {len(cudf_df):,} 行処理済み")
                
                results.append({
                    'worker_id': 'batch',
                    'chunk_idx': 0,
                    'gpu_transfer_time': gpu_transfer_time,
                    'gpu_processing_time': detailed_timing.get('overall_total', 0),
                    'rows_processed': len(cudf_df),
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"  ❌ バッチ処理失敗: {e}")
                true_batch_mode = False
        
        if not true_batch_mode:
            # 個別処理モード
            for i, (task, offset_info) in enumerate(zip(batch_tasks, task_offsets)):
                print(f"\n  処理中 [{i+1}/{num_tasks}]: Worker{offset_info['worker_id']}-Chunk{offset_info['chunk_idx']}")
                
                dataset_gpu = gpu_array[offset_info['offset']:offset_info['offset'] + offset_info['size']]
                raw_dev = cuda.as_cuda_array(dataset_gpu)
                header_sample = dataset_gpu[:min(128, len(dataset_gpu))].get()
                header_size = detect_pg_header_size(header_sample)
                
                output_path = f"{output_base}_worker{offset_info['worker_id']}_chunk{offset_info['chunk_idx']}.parquet"
                
                cudf_df, detailed_timing = postgresql_to_cudf_parquet(
                    raw_dev=raw_dev,
                    columns=columns,
                    ncols=len(columns),
                    header_size=header_size,
                    output_path=output_path,
                    compression='snappy',
                    use_rmm=False,
                    optimize_gpu=True
                )
                
                results.append({
                    'worker_id': offset_info['worker_id'],
                    'chunk_idx': offset_info['chunk_idx'],
                    'gpu_transfer_time': gpu_transfer_time / num_tasks,
                    'gpu_processing_time': detailed_timing.get('overall_total', 0),
                    'rows_processed': len(cudf_df),
                    'status': 'success'
                })
        
        gpu_processing_time = time.time() - start_gpu_processing_time
        total_time = time.time() - start_total_time
        
        print(f"\n✅ バッチGPU処理完了:")
        print(f"  総処理時間: {total_time:.2f}秒")
        print(f"  GPU転送: {gpu_transfer_time:.2f}秒")
        print(f"  GPU処理: {gpu_processing_time:.2f}秒")
        print(f"  スループット: {(total_size/(1024*1024))/total_time:.2f} MB/sec")
        
        # メモリ解放
        del gpu_array
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cuda.synchronize()
        
        return results
        
    except Exception as e:
        print(f"  ❌バッチGPU処理失敗: {e}")
        for task in batch_tasks:
            results.append({
                'worker_id': task['worker_id'],
                'chunk_idx': task['chunk_idx'],
                'status': 'error',
                'error': str(e)
            })
        
        try:
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cuda.synchronize()
        except:
            pass
            
        return results


def get_table_blocks(dsn: str, table_name: str) -> int:
    """テーブルの総ブロック数を取得"""
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT pg_relation_size('{table_name}') / 8192 AS blocks")
            blocks = cur.fetchone()[0]
            return int(blocks)
    finally:
        conn.close()


def make_ctid_ranges(total_blocks: int, parallel_count: int) -> List[Tuple[int, int]]:
    """ctid範囲リストを生成"""
    chunk_size = math.ceil(total_blocks / parallel_count)
    ranges = []
    
    for i in range(parallel_count):
        start_block = i * chunk_size
        end_block = min((i + 1) * chunk_size, total_blocks)
        if start_block < total_blocks:
            ranges.append((start_block, end_block))
    
    return ranges


def check_gpu_direct_support() -> bool:
    """GPU Direct サポート確認"""
    print("\n=== GPU Direct サポート確認 ===")
    
    try:
        if os.path.exists("/proc/driver/nvidia-fs/stats"):
            print("✅ nvidia-fs ドライバ検出")
        else:
            print("⚠️  nvidia-fs ドライバが見つかりません")
            return False
    except Exception:
        print("⚠️  nvidia-fs 確認エラー")
        return False
    
    try:
        import kvikio
        print(f"✅ kvikio バージョン: {kvikio.__version__}")
        os.environ["KVIKIO_COMPAT_MODE"] = "OFF"
        print("✅ KVIKIO_COMPAT_MODE=OFF 設定完了")
        return True
    except ImportError:
        print("⚠️  kvikio がインストールされていません")
        return False


def run_ray_parallel_sequential_gpu(
    limit_rows: int = 10000000,
    parallel_count: int = DEFAULT_PARALLEL,
    use_gpu_direct: bool = True,
    batch_size: int = 4,
    true_batch_mode: bool = False
):
    """Ray並列COPY + 順次GPU処理（直接GPU転送版）"""
    
    # 全体メトリクス
    global_metrics = DetailedMetrics()
    global_metrics.log_memory_snapshot("start")
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"=== PostgreSQL → GPU Ray並列 順次GPU処理版（直接GPU転送） ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}" if limit_rows else "行数制限: なし（全件処理）")
    print(f"並列数: {parallel_count}")
    print(f"チャンク数: {CHUNK_COUNT}")
    print(f"総タスク数: {parallel_count * CHUNK_COUNT}")
    print(f"バッチサイズ: {batch_size}")
    print(f"転送方式: 🚀 memoryview → GPU直接転送（中間バッファなし）")
    print(f"GPU処理方式: {'🚀 真のバッチ処理（実験的）' if true_batch_mode else '従来の個別処理'}")
    
    # GPU Direct サポート確認
    gds_supported = check_gpu_direct_support() if use_gpu_direct else False
    
    # Ray初期化
    print("\n=== Ray初期化 ===")
    if not ray.is_initialized():
        ray.init(num_cpus=parallel_count * 2)
        print(f"✅ Ray初期化完了")
    
    start_total_time = time.time()
    
    # メタデータ取得
    print("\nメタデータを取得中...")
    start_meta_time = time.time()
    conn = psycopg.connect(dsn)
    try:
        columns = fetch_column_meta(conn, f"SELECT * FROM {TABLE_NAME}")
        meta_time = time.time() - start_meta_time
        print(f"メタデータ取得完了 ({meta_time:.4f}秒)")
    finally:
        conn.close()
    
    global_metrics.log_memory_snapshot("after_metadata")
    
    # テーブルブロック数取得
    print("テーブルブロック数を取得中...")
    total_blocks = get_table_blocks(dsn, TABLE_NAME)
    print(f"総ブロック数: {total_blocks:,}")
    
    # ctid範囲分割
    ranges = make_ctid_ranges(total_blocks, parallel_count)
    print(f"ctid範囲分割: {len(ranges)}個の範囲")
    
    # Ray並列ワーカー作成
    workers = []
    for i in range(len(ranges)):
        worker = PostgreSQLWorker.remote(i, dsn)
        workers.append(worker)
    
    # 全タスクをキューに入れる
    print(f"\n=== フェーズ1: PostgreSQL COPY並列実行（直接GPU転送） ===")
    all_copy_futures = []
    
    for worker_idx, (start_block, end_block) in enumerate(ranges):
        for chunk_idx in range(CHUNK_COUNT):
            future = workers[worker_idx].copy_data_to_memory_direct.remote(
                TABLE_NAME, start_block, end_block, chunk_idx, CHUNK_COUNT, limit_rows
            )
            all_copy_futures.append(future)
    
    print(f"✅ {len(all_copy_futures)}個のCOPYタスクを投入")
    
    # セミパイプライン処理
    copy_results = []
    gpu_results = []
    pending_gpu_tasks = []
    
    total_copy_time = 0
    total_data_size = 0
    total_gpu_transfer_time = 0
    total_gpu_processing_time = 0
    total_rows_processed = 0
    all_worker_metrics = []
    
    output_base = OUTPUT_PARQUET_PATH
    
    print(f"\n=== セミパイプライン処理開始 ===")
    print(f"COPYが{batch_size}個完了するごとにGPU処理を開始します")
    
    while all_copy_futures or pending_gpu_tasks:
        # COPYタスクがまだある場合
        if all_copy_futures:
            num_to_wait = min(batch_size, len(all_copy_futures))
            ready_futures, remaining_futures = ray.wait(all_copy_futures, num_returns=num_to_wait, timeout=0.1)
            all_copy_futures = remaining_futures
            
            for future in ready_futures:
                result = ray.get(future)
                
                if result['status'] == 'success':
                    total_copy_time += result['copy_time']
                    total_data_size += result['data_size']
                    pending_gpu_tasks.append(result)
                    all_worker_metrics.append(result['metrics'])
                    print(f"✅ COPY完了: Worker{result['worker_id']}-Chunk{result['chunk_idx']} "
                          f"({result['data_size']/(1024*1024):.1f}MB)")
                    
                    # 統計情報のみを保持（大きなデータは除外）
                    copy_results.append({
                        'worker_id': result['worker_id'],
                        'chunk_idx': result['chunk_idx'],
                        'status': 'success',
                        'copy_time': result['copy_time'],
                        'data_size': result['data_size']
                    })
                else:
                    print(f"❌ COPY失敗: Worker{result['worker_id']}-Chunk{result['chunk_idx']} - "
                          f"{result.get('error', 'Unknown')}")
                    copy_results.append({
                        'worker_id': result['worker_id'],
                        'chunk_idx': result['chunk_idx'],
                        'status': 'error',
                        'error': result.get('error', 'Unknown')
                    })
        
        # GPU処理
        if (len(pending_gpu_tasks) >= batch_size) or (not all_copy_futures and pending_gpu_tasks):
            tasks_to_process = pending_gpu_tasks[:batch_size] if len(pending_gpu_tasks) >= batch_size else pending_gpu_tasks
            pending_gpu_tasks = pending_gpu_tasks[len(tasks_to_process):]
            
            global_metrics.log_memory_snapshot(f"before_gpu_batch_{len(gpu_results)}")
            
            batch_results = process_batch_on_gpu(
                tasks_to_process,
                columns,
                gds_supported,
                output_base,
                true_batch_mode=true_batch_mode
            )
            
            global_metrics.log_memory_snapshot(f"after_gpu_batch_{len(gpu_results)}")
            
            for gpu_result in batch_results:
                gpu_results.append(gpu_result)
                
                if gpu_result['status'] == 'success':
                    total_gpu_transfer_time += gpu_result['gpu_transfer_time']
                    total_gpu_processing_time += gpu_result['gpu_processing_time']
                    total_rows_processed += gpu_result['rows_processed']
            
            # GPU処理後にCPUメモリを解放
            for task in tasks_to_process:
                if 'data' in task:
                    del task['data']  # 大きなデータバッファを解放
            gc.collect()
            
            print(f"処理済みタスク累計: {len(gpu_results)}/{len(copy_results)}")
    
    # 成功した処理の数を集計
    successful_copies = [r for r in copy_results if r['status'] == 'success']
    successful_gpu = [r for r in gpu_results if r['status'] == 'success']
    
    # メトリクス集計
    if all_worker_metrics:
        total_chunk_count = sum(m['memory']['chunk_count'] for m in all_worker_metrics)
        avg_chunk_size = sum(m['memory']['avg_chunk_size'] * m['memory']['chunk_count'] for m in all_worker_metrics) / total_chunk_count
        total_gpu_transfer_count = sum(m['performance']['gpu_transfer_count'] for m in all_worker_metrics)
        
        global_metrics.metrics['memory']['chunk_count'] = total_chunk_count
        global_metrics.metrics['memory']['avg_chunk_size'] = avg_chunk_size
        global_metrics.metrics['performance']['gpu_transfer_count'] = total_gpu_transfer_count
    
    global_metrics.log_memory_snapshot("end")
    
    print(f"\n✅ 全処理完了")
    print(f"COPY: {len(successful_copies)}/{len(copy_results)} タスク成功")
    print(f"GPU: {len(successful_gpu)}/{len(gpu_results)} タスク成功")
    print(f"総データサイズ: {total_data_size / (1024*1024):.2f} MB")
    
    # 総合結果
    total_time = time.time() - start_total_time
    
    # グローバルメトリクス更新
    global_metrics.metrics['memory']['total_data_size_mb'] = total_data_size / (1024*1024)
    global_metrics.metrics['performance']['copy_time_sec'] = total_copy_time
    global_metrics.metrics['performance']['gpu_processing_time_sec'] = total_gpu_processing_time
    global_metrics.metrics['performance']['total_time_sec'] = total_time
    global_metrics.metrics['performance']['throughput_mb_sec'] = (total_data_size / (1024*1024)) / total_time if total_time > 0 else 0
    
    # メモリオーバーヘッド計算
    peak_memory = global_metrics.metrics['memory']['peak_system_mb']
    initial_memory = global_metrics.metrics['memory']['initial_system_mb']
    memory_used = peak_memory - initial_memory
    if total_data_size > 0:
        global_metrics.metrics['memory']['memory_overhead_ratio'] = memory_used / (total_data_size / (1024*1024))
    
    print(f"\n{'='*60}")
    print(f"=== Ray並列セミパイプラインGPU処理ベンチマーク完了（直接GPU転送） ===")
    print(f"{'='*60}")
    print(f"総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得    : {meta_time:.4f} 秒")
    print(f"  PostgreSQL COPY   : {total_copy_time:.4f} 秒 (累積)")
    print(f"  GPU転送          : {total_gpu_transfer_time:.4f} 秒 (累積)")
    print(f"  GPU処理          : {total_gpu_processing_time:.4f} 秒 (累積)")
    print("--- 統計情報 ---")
    print(f"  処理行数（合計）  : {total_rows_processed:,} 行")
    print(f"  処理列数         : {len(columns)} 列")
    print(f"  総データサイズ    : {total_data_size / (1024*1024):.2f} MB")
    print(f"  成功率           : COPY {len(successful_copies)}/{len(copy_results)}, "
          f"GPU {len(successful_gpu)}/{len(gpu_results)}")
    
    print("\n--- メモリ使用量 ---")
    print(f"  初期システムメモリ : {initial_memory:.2f} MB")
    print(f"  ピークシステムメモリ: {peak_memory:.2f} MB")
    print(f"  メモリ増加量      : {memory_used:.2f} MB")
    print(f"  メモリオーバーヘッド: {global_metrics.metrics['memory']['memory_overhead_ratio']:.2f}x")
    print(f"  総チャンク数      : {global_metrics.metrics['memory']['chunk_count']:,}")
    print(f"  平均チャンクサイズ : {global_metrics.metrics['memory']['avg_chunk_size']:.0f} bytes")
    print(f"  GPU転送回数       : {global_metrics.metrics['performance']['gpu_transfer_count']:,}")
    
    # パフォーマンス指標
    if total_data_size > 0:
        overall_throughput = (total_data_size / (1024*1024)) / total_time
        print("\n--- パフォーマンス指標 ---")
        print(f"  全体スループット  : {overall_throughput:.2f} MB/sec")
    
    print("\n--- 処理方式の特徴 ---")
    print("  ✅ 直接GPU転送: memoryview → GPU（中間バッファなし）")
    print("  ✅ メモリ効率: BytesIO/bytearrayを完全に排除")
    print("  ✅ GPU転送最適化: チャンクごとに直接転送")
    print("=========================================")
    
    # メトリクスをJSONファイルに保存
    metrics_file = f"{OUTPUT_PARQUET_PATH}_metrics_direct.json"
    global_metrics.save_to_json(metrics_file)
    print(f"\n✅ メトリクス保存: {metrics_file}")
    
    # Ray終了
    ray.shutdown()
    print("\n✅ Ray終了")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU Ray並列 直接GPU転送版')
    parser.add_argument('--rows', type=int, default=10000000, help='処理行数制限')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, help='並列数')
    parser.add_argument('--chunks', type=int, default=4, help='チャンク数')
    parser.add_argument('--batch-size', type=int, default=4, help='セミパイプラインのバッチサイズ')
    parser.add_argument('--true-batch', action='store_true', help='真のバッチGPU処理を有効化（実験的）')
    parser.add_argument('--no-limit', action='store_true', help='LIMIT無し（全件高速モード）')
    parser.add_argument('--check-support', action='store_true', help='GPU Directサポート確認のみ')
    parser.add_argument('--no-gpu-direct', action='store_true', help='GPU Direct無効化')
    
    args = parser.parse_args()
    
    # CUDA確認
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    if args.check_support:
        check_gpu_direct_support()
        return
    
    # チャンク数を設定
    global CHUNK_COUNT
    CHUNK_COUNT = args.chunks
    
    # ベンチマーク実行
    final_limit_rows = None if args.no_limit else args.rows
    run_ray_parallel_sequential_gpu(
        limit_rows=final_limit_rows,
        parallel_count=args.parallel,
        use_gpu_direct=not args.no_gpu_direct,
        batch_size=args.batch_size,
        true_batch_mode=args.true_batch
    )

if __name__ == "__main__":
    main()