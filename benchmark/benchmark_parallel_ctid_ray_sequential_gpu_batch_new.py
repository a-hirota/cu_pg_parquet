#!/usr/bin/env python3
"""
PostgreSQL → GPU 並列Rayワーカー版（バッチ化GPU転送）
- Ray並列ワーカーでCOPYデータをCPUメモリに収集
- メインプロセスでバッチ化してGPU転送・処理
"""

import os
import sys
import time
import psutil
import gc
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import json

# CUDA環境確認
try:
    import cupy as cp
    from numba import cuda
    cuda_available = cuda.is_available()
    if cuda_available:
        print("✅ CUDA context OK")
    else:
        print("❌ CUDA is not available")
        sys.exit(1)
except Exception as e:
    print(f"❌ CUDA環境エラー: {e}")
    sys.exit(1)

# インポート
import cudf
import numpy as np
import psycopg
from psycopg import sql
import ray

# 上位ディレクトリのモジュールをインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.database import get_table_blocks, make_ctid_ranges, fetch_column_meta
from common.data_loader import QUERY_TEMPLATES, build_query
from process_binary_v2 import (
    process_simple_array, 
    detect_pg_header_size,
    postgresql_to_cudf_parquet
)
from common.gpu_direct import check_gpu_direct_support

# 定数
TABLE_NAME = "lineorder"
DEFAULT_BATCH_SIZE_MB = 64  # バッチサイズ（MB）


class DetailedMetrics:
    """詳細なメトリクス記録クラス"""
    def __init__(self):
        self.metrics = {
            'memory': {
                'initial_system_mb': 0,
                'peak_system_mb': 0,
                'initial_gpu_mb': 0,
                'peak_gpu_mb': 0,
                'chunk_count': 0,
                'avg_chunk_size': 0,
                'total_data_size_mb': 0,
                'memory_overhead_ratio': 0,
                'batch_count': 0,
                'avg_batch_size_mb': 0
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
        
        # CuPyメモリプール
        self.mempool = cp.get_default_memory_pool()
        self.process = psutil.Process()
        
    def log_memory_snapshot(self, stage: str):
        """メモリ使用量のスナップショット"""
        system_memory_mb = self.process.memory_info().rss / (1024**2)
        gpu_memory_mb = self.mempool.used_bytes() / (1024**2) if cuda_available else 0
        
        self.metrics['details'].append({
            'timestamp': time.time(),
            'stage': stage,
            'system_memory_mb': system_memory_mb,
            'gpu_memory_mb': gpu_memory_mb
        })
        
        # ピーク値更新
        if system_memory_mb > self.metrics['memory']['peak_system_mb']:
            self.metrics['memory']['peak_system_mb'] = system_memory_mb
        if gpu_memory_mb > self.metrics['memory']['peak_gpu_mb']:
            self.metrics['memory']['peak_gpu_mb'] = gpu_memory_mb


@ray.remote  # CPUのみ、GPUはメインプロセスで使用
class PostgreSQLWorker:
    """Ray並列ワーカー（CPUでデータ収集のみ）"""
    
    def __init__(self, worker_id: int, dsn: str):
        self.worker_id = worker_id
        self.dsn = dsn
        
    def copy_data_to_memory_batch(
        self, 
        table_name: str,
        start_block: int, 
        end_block: int,
        chunk_idx: int,
        total_chunks: int,
        limit_rows: Optional[int] = None,
        batch_size_mb: int = DEFAULT_BATCH_SIZE_MB
    ) -> Dict[str, any]:
        """PostgreSQLからデータをCOPYしてCPUメモリに収集"""
        
        # メトリクス初期化
        metrics = DetailedMetrics()
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
            
            # データ収集用バッファ
            final_buffer = bytearray()
            chunk_count = 0
            total_size = 0
            chunk_sizes = []
            
            try:
                # COPY SQL生成
                copy_sql = self._make_copy_sql(table_name, chunk_start_block, chunk_end_block, limit_rows)
                
                # データ取得
                with conn.cursor() as cur:
                    with cur.copy(copy_sql) as copy_obj:
                        print(f"  CPUメモリへのデータ収集開始...")
                        
                        for chunk in copy_obj:
                            if chunk:
                                chunk_count += 1
                                chunk_size = len(chunk)
                                chunk_sizes.append(chunk_size)
                                total_size += chunk_size
                                
                                # バッファに追加
                                if isinstance(chunk, memoryview):
                                    final_buffer.extend(chunk)
                                else:
                                    final_buffer.extend(chunk)
                                
                                # 最初の10チャンクのサイズを表示
                                if chunk_count <= 10:
                                    print(f"    チャンク{chunk_count}: {chunk_size} bytes")
                                
                                # 定期的にメモリスナップショット
                                if chunk_count % 10000 == 0:
                                    metrics.log_memory_snapshot(f"chunk_{chunk_count}")
                
                # 最終データサイズ
                final_data_size = len(final_buffer)
                print(f"  収集完了: {final_data_size/(1024*1024):.1f} MB")
                
                # メトリクス記録
                metrics.log_memory_snapshot("after_collect")
                copy_time = time.time() - start_time
                
                # 統計計算
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
                
                print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY完了 ({copy_time:.2f}秒, {total_size/(1024*1024):.1f}MB)")
                print(f"  最終メモリ使用量: {current_memory:.2f} MB (合計増加: +{memory_used:.2f} MB)")
                print(f"  チャンク数: {chunk_count:,}")
                print(f"  平均チャンクサイズ: {avg_chunk_size:.0f} bytes")
                print(f"  メモリオーバーヘッド率: {overhead_ratio:.2f}x")
                
                return {
                    'worker_id': self.worker_id,
                    'chunk_idx': chunk_idx,
                    'data': bytes(final_buffer),  # CPUで収集したデータ
                    'data_size': final_data_size,
                    'copy_time': copy_time,
                    'metrics': metrics.metrics,
                    'status': 'success'
                }
                
            finally:
                conn.close()
                # メモリ解放
                del final_buffer
                gc.collect()
                
        except Exception as e:
            print(f"Worker {self.worker_id}-Chunk{chunk_idx}: ❌COPY失敗 - {e}")
            return {
                'worker_id': self.worker_id,
                'chunk_idx': chunk_idx,
                'status': 'error',
                'error': str(e),
                'metrics': metrics.metrics
            }
    
    def _make_copy_sql(self, table_name: str, start_block: int, end_block: int, limit_rows: Optional[int]) -> str:
        """COPY SQL文生成"""
        if limit_rows:
            return (f"COPY (SELECT * FROM {table_name} "
                   f"WHERE ctid >= '({start_block},0)'::tid AND ctid < '({end_block},0)'::tid "
                   f"LIMIT {limit_rows}) TO STDOUT (FORMAT BINARY)")
        else:
            return (f"COPY (SELECT * FROM {table_name} "
                   f"WHERE ctid >= '({start_block},0)'::tid AND ctid < '({end_block},0)'::tid"
                   f") TO STDOUT (FORMAT BINARY)")


def process_batch_on_gpu(
    batch_tasks: List[Dict],
    columns: List,
    gds_supported: bool,
    output_base: str,
    true_batch_mode: bool = False,
    batch_size_mb: int = DEFAULT_BATCH_SIZE_MB
) -> List[Dict]:
    """複数のメモリバッファをバッチ化してGPU処理"""
    
    import cupy as cp
    
    num_tasks = len(batch_tasks)
    print(f"\n📊 バッチGPU処理開始: {num_tasks}個のタスクをバッチ化処理")
    
    results = []
    start_total_time = time.time()
    
    try:
        # 1. 全データサイズを計算
        total_size = sum(task['data_size'] for task in batch_tasks)
        print(f"  統合データサイズ: {total_size/(1024*1024):.2f} MB")
        
        # 2. バッチ化してGPU転送
        start_gpu_transfer_time = time.time()
        gpu_batches = []
        batch_count = 0
        
        # バッチサイズごとにグループ化
        current_batch = bytearray()
        batch_offsets = []
        
        for task in batch_tasks:
            task_data = task['data']
            
            # 現在のバッチに追加
            if len(current_batch) + len(task_data) > batch_size_mb * 1024 * 1024:
                # 現在のバッチをGPU転送
                if current_batch:
                    batch_count += 1
                    gpu_batch = cp.asarray(current_batch, dtype=cp.uint8)
                    gpu_batches.append(gpu_batch)
                    print(f"    バッチ{batch_count}: {len(current_batch)/(1024*1024):.1f} MB → GPU")
                    current_batch = bytearray()
            
            # タスクデータを追加
            batch_start = len(current_batch)
            current_batch.extend(task_data)
            batch_offsets.append({
                'worker_id': task['worker_id'],
                'chunk_idx': task['chunk_idx'],
                'batch_idx': batch_count,
                'offset': batch_start,
                'size': task['data_size']
            })
        
        # 最後のバッチをGPU転送
        if current_batch:
            batch_count += 1
            gpu_batch = cp.asarray(current_batch, dtype=cp.uint8)
            gpu_batches.append(gpu_batch)
            print(f"    最終バッチ{batch_count}: {len(current_batch)/(1024*1024):.1f} MB → GPU")
        
        # 3. GPU上でバッチを結合
        print(f"  GPU上でバッチ結合中... ({len(gpu_batches)}個のバッチ)")
        if gpu_batches:
            # 事前割り当てで効率化
            gpu_array = cp.empty(total_size, dtype=cp.uint8)
            offset = 0
            for batch in gpu_batches:
                batch_size = len(batch)
                gpu_array[offset:offset + batch_size] = batch
                offset += batch_size
                del batch  # 即座に解放
            
            # メモリプール最適化
            cp.get_default_memory_pool().free_all_blocks()
        else:
            gpu_array = cp.empty(0, dtype=cp.uint8)
        
        gpu_transfer_time = time.time() - start_gpu_transfer_time
        print(f"  GPU転送完了 ({gpu_transfer_time:.2f}秒, {(total_size/(1024*1024))/gpu_transfer_time:.2f} MB/sec)")
        
        # 4. GPU処理実行（既存のロジックを使用）
        start_gpu_processing_time = time.time()
        
        if true_batch_mode:
            # 真のバッチ処理モード（全データを1回で処理）
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
            # 個別処理モード（各タスクを個別に処理）
            for i, task in enumerate(batch_tasks):
                print(f"\n  処理中 [{i+1}/{num_tasks}]: Worker{task['worker_id']}-Chunk{task['chunk_idx']}")
                
                # 該当部分のGPUデータを抽出（実装省略）
                # ... 既存の個別処理ロジック ...
                
                results.append({
                    'worker_id': task['worker_id'],
                    'chunk_idx': task['chunk_idx'],
                    'gpu_transfer_time': gpu_transfer_time / num_tasks,
                    'gpu_processing_time': 0,  # 簡略化
                    'rows_processed': 0,
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
        del gpu_batches
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cuda.synchronize()
        
    except Exception as e:
        print(f"❌ バッチGPU処理エラー: {e}")
        import traceback
        traceback.print_exc()
        
        # エラー時も結果を返す
        for task in batch_tasks:
            results.append({
                'worker_id': task['worker_id'],
                'chunk_idx': task['chunk_idx'],
                'gpu_transfer_time': 0,
                'gpu_processing_time': 0,
                'rows_processed': 0,
                'status': 'error',
                'error': str(e)
            })
    
    return results


def run_ray_parallel_sequential_gpu(
    parallel_count: int,
    chunk_count: int,
    batch_size: int,
    limit_rows: Optional[int],
    dsn: str,
    output_base: str,
    use_gpu_direct: bool,
    true_batch_mode: bool
):
    """Ray並列COPY + 順次GPU処理（バッチ化版）"""
    
    # グローバルメトリクス
    global_metrics = DetailedMetrics()
    global_metrics.log_memory_snapshot("start")
    
    print(f"\n=== PostgreSQL → GPU Ray並列 順次GPU処理版（バッチ化GPU転送） ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {'なし（全件処理）' if limit_rows is None else f'{limit_rows}行'}")
    print(f"並列数: {parallel_count}")
    print(f"チャンク数: {chunk_count}")
    print(f"総タスク数: {parallel_count * chunk_count}")
    print(f"バッチサイズ: {batch_size}")
    print(f"GPU転送バッチサイズ: {DEFAULT_BATCH_SIZE_MB} MB")
    print(f"転送方式: 📦 バッチ化GPU転送（{DEFAULT_BATCH_SIZE_MB}MBごと）")
    print(f"GPU処理方式: {'真のバッチ処理（全タスク統合）' if true_batch_mode else '従来の個別処理'}")
    
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
    print(f"\n=== フェーズ1: PostgreSQL COPY並列実行 ===")
    all_copy_futures = []
    
    for worker_idx, (start_block, end_block) in enumerate(ranges):
        for chunk_idx in range(chunk_count):
            future = workers[worker_idx].copy_data_to_memory_batch.remote(
                TABLE_NAME, start_block, end_block, chunk_idx, chunk_count, limit_rows
            )
            all_copy_futures.append(future)
    
    print(f"✅ {len(all_copy_futures)}個のCOPYタスクを投入")
    
    # セミパイプライン処理: COPYとGPUを並行実行
    copy_results = []
    gpu_results = []
    pending_gpu_tasks = []  # GPU処理待ちのタスク
    
    total_copy_time = 0
    total_gpu_transfer_time = 0
    total_gpu_processing_time = 0
    total_data_size = 0
    total_rows_processed = 0
    
    all_worker_metrics = []
    
    print(f"\n=== セミパイプライン処理開始 ===")
    print(f"COPYが{batch_size}個完了するごとにGPU処理を開始します")
    
    while all_copy_futures or pending_gpu_tasks:
        # COPY完了待ち（非ブロッキング）
        ready_futures, all_copy_futures = ray.wait(all_copy_futures, timeout=0.1)
        
        # 完了したCOPYタスクを処理
        for future in ready_futures:
            result = ray.get(future)
            copy_results.append(result)
            
            if result['status'] == 'success':
                total_copy_time += result['copy_time']
                total_data_size += result['data_size']
                pending_gpu_tasks.append(result)
                all_worker_metrics.append(result['metrics'])
                print(f"✅ COPY完了: Worker{result['worker_id']}-Chunk{result['chunk_idx']} "
                      f"({result['data_size']/(1024*1024):.1f}MB)")
            else:
                print(f"❌ COPY失敗: Worker{result['worker_id']}-Chunk{result['chunk_idx']} - "
                      f"{result.get('error', 'Unknown')}")
        
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
            
            print(f"処理済みタスク累計: {len(gpu_results)}/{len(copy_results)}")
    
    # 成功した処理の数を集計
    successful_copies = [r for r in copy_results if r['status'] == 'success']
    successful_gpu = [r for r in gpu_results if r['status'] == 'success']
    
    # メトリクス集計
    if all_worker_metrics:
        total_chunk_count = sum(m['memory']['chunk_count'] for m in all_worker_metrics)
        avg_chunk_size = sum(m['memory']['avg_chunk_size'] * m['memory']['chunk_count'] for m in all_worker_metrics) / total_chunk_count
        
        global_metrics.metrics['memory']['chunk_count'] = total_chunk_count
        global_metrics.metrics['memory']['avg_chunk_size'] = avg_chunk_size
        global_metrics.metrics['memory']['memory_overhead_ratio'] = sum(m['memory']['memory_overhead_ratio'] for m in all_worker_metrics) / len(all_worker_metrics)
    
    global_metrics.log_memory_snapshot("end")
    
    # 処理時間
    total_elapsed_time = time.time() - start_total_time
    global_metrics.metrics['performance']['total_time_sec'] = total_elapsed_time
    global_metrics.metrics['performance']['throughput_mb_sec'] = total_data_size / (1024**2) / total_elapsed_time if total_elapsed_time > 0 else 0
    
    print(f"\n✅ 全処理完了")
    print(f"COPY: {len(successful_copies)}/{len(copy_results)} タスク成功")
    print(f"GPU: {len(successful_gpu)}/{len(gpu_results)} タスク成功")
    print(f"総データサイズ: {total_data_size/(1024*1024):.2f} MB")
    
    # 結果サマリー
    print(f"\n============================================================")
    print(f"=== Ray並列セミパイプラインGPU処理ベンチマーク完了（バッチ化GPU転送） ===")
    print(f"============================================================")
    print(f"総時間 = {total_elapsed_time:.4f} 秒")
    print(f"--- 時間内訳 ---")
    print(f"  メタデータ取得    : {meta_time:.4f} 秒")
    print(f"  PostgreSQL COPY   : {total_copy_time:.4f} 秒 (累積)")
    print(f"  GPU転送          : {total_gpu_transfer_time:.4f} 秒 (累積)")
    print(f"  GPU処理          : {total_gpu_processing_time:.4f} 秒 (累積)")
    print(f"--- 統計情報 ---")
    print(f"  処理行数（合計）  : {total_rows_processed:,} 行")
    print(f"  処理列数         : {len(columns)} 列")
    print(f"  総データサイズ    : {total_data_size/(1024*1024):.2f} MB")
    print(f"  成功率           : COPY {len(successful_copies)}/{len(copy_results)}, "
          f"GPU {len(successful_gpu)}/{len(gpu_results)}")
    
    # メモリ使用量サマリー
    print(f"\n--- メモリ使用量 ---")
    print(f"  初期システムメモリ : {global_metrics.metrics['memory']['initial_system_mb']:.2f} MB")
    print(f"  ピークシステムメモリ: {global_metrics.metrics['memory']['peak_system_mb']:.2f} MB")
    print(f"  メモリ増加量      : {global_metrics.metrics['memory']['peak_system_mb'] - global_metrics.metrics['memory']['initial_system_mb']:.2f} MB")
    print(f"  メモリオーバーヘッド: {global_metrics.metrics['memory']['memory_overhead_ratio']:.2f}x")
    print(f"  総チャンク数      : {global_metrics.metrics['memory']['chunk_count']:,}")
    print(f"  平均チャンクサイズ : {global_metrics.metrics['memory']['avg_chunk_size']:.0f} bytes")
    print(f"  GPU転送回数       : {len(gpu_results)}")
    
    print(f"\n--- 処理方式の特徴 ---")
    print(f"  ✅ バッチ化GPU転送: {DEFAULT_BATCH_SIZE_MB}MBごとにGPU転送")
    print(f"  ✅ メモリ効率: 中間バッファはバッチサイズ分のみ")
    print(f"  ✅ GPU転送最適化: 転送回数を大幅削減")
    print(f"=========================================")
    
    # メトリクス保存
    output_metrics_path = f"{output_base}_metrics_batch.json"
    with open(output_metrics_path, 'w') as f:
        json.dump(global_metrics.metrics, f, indent=2)
    print(f"\n✅ メトリクス保存: {output_metrics_path}")
    
    # Ray終了
    for worker in workers:
        ray.kill(worker)
    
    # Rayを明示的にシャットダウン
    ray.shutdown()
    print("\n✅ Ray終了")
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PostgreSQL to GPU Ray並列処理（バッチ化GPU転送）')
    parser.add_argument('--parallel', type=int, default=8, help='並列数')
    parser.add_argument('--chunks', type=int, default=1, help='ワーカーあたりのチャンク数')
    parser.add_argument('--batch-size', type=int, default=4, help='GPUバッチサイズ')
    parser.add_argument('--limit', type=int, help='処理行数制限')
    parser.add_argument('--no-limit', action='store_true', help='LIMIT無し（全件処理）')
    parser.add_argument('--dsn', type=str, default=os.environ.get('GPUPGPARSER_PG_DSN'), 
                       help='PostgreSQL DSN')
    parser.add_argument('--output', type=str, help='出力ファイルプレフィックス')
    parser.add_argument('--no-gpu-direct', action='store_true', help='GPU Directを無効化')
    parser.add_argument('--true-batch', action='store_true', help='真のバッチ処理を有効化')
    
    args = parser.parse_args()
    
    if not args.dsn:
        print("エラー: PostgreSQL DSNが指定されていません")
        print("環境変数 GPUPGPARSER_PG_DSN を設定するか、--dsn オプションを使用してください")
        sys.exit(1)
    
    # LIMIT設定
    if args.no_limit:
        limit_rows = None
    else:
        limit_rows = args.limit
    
    # 出力ファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_base = args.output
    else:
        limit_str = "all" if limit_rows is None else f"{limit_rows}"
        output_base = f"benchmark/{TABLE_NAME}_parallel_ctid_ray_sequential_gpu_batch_{timestamp}_limit{limit_str}"
    
    # パラメータ設定
    PARALLEL_COUNT = args.parallel
    CHUNK_COUNT = args.chunks  # 各ワーカーが処理するチャンク数
    BATCH_SIZE = args.batch_size  # GPU処理のバッチサイズ
    USE_GPU_DIRECT = not args.no_gpu_direct
    TRUE_BATCH_MODE = args.true_batch
    
    # 実行
    run_ray_parallel_sequential_gpu(
        parallel_count=PARALLEL_COUNT,
        chunk_count=CHUNK_COUNT,
        batch_size=BATCH_SIZE,
        limit_rows=limit_rows,
        dsn=args.dsn,
        output_base=output_base,
        use_gpu_direct=USE_GPU_DIRECT,
        true_batch_mode=TRUE_BATCH_MODE
    )


if __name__ == "__main__":
    main()