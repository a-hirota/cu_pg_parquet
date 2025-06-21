"""
PostgreSQL → GPU Ray並列 順次GPU処理版（詳細デバッグ版）
psycopg3のボトルネック分析用の詳細プロファイリング機能付き

最適化:
- 16並列でPostgreSQL COPYを実行
- GPU処理は1つずつ順次実行（メモリ競合回避）
- CuPy配列による安定したメモリ管理
- 64個の独立したParquetファイル出力

デバッグ機能:
- psycopg3チャンクサイズ可視化
- 詳細タイミング測定
- CSV/JSON結果出力

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import math
import tempfile
import psycopg
import ray
import gc
import numpy as np
from numba import cuda
import argparse
from typing import List, Dict, Tuple, Optional
import json
import csv
from datetime import datetime

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_parallel_ctid_ray_sequential_gpu_debug.output"
DEBUG_OUTPUT_DIR = "benchmark/debug_output"

# 並列設定
DEFAULT_PARALLEL = 16
CHUNK_COUNT = 4  # 各ワーカーのctid範囲をチャンクに分割

# デバッグ用グローバル統計
DEBUG_STATS = {
    'chunk_sizes': [],
    'chunk_times': [],
    'connection_times': [],
    'copy_init_times': [],
    'buffer_operations': []
}

@ray.remote
class PostgreSQLWorker:
    """Ray並列ワーカー: PostgreSQL COPY専用（デバッグ機能付き）"""
    
    def __init__(self, worker_id: int, dsn: str):
        self.worker_id = worker_id
        self.dsn = dsn
        self.debug_stats = {
            'chunks_received': [],
            'chunk_times': [],
            'buffer_resizes': 0
        }
        
    def copy_data_to_memory(
        self, 
        table_name: str,
        start_block: int, 
        end_block: int,
        chunk_idx: int,
        total_chunks: int,
        limit_rows: Optional[int] = None,
        initial_buffer_size: int = 64 * 1024 * 1024,  # 64MB
        debug_mode: bool = True
    ) -> Dict[str, any]:
        """PostgreSQLからデータをCOPYしてピンメモリに保存（詳細デバッグ付き）"""
        
        # チャンクに応じてctid範囲を分割
        block_range = end_block - start_block
        chunk_block_size = block_range // total_chunks
        
        chunk_start_block = start_block + (chunk_idx * chunk_block_size)
        chunk_end_block = start_block + ((chunk_idx + 1) * chunk_block_size) if chunk_idx < total_chunks - 1 else end_block
        
        print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY開始 ctid範囲 ({chunk_start_block},{chunk_end_block})...")
        
        start_time = time.time()
        timing_details = {
            'connection': 0,
            'copy_init': 0,
            'chunk_reads': [],
            'buffer_ops': 0,
            'total': 0
        }
        
        try:
            # PostgreSQL接続
            conn_start = time.time()
            conn = psycopg.connect(self.dsn)
            timing_details['connection'] = time.time() - conn_start
            
            # 通常のbytearrayを使用（Rayワーカーでピンメモリ確保は困難）
            buffer = bytearray()
            chunk_count = 0
            total_bytes = 0
            
            try:
                # COPY SQL生成
                copy_sql = self._make_copy_sql(table_name, chunk_start_block, chunk_end_block, limit_rows)
                
                # データ取得
                copy_init_start = time.time()
                with conn.cursor() as cur:
                    with cur.copy(copy_sql) as copy_obj:
                        timing_details['copy_init'] = time.time() - copy_init_start
                        
                        for chunk in copy_obj:
                            chunk_start = time.time()
                            if chunk:
                                # チャンクサイズ記録
                                chunk_size = len(chunk)
                                chunk_count += 1
                                
                                # memoryview → bytes変換
                                if isinstance(chunk, memoryview):
                                    chunk_bytes = chunk.tobytes()
                                else:
                                    chunk_bytes = bytes(chunk)
                                
                                # bytearrayに追加
                                buffer_start = time.time()
                                buffer.extend(chunk_bytes)
                                buffer_time = time.time() - buffer_start
                                
                                chunk_time = time.time() - chunk_start
                                total_bytes += chunk_size
                                
                                # デバッグ情報記録
                                if debug_mode:
                                    timing_details['chunk_reads'].append({
                                        'chunk_num': chunk_count,
                                        'size': chunk_size,
                                        'time': chunk_time,
                                        'buffer_time': buffer_time,
                                        'cumulative_size': total_bytes
                                    })
                                    
                                    # 10チャンクごとに進捗表示
                                    if chunk_count % 10 == 0:
                                        avg_chunk_size = total_bytes / chunk_count
                                        print(f"  Worker{self.worker_id}-Chunk{chunk_idx}: "
                                              f"{chunk_count}チャンク読込済 "
                                              f"(平均{avg_chunk_size/1024:.1f}KB/chunk, "
                                              f"合計{total_bytes/(1024*1024):.1f}MB)")
                
                copy_time = time.time() - start_time
                data_size = len(buffer)  # 実際のデータサイズ
                
                print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY完了 "
                      f"({copy_time:.2f}秒, {data_size/(1024*1024):.1f}MB, "
                      f"{chunk_count}チャンク)")
                
                # デバッグ統計サマリー
                if debug_mode and timing_details['chunk_reads']:
                    avg_chunk_size = np.mean([c['size'] for c in timing_details['chunk_reads']])
                    max_chunk_size = max([c['size'] for c in timing_details['chunk_reads']])
                    min_chunk_size = min([c['size'] for c in timing_details['chunk_reads']])
                    
                    print(f"  チャンクサイズ統計: 平均{avg_chunk_size/1024:.1f}KB, "
                          f"最小{min_chunk_size/1024:.1f}KB, 最大{max_chunk_size/1024:.1f}KB")
                
                timing_details['total'] = copy_time
                
                # bytearrayをbytesに変換して返す
                return {
                    'worker_id': self.worker_id,
                    'chunk_idx': chunk_idx,
                    'data': bytes(buffer),  # bytesに変換してシリアライズ可能に
                    'data_size': data_size,
                    'copy_time': copy_time,
                    'chunk_count': chunk_count,
                    'timing_details': timing_details,
                    'status': 'success'
                }
                
            finally:
                conn.close()
                
        except Exception as e:
            print(f"Worker {self.worker_id}-Chunk{chunk_idx}: ❌COPY失敗 - {e}")
            return {
                'worker_id': self.worker_id,
                'chunk_idx': chunk_idx,
                'status': 'error',
                'error': str(e),
                'copy_time': time.time() - start_time,
                'timing_details': timing_details
            }
    
    def _make_copy_sql(self, table_name: str, start_block: int, end_block: int, limit_rows: Optional[int]) -> str:
        """COPY SQL生成"""
        sql = f"""
        COPY (
            SELECT * FROM {table_name}
            WHERE ctid >= '({start_block},1)'::tid
              AND ctid < '({end_block+1},1)'::tid
        """
        
        if limit_rows:
            # 64タスク全体での行数制限
            sql += f" LIMIT {limit_rows // (DEFAULT_PARALLEL * CHUNK_COUNT)}"
        
        sql += ") TO STDOUT (FORMAT binary)"
        return sql


def process_batch_on_gpu(
    batch_tasks: List[Dict],
    columns: List,
    gds_supported: bool,
    output_base: str,
    debug_mode: bool = True
) -> List[Dict]:
    """複数のメモリバッファを結合して1回のGPU処理で実行（デバッグ付き）"""
    
    import cupy as cp  # メインプロセスでインポート
    
    num_tasks = len(batch_tasks)
    print(f"\n📊 バッチGPU処理開始: {num_tasks}個のタスクを統合処理")
    
    results = []
    start_total_time = time.time()
    timing_breakdown = {
        'pinned_alloc': 0,
        'data_copy_to_pinned': 0,
        'gpu_alloc': 0,
        'gpu_transfer': 0,
        'gpu_processing': [],
        'memory_cleanup': 0
    }
    
    try:
        # 1. 全データサイズを計算
        total_size = sum(task['data_size'] for task in batch_tasks)
        print(f"  統合データサイズ: {total_size/(1024*1024):.2f} MB")
        
        # 2. cupyxの高レベルAPIを使用してピンメモリ配列を作成
        pinned_start = time.time()
        import cupyx
        pinned_array = cupyx.zeros_pinned(total_size, dtype=np.uint8)
        timing_breakdown['pinned_alloc'] = time.time() - pinned_start
        
        # 3. 各タスクのデータをピンメモリにコピー
        copy_start = time.time()
        task_offsets = []
        current_offset = 0
        
        for task in batch_tasks:
            # データを直接ピンメモリにコピー
            data_len = task['data_size']
            pinned_array[current_offset:current_offset + data_len] = np.frombuffer(task['data'], dtype=np.uint8)
            
            task_offsets.append({
                'worker_id': task['worker_id'],
                'chunk_idx': task['chunk_idx'],
                'offset': current_offset,
                'size': task['data_size']
            })
            current_offset += task['data_size']
        
        timing_breakdown['data_copy_to_pinned'] = time.time() - copy_start
        
        # 4. GPU配列を確保し、ピンメモリから高速転送（set()を使用）
        gpu_alloc_start = time.time()
        gpu_array = cp.empty(total_size, dtype=cp.uint8)
        timing_breakdown['gpu_alloc'] = time.time() - gpu_alloc_start
        
        gpu_transfer_start = time.time()
        gpu_array.set(pinned_array)  # 効率的なコピー方法
        cuda.synchronize()  # 転送完了を待つ
        timing_breakdown['gpu_transfer'] = time.time() - gpu_transfer_start
        
        gpu_transfer_bandwidth = (total_size / (1024**3)) / timing_breakdown['gpu_transfer']  # GB/s
        print(f"  GPU転送完了 ({timing_breakdown['gpu_transfer']:.2f}秒, {gpu_transfer_bandwidth:.2f} GB/s)")
        
        # 5. 統合データで1回のGPU処理を実行
        start_gpu_processing_time = time.time()
        
        # 各データセットを個別に処理（将来的には統合処理に拡張可能）
        for i, (task, offset_info) in enumerate(zip(batch_tasks, task_offsets)):
            gpu_task_start = time.time()
            print(f"\n  処理中 [{i+1}/{num_tasks}]: Worker{offset_info['worker_id']}-Chunk{offset_info['chunk_idx']}")
            
            # データセットの切り出し
            dataset_gpu = gpu_array[offset_info['offset']:offset_info['offset'] + offset_info['size']]
            
            # numba用の配列に変換
            raw_dev = cuda.as_cuda_array(dataset_gpu)
            header_sample = dataset_gpu[:min(128, len(dataset_gpu))].get()
            header_size = detect_pg_header_size(header_sample)
            
            # 出力パス
            output_path = f"{output_base}_worker{offset_info['worker_id']}_chunk{offset_info['chunk_idx']}.parquet"
            
            # GPU最適化処理
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
            
            gpu_task_time = time.time() - gpu_task_start
            timing_breakdown['gpu_processing'].append({
                'task_id': f"Worker{offset_info['worker_id']}-Chunk{offset_info['chunk_idx']}",
                'time': gpu_task_time,
                'rows': len(cudf_df),
                'detailed': detailed_timing
            })
            
            results.append({
                'worker_id': offset_info['worker_id'],
                'chunk_idx': offset_info['chunk_idx'],
                'gpu_transfer_time': timing_breakdown['gpu_transfer'] / num_tasks,  # 転送時間を分配
                'gpu_processing_time': gpu_task_time,
                'rows_processed': len(cudf_df),
                'status': 'success'
            })
        
        gpu_processing_time = time.time() - start_gpu_processing_time
        total_time = time.time() - start_total_time
        
        # メモリ解放
        cleanup_start = time.time()
        del gpu_array
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cuda.synchronize()
        timing_breakdown['memory_cleanup'] = time.time() - cleanup_start
        
        print(f"\n✅ バッチGPU処理完了:")
        print(f"  総処理時間: {total_time:.2f}秒")
        if debug_mode:
            print(f"  --- 詳細タイミング ---")
            print(f"  ピンメモリ確保: {timing_breakdown['pinned_alloc']:.3f}秒")
            print(f"  データコピー: {timing_breakdown['data_copy_to_pinned']:.3f}秒")
            print(f"  GPU配列確保: {timing_breakdown['gpu_alloc']:.3f}秒")
            print(f"  GPU転送: {timing_breakdown['gpu_transfer']:.3f}秒 ({gpu_transfer_bandwidth:.2f} GB/s)")
            print(f"  GPU処理: {gpu_processing_time:.3f}秒")
            print(f"  メモリ解放: {timing_breakdown['memory_cleanup']:.3f}秒")
        print(f"  スループット: {(total_size/(1024*1024))/total_time:.2f} MB/sec")
        
        # タイミング詳細を結果に追加
        for result in results:
            result['timing_breakdown'] = timing_breakdown
        
        return results
        
    except Exception as e:
        print(f"  ❌バッチGPU処理失敗: {e}")
        # エラー時は個別にエラー結果を返す
        for task in batch_tasks:
            results.append({
                'worker_id': task['worker_id'],
                'chunk_idx': task['chunk_idx'],
                'status': 'error',
                'error': str(e),
                'timing_breakdown': timing_breakdown
            })
        
        # メモリ解放を試みる
        try:
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cuda.synchronize()
        except:
            pass
            
        return results


def save_debug_results(all_results: Dict, output_dir: str = DEBUG_OUTPUT_DIR):
    """デバッグ結果をCSVとJSONで保存"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON出力（全詳細）
    json_path = os.path.join(output_dir, f"debug_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ 詳細結果をJSON保存: {json_path}")
    
    # CSV出力（サマリー）
    csv_path = os.path.join(output_dir, f"debug_summary_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ヘッダー
        writer.writerow([
            'worker_id', 'chunk_idx', 'status',
            'data_size_mb', 'copy_time', 'chunk_count',
            'avg_chunk_kb', 'connection_time', 'copy_init_time',
            'gpu_transfer_time', 'gpu_processing_time', 'rows_processed'
        ])
        
        # データ行
        for copy_result in all_results['copy_results']:
            if copy_result['status'] == 'success':
                # チャンクサイズ統計
                chunk_reads = copy_result.get('timing_details', {}).get('chunk_reads', [])
                avg_chunk_kb = 0
                if chunk_reads:
                    avg_chunk_kb = np.mean([c['size'] for c in chunk_reads]) / 1024
                
                # 対応するGPU結果を探す
                gpu_result = next(
                    (g for g in all_results['gpu_results'] 
                     if g['worker_id'] == copy_result['worker_id'] 
                     and g['chunk_idx'] == copy_result['chunk_idx']),
                    {}
                )
                
                writer.writerow([
                    copy_result['worker_id'],
                    copy_result['chunk_idx'],
                    copy_result['status'],
                    copy_result['data_size'] / (1024*1024),
                    copy_result['copy_time'],
                    copy_result.get('chunk_count', 0),
                    avg_chunk_kb,
                    copy_result.get('timing_details', {}).get('connection', 0),
                    copy_result.get('timing_details', {}).get('copy_init', 0),
                    gpu_result.get('gpu_transfer_time', 0),
                    gpu_result.get('gpu_processing_time', 0),
                    gpu_result.get('rows_processed', 0)
                ])
    
    print(f"✅ サマリーをCSV保存: {csv_path}")
    
    # チャンクサイズ分析
    analyze_chunk_sizes(all_results, output_dir, timestamp)


def analyze_chunk_sizes(all_results: Dict, output_dir: str, timestamp: str):
    """チャンクサイズの詳細分析"""
    chunk_sizes = []
    
    for copy_result in all_results['copy_results']:
        if copy_result['status'] == 'success':
            chunk_reads = copy_result.get('timing_details', {}).get('chunk_reads', [])
            chunk_sizes.extend([c['size'] for c in chunk_reads])
    
    if chunk_sizes:
        print(f"\n=== psycopg3チャンクサイズ分析 ===")
        print(f"総チャンク数: {len(chunk_sizes)}")
        print(f"平均サイズ: {np.mean(chunk_sizes)/1024:.1f} KB")
        print(f"中央値: {np.median(chunk_sizes)/1024:.1f} KB")
        print(f"最小: {min(chunk_sizes)/1024:.1f} KB")
        print(f"最大: {max(chunk_sizes)/1024:.1f} KB")
        print(f"標準偏差: {np.std(chunk_sizes)/1024:.1f} KB")
        
        # ヒストグラム出力
        hist_path = os.path.join(output_dir, f"chunk_size_histogram_{timestamp}.txt")
        with open(hist_path, 'w') as f:
            f.write("=== チャンクサイズ分布 ===\n")
            hist, bins = np.histogram(chunk_sizes, bins=20)
            for i in range(len(hist)):
                size_kb = bins[i] / 1024
                count = hist[i]
                f.write(f"{size_kb:6.1f}KB - {bins[i+1]/1024:6.1f}KB: {'#' * min(count, 50)} ({count})\n")
        
        print(f"✅ チャンクサイズ分布を保存: {hist_path}")


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


def run_ray_parallel_sequential_gpu_debug(
    limit_rows: int = 10000000,
    parallel_count: int = DEFAULT_PARALLEL,
    use_gpu_direct: bool = True,
    batch_size: int = 4,
    debug_mode: bool = True
):
    """Ray並列COPY + 順次GPU処理（デバッグ版）"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"=== PostgreSQL → GPU Ray並列 順次GPU処理版（デバッグ） ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}" if limit_rows else "行数制限: なし（全件処理）")
    print(f"並列数: {parallel_count}")
    print(f"チャンク数: {CHUNK_COUNT}")
    print(f"総タスク数: {parallel_count * CHUNK_COUNT}")
    print(f"バッチサイズ: {batch_size}")
    print(f"デバッグモード: {'ON' if debug_mode else 'OFF'}")
    print(f"処理方式:")
    print(f"  ① CPU: {parallel_count}並列でPostgreSQL COPY実行")
    print(f"  ② GPU: セミパイプライン処理（{batch_size}個のCOPY完了ごとにGPU処理開始）")
    
    # GPU Direct サポート確認
    gds_supported = check_gpu_direct_support() if use_gpu_direct else False
    
    # Ray初期化（GPU指定なし）
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
        for chunk_idx in range(CHUNK_COUNT):
            future = workers[worker_idx].copy_data_to_memory.remote(
                TABLE_NAME, start_block, end_block, chunk_idx, CHUNK_COUNT, 
                limit_rows, debug_mode=debug_mode
            )
            all_copy_futures.append(future)
    
    print(f"✅ {len(all_copy_futures)}個のCOPYタスクを投入")
    
    # セミパイプライン処理: COPYとGPUを並行実行
    copy_results = []
    gpu_results = []
    pending_gpu_tasks = []  # GPU処理待ちのタスク
    
    total_copy_time = 0
    total_data_size = 0
    total_gpu_transfer_time = 0
    total_gpu_processing_time = 0
    total_rows_processed = 0
    
    output_base = OUTPUT_PARQUET_PATH
    
    print(f"\n=== セミパイプライン処理開始 ===")
    print(f"COPYが{batch_size}個完了するごとにGPU処理を開始します")
    
    while all_copy_futures or pending_gpu_tasks:
        # COPYタスクがまだある場合
        if all_copy_futures:
            # 完了したCOPYタスクをバッチサイズまで収集
            num_to_wait = min(batch_size, len(all_copy_futures))
            ready_futures, remaining_futures = ray.wait(all_copy_futures, num_returns=num_to_wait, timeout=0.1)
            all_copy_futures = remaining_futures
            
            # COPY結果を処理
            for future in ready_futures:
                result = ray.get(future)
                copy_results.append(result)
                
                if result['status'] == 'success':
                    total_copy_time += result['copy_time']
                    total_data_size += result['data_size']
                    pending_gpu_tasks.append(result)
                    print(f"✅ COPY完了: Worker{result['worker_id']}-Chunk{result['chunk_idx']} "
                          f"({result['data_size']/(1024*1024):.1f}MB, "
                          f"{result.get('chunk_count', 0)}チャンク)")
                else:
                    print(f"❌ COPY失敗: Worker{result['worker_id']}-Chunk{result['chunk_idx']} - "
                          f"{result.get('error', 'Unknown')}")
        
        # GPU処理待ちタスクがバッチサイズに達した場合、またはCOPYが全て完了した場合
        if (len(pending_gpu_tasks) >= batch_size) or (not all_copy_futures and pending_gpu_tasks):
            # バッチサイズ分のタスクを取得
            tasks_to_process = pending_gpu_tasks[:batch_size] if len(pending_gpu_tasks) >= batch_size else pending_gpu_tasks
            pending_gpu_tasks = pending_gpu_tasks[len(tasks_to_process):]
            
            # バッチGPU処理を実行
            batch_results = process_batch_on_gpu(
                tasks_to_process,
                columns,
                gds_supported,
                output_base,
                debug_mode=debug_mode
            )
            
            # 結果を集計
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
    
    print(f"\n✅ 全処理完了")
    print(f"COPY: {len(successful_copies)}/{len(copy_results)} タスク成功")
    print(f"GPU: {len(successful_gpu)}/{len(gpu_results)} タスク成功")
    print(f"総データサイズ: {total_data_size / (1024*1024):.2f} MB")
    
    # 総合結果
    total_time = time.time() - start_total_time
    
    # デバッグ結果の保存
    if debug_mode:
        all_results = {
            'metadata': {
                'table_name': TABLE_NAME,
                'limit_rows': limit_rows,
                'parallel_count': parallel_count,
                'chunk_count': CHUNK_COUNT,
                'batch_size': batch_size,
                'total_blocks': total_blocks,
                'total_time': total_time,
                'timestamp': datetime.now().isoformat()
            },
            'copy_results': copy_results,
            'gpu_results': gpu_results,
            'summary': {
                'total_data_size_mb': total_data_size / (1024*1024),
                'total_copy_time': total_copy_time,
                'total_gpu_transfer_time': total_gpu_transfer_time,
                'total_gpu_processing_time': total_gpu_processing_time,
                'total_rows_processed': total_rows_processed
            }
        }
        save_debug_results(all_results)
    
    print(f"\n{'='*60}")
    print(f"=== Ray並列セミパイプラインGPU処理ベンチマーク完了 ===")
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
    
    # パフォーマンス指標
    if total_data_size > 0:
        overall_throughput = (total_data_size / (1024*1024)) / total_time
        print("\n--- パフォーマンス指標 ---")
        print(f"  全体スループット  : {overall_throughput:.2f} MB/sec")
    
    print("\n--- 処理方式の特徴 ---")
    print("  ✅ セミパイプライン: COPYとGPU処理を並行実行")
    print(f"  ✅ バッチサイズ: {batch_size}個のCOPY完了ごとにGPU処理開始")
    print("  ✅ メモリ効率: GPUメモリ競合を回避")
    print("  ✅ 確実な処理: 64個のParquetファイル生成")
    print("=========================================")
    
    # Ray終了
    ray.shutdown()
    print("\n✅ Ray終了")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU Ray並列 セミパイプライン処理版（デバッグ）')
    parser.add_argument('--rows', type=int, default=10000000, help='処理行数制限')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, help='並列数')
    parser.add_argument('--chunks', type=int, default=4, help='チャンク数')
    parser.add_argument('--batch-size', type=int, default=4, help='セミパイプラインのバッチサイズ')
    parser.add_argument('--no-limit', action='store_true', help='LIMIT無し（全件高速モード）')
    parser.add_argument('--check-support', action='store_true', help='GPU Directサポート確認のみ')
    parser.add_argument('--no-gpu-direct', action='store_true', help='GPU Direct無効化')
    parser.add_argument('--no-debug', action='store_true', help='デバッグモード無効化')
    
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
    run_ray_parallel_sequential_gpu_debug(
        limit_rows=final_limit_rows,
        parallel_count=args.parallel,
        use_gpu_direct=not args.no_gpu_direct,
        batch_size=args.batch_size,
        debug_mode=not args.no_debug
    )

if __name__ == "__main__":
    main()