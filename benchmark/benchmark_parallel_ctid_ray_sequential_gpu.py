"""
PostgreSQL → GPU Ray並列 順次GPU処理版
CPU並列でCOPY、GPU処理は順次実行して安定性を確保

最適化:
- 16並列でPostgreSQL COPYを実行
- GPU処理は1つずつ順次実行（メモリ競合回避）
- CuPy配列による安定したメモリ管理
- 64個の独立したParquetファイル出力

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
import cupy as cp
import argparse
from typing import List, Dict, Tuple, Optional

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_parallel_ctid_ray_sequential_gpu.output"

# 並列設定
DEFAULT_PARALLEL = 16
CHUNK_COUNT = 4  # 各ワーカーのctid範囲をチャンクに分割

@ray.remote
class PostgreSQLWorker:
    """Ray並列ワーカー: PostgreSQL COPY専用"""
    
    def __init__(self, worker_id: int, dsn: str):
        self.worker_id = worker_id
        self.dsn = dsn
        
    def copy_data_to_file(
        self, 
        table_name: str,
        start_block: int, 
        end_block: int,
        chunk_idx: int,
        total_chunks: int,
        limit_rows: Optional[int] = None,
        chunk_size: int = 16 * 1024 * 1024  # 16MB
    ) -> Dict[str, any]:
        """PostgreSQLからデータをCOPYしてファイルに保存"""
        
        # チャンクに応じてctid範囲を分割
        block_range = end_block - start_block
        chunk_block_size = block_range // total_chunks
        
        chunk_start_block = start_block + (chunk_idx * chunk_block_size)
        chunk_end_block = start_block + ((chunk_idx + 1) * chunk_block_size) if chunk_idx < total_chunks - 1 else end_block
        
        print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY開始 ctid範囲 ({chunk_start_block},{chunk_end_block})...")
        
        # 一時ファイル作成
        temp_file = os.path.join(
            tempfile.gettempdir(),
            f"ray_worker_{self.worker_id}_chunk{chunk_idx}_{chunk_start_block}_{chunk_end_block}.bin"
        )
        
        start_time = time.time()
        
        try:
            # PostgreSQL接続
            conn = psycopg.connect(self.dsn)
            data_size = 0
            
            try:
                # COPY SQL生成
                copy_sql = self._make_copy_sql(table_name, chunk_start_block, chunk_end_block, limit_rows)
                
                # データ取得
                with conn.cursor() as cur:
                    with cur.copy(copy_sql) as copy_obj:
                        with open(temp_file, 'wb') as f:
                            buffer = bytearray()
                            
                            for chunk in copy_obj:
                                if chunk:
                                    # memoryview → bytes変換
                                    if isinstance(chunk, memoryview):
                                        chunk_bytes = chunk.tobytes()
                                    else:
                                        chunk_bytes = bytes(chunk)
                                    
                                    buffer.extend(chunk_bytes)
                                    
                                    # チャンクサイズに達したら書き込み
                                    if len(buffer) >= chunk_size:
                                        f.write(buffer)
                                        data_size += len(buffer)
                                        buffer.clear()
                            
                            # 残りバッファを書き込み
                            if buffer:
                                f.write(buffer)
                                data_size += len(buffer)
                
                copy_time = time.time() - start_time
                print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY完了 "
                      f"({copy_time:.2f}秒, {data_size/(1024*1024):.1f}MB)")
                
                return {
                    'worker_id': self.worker_id,
                    'chunk_idx': chunk_idx,
                    'temp_file': temp_file,
                    'data_size': data_size,
                    'copy_time': copy_time,
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
                'copy_time': time.time() - start_time
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


def process_file_on_gpu(
    temp_file: str,
    data_size: int,
    columns: List,
    gds_supported: bool,
    output_path: str,
    worker_id: int,
    chunk_idx: int
) -> Dict[str, any]:
    """単一ファイルをGPUで処理（メインプロセスで実行）"""
    
    print(f"\nGPU処理開始: Worker {worker_id}-Chunk{chunk_idx}")
    timing_info = {
        'gpu_transfer_time': 0.0,
        'gpu_processing_time': 0.0,
        'rows_processed': 0
    }
    
    try:
        # GPU転送
        start_gpu_transfer_time = time.time()
        
        # CuPy配列として確保
        gpu_array = cp.zeros(data_size, dtype=cp.uint8)
        
        if gds_supported:
            # GPU Direct転送
            import kvikio
            from kvikio import CuFile
            
            with CuFile(temp_file, "r") as cufile:
                future = cufile.pread(gpu_array)
                future.get()
        else:
            # 通常転送
            with open(temp_file, 'rb') as f:
                file_data = f.read()
            gpu_array[:] = cp.frombuffer(file_data, dtype=cp.uint8)
        
        timing_info['gpu_transfer_time'] = time.time() - start_gpu_transfer_time
        print(f"  GPU転送完了 ({timing_info['gpu_transfer_time']:.2f}秒)")
        
        # 一時ファイル削除
        os.remove(temp_file)
        
        # GPU処理
        start_gpu_processing_time = time.time()
        
        # numba用の配列に変換
        raw_dev = cuda.as_cuda_array(gpu_array)
        header_sample = gpu_array[:min(128, len(gpu_array))].get()
        header_size = detect_pg_header_size(header_sample)
        
        # GPU最適化処理
        cudf_df, detailed_timing = postgresql_to_cudf_parquet(
            raw_dev=raw_dev,
            columns=columns,
            ncols=len(columns),
            header_size=header_size,
            output_path=output_path,
            compression='snappy',
            use_rmm=False,  # CuPyでメモリ管理
            optimize_gpu=True
        )
        
        timing_info['gpu_processing_time'] = time.time() - start_gpu_processing_time
        timing_info['rows_processed'] = len(cudf_df)
        
        print(f"  GPU処理完了 ({timing_info['gpu_processing_time']:.2f}秒, {timing_info['rows_processed']:,}行)")
        
        # メモリ解放
        del gpu_array
        del raw_dev
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cuda.synchronize()
        
        timing_info['status'] = 'success'
        return timing_info
        
    except Exception as e:
        print(f"  ❌GPU処理失敗: {e}")
        timing_info['status'] = 'error'
        timing_info['error'] = str(e)
        
        # エラー時もメモリ解放を試みる
        try:
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cuda.synchronize()
        except:
            pass
            
        return timing_info


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
    use_gpu_direct: bool = True
):
    """Ray並列COPY + 順次GPU処理"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"=== PostgreSQL → GPU Ray並列 順次GPU処理版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}" if limit_rows else "行数制限: なし（全件処理）")
    print(f"並列数: {parallel_count}")
    print(f"チャンク数: {CHUNK_COUNT}")
    print(f"総タスク数: {parallel_count * CHUNK_COUNT}")
    print(f"処理方式:")
    print(f"  ① CPU: {parallel_count}並列でPostgreSQL COPY実行")
    print(f"  ② GPU: 順次処理（メモリ競合回避）")
    
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
            future = workers[worker_idx].copy_data_to_file.remote(
                TABLE_NAME, start_block, end_block, chunk_idx, CHUNK_COUNT, limit_rows
            )
            all_copy_futures.append(future)
    
    print(f"✅ {len(all_copy_futures)}個のCOPYタスクを投入")
    
    # COPY結果を収集
    copy_results = []
    total_copy_time = 0
    total_data_size = 0
    
    while all_copy_futures:
        # 完了したタスクを1つずつ処理
        ready_futures, remaining_futures = ray.wait(all_copy_futures, num_returns=1)
        all_copy_futures = remaining_futures
        
        for future in ready_futures:
            result = ray.get(future)
            copy_results.append(result)
            
            if result['status'] == 'success':
                total_copy_time += result['copy_time']
                total_data_size += result['data_size']
            else:
                print(f"❌ Worker {result['worker_id']}-Chunk{result['chunk_idx']} COPY失敗: {result.get('error', 'Unknown')}")
    
    # 成功したCOPY結果のみ抽出
    successful_copies = [r for r in copy_results if r['status'] == 'success']
    print(f"\n✅ COPY完了: {len(successful_copies)}/{len(copy_results)} タスク成功")
    print(f"総データサイズ: {total_data_size / (1024*1024):.2f} MB")
    
    # GPU処理を順次実行
    print(f"\n=== フェーズ2: GPU順次処理 ===")
    gpu_results = []
    total_gpu_transfer_time = 0
    total_gpu_processing_time = 0
    total_rows_processed = 0
    
    output_base = OUTPUT_PARQUET_PATH
    
    for i, copy_result in enumerate(successful_copies):
        if 'temp_file' not in copy_result:
            continue
            
        output_path = f"{output_base}_worker{copy_result['worker_id']}_chunk{copy_result['chunk_idx']}.parquet"
        
        # GPU処理（メインプロセスで実行）
        gpu_result = process_file_on_gpu(
            copy_result['temp_file'],
            copy_result['data_size'],
            columns,
            gds_supported,
            output_path,
            copy_result['worker_id'],
            copy_result['chunk_idx']
        )
        
        gpu_results.append(gpu_result)
        
        if gpu_result['status'] == 'success':
            total_gpu_transfer_time += gpu_result['gpu_transfer_time']
            total_gpu_processing_time += gpu_result['gpu_processing_time']
            total_rows_processed += gpu_result['rows_processed']
        
        print(f"進捗: {i+1}/{len(successful_copies)} 完了")
    
    # 成功したGPU処理の数
    successful_gpu = [r for r in gpu_results if r['status'] == 'success']
    
    # 総合結果
    total_time = time.time() - start_total_time
    
    print(f"\n{'='*60}")
    print(f"=== Ray並列順次GPU処理ベンチマーク完了 ===")
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
    print("  ✅ 安定性優先: GPU処理を順次実行")
    print("  ✅ メモリ効率: GPUメモリ競合を回避")
    print("  ✅ 確実な処理: 64個のParquetファイル生成")
    print("=========================================")
    
    # Ray終了
    ray.shutdown()
    print("\n✅ Ray終了")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU Ray並列 順次GPU処理版')
    parser.add_argument('--rows', type=int, default=10000000, help='処理行数制限')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, help='並列数')
    parser.add_argument('--chunks', type=int, default=4, help='チャンク数')
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
        use_gpu_direct=not args.no_gpu_direct
    )

if __name__ == "__main__":
    main()