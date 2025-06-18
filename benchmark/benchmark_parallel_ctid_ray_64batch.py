"""
PostgreSQL → GPU Ray並列 64バッチ処理版
16ワーカー × 4チャンク = 64個の独立したタスクで真の並列化を実現

最適化:
- 64個の独立したRayタスクによる完全並列処理
- チャンク単位の同期待ちを排除
- 各タスクが独立してCOPY → GPU転送 → GPU処理 → Parquet出力
- CuPy配列による安定したメモリ管理

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
OUTPUT_PARQUET_PATH = "benchmark/lineorder_parallel_ctid_ray_64batch.output"

# 並列設定
DEFAULT_PARALLEL = 16
CHUNK_COUNT = 4  # 各ワーカーのctid範囲をチャンクに分割

@ray.remote(num_gpus=0.1)  # GPU割り当てを指定
class PostgreSQLWorker:
    """Ray並列ワーカー: 独立したPostgreSQL接続を持つ"""
    
    def __init__(self, worker_id: int, dsn: str):
        self.worker_id = worker_id
        self.dsn = dsn
        
    def process_and_convert_to_parquet(
        self, 
        table_name: str,
        start_block: int, 
        end_block: int,
        chunk_idx: int,
        total_chunks: int,
        columns: List,
        gds_supported: bool,
        output_base_path: str,
        limit_rows: Optional[int] = None,
        chunk_size: int = 16 * 1024 * 1024  # 16MB
    ) -> Dict[str, any]:
        """ctid範囲の一部を処理し、直接Parquetに変換"""
        
        # チャンクに応じてctid範囲を分割
        block_range = end_block - start_block
        chunk_block_size = block_range // total_chunks
        
        chunk_start_block = start_block + (chunk_idx * chunk_block_size)
        chunk_end_block = start_block + ((chunk_idx + 1) * chunk_block_size) if chunk_idx < total_chunks - 1 else end_block
        
        print(f"Worker {self.worker_id}-Chunk{chunk_idx}: ctid範囲 ({chunk_start_block},{chunk_end_block}) 開始...")
        
        # 一時ファイル作成
        temp_file = os.path.join(
            tempfile.gettempdir(),
            f"ray_worker_{self.worker_id}_chunk{chunk_idx}_{chunk_start_block}_{chunk_end_block}.bin"
        )
        
        # タイミング記録用
        timing_info = {
            'worker_id': self.worker_id,
            'chunk_idx': chunk_idx,
            'copy_time': 0.0,
            'gpu_transfer_time': 0.0,
            'gpu_processing_time': 0.0,
            'total_time': 0.0,
            'data_size': 0,
            'rows_processed': 0
        }
        
        start_total_time = time.time()
        
        try:
            # ========== STEP 1: PostgreSQL COPY ==========
            start_copy_time = time.time()
            data_size = self._copy_data_from_postgres(
                table_name, chunk_start_block, chunk_end_block, 
                temp_file, limit_rows, chunk_size
            )
            timing_info['copy_time'] = time.time() - start_copy_time
            timing_info['data_size'] = data_size
            
            if data_size == 0:
                print(f"Worker {self.worker_id}-Chunk{chunk_idx}: データなし")
                return timing_info
            
            print(f"Worker {self.worker_id}-Chunk{chunk_idx}: COPY完了 "
                  f"({timing_info['copy_time']:.2f}秒, {data_size/(1024*1024):.1f}MB)")
            
            # ========== STEP 2: GPU転送 ==========
            start_gpu_transfer_time = time.time()
            gpu_array = self._transfer_to_gpu(temp_file, data_size, gds_supported)
            timing_info['gpu_transfer_time'] = time.time() - start_gpu_transfer_time
            
            print(f"Worker {self.worker_id}-Chunk{chunk_idx}: GPU転送完了 "
                  f"({timing_info['gpu_transfer_time']:.2f}秒)")
            
            # 一時ファイル削除
            os.remove(temp_file)
            
            # ========== STEP 3: GPU処理とParquet出力 ==========
            start_gpu_processing_time = time.time()
            output_path = f"{output_base_path}_worker{self.worker_id}_chunk{chunk_idx}.parquet"
            
            rows_processed = self._process_on_gpu_and_save(
                gpu_array, columns, output_path
            )
            timing_info['gpu_processing_time'] = time.time() - start_gpu_processing_time
            timing_info['rows_processed'] = rows_processed
            
            print(f"Worker {self.worker_id}-Chunk{chunk_idx}: GPU処理完了 "
                  f"({timing_info['gpu_processing_time']:.2f}秒, {rows_processed:,}行)")
            
            # メモリ解放
            del gpu_array
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
            timing_info['total_time'] = time.time() - start_total_time
            timing_info['status'] = 'success'
            
            return timing_info
            
        except Exception as e:
            print(f"Worker {self.worker_id}-Chunk{chunk_idx}: ❌エラー - {e}")
            timing_info['total_time'] = time.time() - start_total_time
            timing_info['status'] = 'error'
            timing_info['error'] = str(e)
            return timing_info
    
    def _copy_data_from_postgres(
        self, 
        table_name: str,
        start_block: int,
        end_block: int,
        temp_file: str,
        limit_rows: Optional[int],
        chunk_size: int
    ) -> int:
        """PostgreSQLからデータをCOPY"""
        
        # PostgreSQL接続
        conn = psycopg.connect(self.dsn)
        data_size = 0
        
        try:
            # COPY SQL生成
            copy_sql = self._make_copy_sql(table_name, start_block, end_block, limit_rows)
            
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
            
            return data_size
            
        finally:
            conn.close()
    
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
    
    def _transfer_to_gpu(self, temp_file: str, file_size: int, gds_supported: bool) -> cp.ndarray:
        """ファイルをGPUに転送"""
        
        # CuPy配列として確保
        gpu_array = cp.zeros(file_size, dtype=cp.uint8)
        
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
        
        return gpu_array
    
    def _process_on_gpu_and_save(
        self, 
        gpu_array: cp.ndarray,
        columns: List,
        output_path: str
    ) -> int:
        """GPU処理とParquet保存"""
        
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
        
        return len(cudf_df)

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

def run_ray_parallel_64batch(
    limit_rows: int = 10000000,
    parallel_count: int = DEFAULT_PARALLEL,
    use_gpu_direct: bool = True
):
    """Ray並列64バッチ処理実行"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"=== PostgreSQL → GPU Ray並列 64バッチ処理版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}" if limit_rows else "行数制限: なし（全件処理）")
    print(f"並列数: {parallel_count}")
    print(f"チャンク数: {CHUNK_COUNT}")
    print(f"総タスク数: {parallel_count * CHUNK_COUNT}")
    print(f"最適化設定:")
    print(f"  64個の独立タスク: チャンク単位の同期なし")
    print(f"  CuPy配列: 安定したメモリ管理")
    print(f"  完全並列化: COPY/GPU転送/GPU処理が独立実行")
    
    # GPU Direct サポート確認
    gds_supported = check_gpu_direct_support() if use_gpu_direct else False
    
    # Ray初期化
    print("\n=== Ray初期化 ===")
    if not ray.is_initialized():
        ray.init(num_cpus=parallel_count * 2, num_gpus=1)  # GPU利用を明示
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
    
    # 16ワーカーずつタスクを投入し、完了したワーカーに次のチャンクを割り当てる
    print(f"\n=== ワーカープールによる処理開始 ===")
    print(f"ワーカー数: {parallel_count}")
    print(f"総チャンク数: {parallel_count * CHUNK_COUNT}")
    
    output_base = OUTPUT_PARQUET_PATH
    all_results = []
    
    # 各ワーカーに最初のチャンク（chunk_idx=0）を割り当て
    active_tasks = {}  # future -> (worker_idx, chunk_idx)
    
    for i, (start_block, end_block) in enumerate(ranges):
        future = workers[i].process_and_convert_to_parquet.remote(
            TABLE_NAME, start_block, end_block, 0, CHUNK_COUNT,
            columns, gds_supported, output_base, limit_rows
        )
        active_tasks[future] = (i, 0)
    
    print(f"✅ 初期タスク {len(active_tasks)}個を投入")
    
    # 各ワーカーの次のチャンクインデックスを追跡
    next_chunk_idx = {i: 1 for i in range(len(ranges))}
    
    # アクティブなタスクがある限り処理を続ける
    while active_tasks:
        # 完了したタスクを待つ（最低1つ完了するまで待機）
        ready_futures, remaining_futures = ray.wait(
            list(active_tasks.keys()), 
            num_returns=1
        )
        
        # 完了したタスクを処理
        for future in ready_futures:
            worker_idx, chunk_idx = active_tasks.pop(future)
            result = ray.get(future)
            all_results.append(result)
            
            if result.get('status') == 'success':
                print(f"✅ Worker {worker_idx}-Chunk{chunk_idx} 完了 "
                      f"({result['total_time']:.2f}秒, {result['data_size']/(1024*1024):.1f}MB)")
            else:
                print(f"❌ Worker {worker_idx}-Chunk{chunk_idx} 失敗: {result.get('error', 'Unknown error')}")
            
            # このワーカーに次のチャンクがあれば割り当て
            if next_chunk_idx[worker_idx] < CHUNK_COUNT:
                start_block, end_block = ranges[worker_idx]
                new_chunk_idx = next_chunk_idx[worker_idx]
                
                future = workers[worker_idx].process_and_convert_to_parquet.remote(
                    TABLE_NAME, start_block, end_block, new_chunk_idx, CHUNK_COUNT,
                    columns, gds_supported, output_base, limit_rows
                )
                active_tasks[future] = (worker_idx, new_chunk_idx)
                next_chunk_idx[worker_idx] += 1
                
                print(f"→ Worker {worker_idx} に Chunk{new_chunk_idx} を割り当て")
    
    results = all_results
    
    # 結果集計
    successful_results = [r for r in results if r.get('status') == 'success']
    failed_results = [r for r in results if r.get('status') == 'error']
    
    print(f"\n=== 処理完了 ===")
    print(f"成功タスク: {len(successful_results)}/{len(results)}")
    print(f"失敗タスク: {len(failed_results)}/{len(results)}")
    
    if failed_results:
        print("\n失敗したタスク:")
        for r in failed_results[:5]:  # 最初の5個まで表示
            print(f"  Worker {r['worker_id']}-Chunk{r['chunk_idx']}: {r.get('error', 'Unknown error')}")
    
    # 統計情報集計
    total_copy_time = sum(r['copy_time'] for r in successful_results)
    total_gpu_transfer_time = sum(r['gpu_transfer_time'] for r in successful_results)
    total_gpu_processing_time = sum(r['gpu_processing_time'] for r in successful_results)
    total_data_size = sum(r['data_size'] for r in successful_results)
    total_rows_processed = sum(r['rows_processed'] for r in successful_results)
    
    # 並列実行の実効時間計算
    max_total_time = max(r['total_time'] for r in successful_results) if successful_results else 0
    
    total_time = time.time() - start_total_time
    
    print(f"\n{'='*60}")
    print(f"=== Ray並列64バッチ処理ベンチマーク完了 ===")
    print(f"{'='*60}")
    print(f"総時間 = {total_time:.4f} 秒")
    print(f"実効時間（最長タスク） = {max_total_time:.4f} 秒")
    print("--- 累積時間（全タスクの合計） ---")
    print(f"  PostgreSQL COPY   : {total_copy_time:.4f} 秒")
    print(f"  GPU転送          : {total_gpu_transfer_time:.4f} 秒")
    print(f"  GPU処理          : {total_gpu_processing_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数（合計）  : {total_rows_processed:,} 行")
    print(f"  処理列数         : {len(columns)} 列")
    print(f"  総データサイズ    : {total_data_size / (1024*1024):.2f} MB")
    print(f"  並列数           : {parallel_count}")
    print(f"  チャンク数       : {CHUNK_COUNT}")
    print(f"  総タスク数       : {parallel_count * CHUNK_COUNT}")
    
    # パフォーマンス指標
    if total_data_size > 0 and max_total_time > 0:
        effective_throughput = (total_data_size / (1024*1024)) / max_total_time
        print("\n--- パフォーマンス指標 ---")
        print(f"  実効スループット  : {effective_throughput:.2f} MB/sec")
        print(f"    (最長タスクの実行時間ベース)")
        
        # 並列化効率
        sequential_time = total_copy_time / len(successful_results) + \
                         total_gpu_transfer_time / len(successful_results) + \
                         total_gpu_processing_time / len(successful_results)
        parallelization_efficiency = (sequential_time / max_total_time) * 100
        print(f"  並列化効率       : {parallelization_efficiency:.1f}%")
    
    print("\n--- 64バッチ処理の効果 ---")
    print("  ✅ 完全並列化: チャンク単位の同期待ちを排除")
    print("  ✅ 最適なリソース活用: CPU/GPU/IOが独立動作")
    print("  ✅ スケーラビリティ: タスク数で性能調整可能")
    print("=========================================")
    
    # Ray終了
    ray.shutdown()
    print("\n✅ Ray終了")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU Ray並列 64バッチ処理版')
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
    run_ray_parallel_64batch(
        limit_rows=final_limit_rows,
        parallel_count=args.parallel,
        use_gpu_direct=not args.no_gpu_direct
    )

if __name__ == "__main__":
    main()