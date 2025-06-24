"""
PostgreSQL → GPU Ray並列 ctid分割版（チャンク処理版）
Ray分散処理フレームワークによる真の並列化実装
GPUメモリ制限に対応した2段階チャンク処理

最適化:
- Rayによる真のマルチプロセス並列処理（GIL回避）
- ctidによるテーブル分割並列読み取り
- 各ワーカーが独立したPostgreSQL接続
- GPUメモリに収まるよう2回に分けて処理
- 各回で範囲の半分を処理

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import math
import tempfile
import psycopg
import ray
import rmm
import numpy as np
from numba import cuda
import cupy as cp
import argparse
from typing import List, Dict, Tuple, Optional

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_parallel_ctid_ray_chunked.output.parquet"

# 並列設定
DEFAULT_PARALLEL = 16
CHUNK_COUNT = 2  # 各ワーカーのctid範囲を2つのチャンクに分割

@ray.remote
class PostgreSQLWorker:
    """Ray並列ワーカー: 独立したPostgreSQL接続を持つ"""
    
    def __init__(self, worker_id: int, dsn: str):
        self.worker_id = worker_id
        self.dsn = dsn
        self.temp_files = []
        
    def process_ctid_range_chunked(
        self, 
        table_name: str,
        start_block: int, 
        end_block: int,
        chunk_idx: int,
        total_chunks: int,
        limit_rows: Optional[int] = None,
        chunk_size: int = 16 * 1024 * 1024  # 16MB
    ) -> Dict[str, any]:
        """ctid範囲の一部（チャンク）を処理"""
        
        # チャンクに応じてctid範囲を分割
        block_range = end_block - start_block
        chunk_block_size = block_range // total_chunks
        
        chunk_start_block = start_block + (chunk_idx * chunk_block_size)
        chunk_end_block = start_block + ((chunk_idx + 1) * chunk_block_size) if chunk_idx < total_chunks - 1 else end_block
        
        print(f"Worker {self.worker_id}: チャンク{chunk_idx+1}/{total_chunks} ctid範囲 ({chunk_start_block},{chunk_end_block}) 開始...")
        
        # 一時ファイル作成
        temp_file = os.path.join(
            tempfile.gettempdir(),
            f"ray_ctid_worker_{self.worker_id}_chunk{chunk_idx}_{chunk_start_block}_{chunk_end_block}.bin"
        )
        self.temp_files.append(temp_file)
        
        start_time = time.time()
        data_size = 0
        read_count = 0
        
        try:
            # PostgreSQL接続
            conn = psycopg.connect(self.dsn)
            
            # COPY SQL生成
            copy_sql = self._make_copy_sql(table_name, chunk_start_block, chunk_end_block, limit_rows)
            
            # 大容量バッファでデータ取得
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
                                    read_count += 1
                                    
                                    # 進捗（1GB単位）
                                    if data_size % (1024*1024*1024) < len(buffer):
                                        print(f"Worker {self.worker_id}-C{chunk_idx}: {data_size/(1024*1024*1024):.1f}GB処理")
                                    
                                    buffer.clear()
                        
                        # 残りバッファを書き込み
                        if buffer:
                            f.write(buffer)
                            data_size += len(buffer)
                            read_count += 1
            
            conn.close()
            
            duration = time.time() - start_time
            speed = data_size / (1024*1024) / duration if duration > 0 else 0
            
            print(f"Worker {self.worker_id}-C{chunk_idx}: ✅完了 ({duration:.2f}秒, {data_size/(1024*1024):.1f}MB, {speed:.1f}MB/s)")
            
            return {
                'worker_id': self.worker_id,
                'chunk_idx': chunk_idx,
                'temp_file': temp_file,
                'data_size': data_size,
                'duration': duration,
                'read_count': read_count,
                'speed': speed
            }
            
        except Exception as e:
            print(f"Worker {self.worker_id}-C{chunk_idx}: ❌エラー - {e}")
            return {
                'worker_id': self.worker_id,
                'chunk_idx': chunk_idx,
                'error': str(e),
                'data_size': 0
            }
    
    def _make_copy_sql(self, table_name: str, start_block: int, end_block: int, limit_rows: Optional[int]) -> str:
        """COPY SQL生成（pg2arrow互換）"""
        sql = f"""
        COPY (
            SELECT * FROM {table_name}
            WHERE ctid >= '({start_block},1)'::tid
              AND ctid < '({end_block+1},1)'::tid
        """
        
        if limit_rows:
            # 並列数とチャンク数を考慮した行数制限
            sql += f" LIMIT {limit_rows // (DEFAULT_PARALLEL * CHUNK_COUNT)}"
        
        sql += ") TO STDOUT (FORMAT binary)"
        return sql
    
    def get_temp_files(self) -> List[str]:
        """全ての一時ファイルパスを返す"""
        return self.temp_files

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

def process_chunk_data(
    chunk_results: List[Dict],
    columns: List,
    gds_supported: bool,
    output_base_path: str,
    chunk_num: int
) -> Tuple[int, float, float]:
    """チャンクデータの処理（GPU転送→処理→Parquet出力）
    
    Returns:
        (rows_processed, gpu_processing_time, gpu_transfer_time)
    """
    
    print(f"\n=== チャンク {chunk_num} 処理開始 ===")
    
    # チャンクの総データサイズ計算
    total_data_size = sum(r['data_size'] for r in chunk_results if 'error' not in r)
    if total_data_size == 0:
        print(f"チャンク {chunk_num}: データなし")
        return 0, 0.0, 0.0
    
    print(f"チャンク {chunk_num} データサイズ: {total_data_size / (1024*1024):.2f} MB")
    
    # メモリ使用状況を表示
    mempool = cp.get_default_memory_pool()
    print(f"GPU メモリ使用状況: {mempool.used_bytes() / (1024**3):.2f} GB / {mempool.total_bytes() / (1024**3):.2f} GB")
    
    # GPU転送
    start_gpu_time = time.time()
    gpu_buffer = rmm.DeviceBuffer(size=total_data_size)
    print(f"GPUバッファ確保完了: {total_data_size / (1024*1024):.2f} MB")
    
    current_offset = 0
    gpu_transferred = 0
    
    for result in chunk_results:
        if 'error' in result:
            continue
            
        temp_file = result['temp_file']
        file_size = result['data_size']
        
        try:
            if gds_supported:
                # GPU Direct転送
                import kvikio
                from kvikio import CuFile
                
                with CuFile(temp_file, "r") as cufile:
                    temp_buffer = rmm.DeviceBuffer(size=file_size)
                    future = cufile.pread(temp_buffer)
                    bytes_read = future.get()
                    
                    # GPU to GPU コピー
                    cp.cuda.runtime.memcpy(
                        gpu_buffer.ptr + current_offset,
                        temp_buffer.ptr,
                        file_size,
                        cp.cuda.runtime.memcpyDeviceToDevice
                    )
                    gpu_transferred += bytes_read
            else:
                # 通常転送
                with open(temp_file, 'rb') as f:
                    file_data = f.read()
                
                # Host to GPU コピー
                cp.cuda.runtime.memcpy(
                    gpu_buffer.ptr + current_offset,
                    np.frombuffer(file_data, dtype=np.uint8).ctypes.data,
                    file_size,
                    cp.cuda.runtime.memcpyHostToDevice
                )
                gpu_transferred += file_size
            
            current_offset += file_size
            
            # 一時ファイル削除
            os.remove(temp_file)
            
        except Exception as e:
            print(f"  Worker {result['worker_id']}: GPU転送エラー - {e}")
    
    gpu_time = time.time() - start_gpu_time
    gpu_speed = gpu_transferred / (1024*1024) / gpu_time if gpu_time > 0 else 0
    print(f"✅ GPU転送完了 ({gpu_time:.4f}秒, {gpu_speed:.2f} MB/sec)")
    
    # GPU処理
    raw_dev = cuda.as_cuda_array(gpu_buffer).view(dtype=np.uint8)
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    
    # 出力パス
    output_path = f"{output_base_path}_chunk{chunk_num}.parquet"
    
    # GPU最適化処理
    print(f"GPU最適化処理中...")
    start_processing_time = time.time()
    
    try:
        cudf_df, detailed_timing = postgresql_to_cudf_parquet(
            raw_dev=raw_dev,
            columns=columns,
            ncols=len(columns),
            header_size=header_size,
            output_path=output_path,
            compression='snappy',
            use_rmm=True,
            optimize_gpu=True
        )
        
        processing_time = time.time() - start_processing_time
        rows = len(cudf_df)
        
        print(f"✅ チャンク {chunk_num} 処理完了: {rows:,} 行")
        
        # メモリ解放（重要！）
        del gpu_buffer
        del raw_dev
        cp.get_default_memory_pool().free_all_blocks()
        
        # RMMのフラッシュ（オプション）
        try:
            rmm.mr.get_current_device_resource().flush()
        except:
            pass
        
        print(f"GPU メモリ解放完了")
        
        return rows, processing_time, gpu_time
        
    except Exception as e:
        print(f"❌ チャンク {chunk_num} GPU処理エラー: {e}")
        # エラー時もメモリ解放を試みる
        try:
            if 'gpu_buffer' in locals():
                del gpu_buffer
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass
        return 0, 0.0, gpu_time if 'gpu_time' in locals() else 0.0

def run_ray_parallel_benchmark_chunked(
    limit_rows: int = 10000000,
    parallel_count: int = DEFAULT_PARALLEL,
    use_gpu_direct: bool = True
):
    """Ray並列ctid分割ベンチマーク実行（チャンク処理版）"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"=== PostgreSQL → GPU Ray並列 ctid分割版（チャンク処理） ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}" if limit_rows else "行数制限: なし（全件処理）")
    print(f"並列数: {parallel_count}")
    print(f"チャンク数: {CHUNK_COUNT} (各ワーカーがctid範囲を{CHUNK_COUNT}分割)")
    print(f"最適化設定:")
    print(f"  Ray分散処理: 真のマルチプロセス並列")
    print(f"  ctid分割: {parallel_count}並列 × {CHUNK_COUNT}チャンク")
    
    # GPU Direct サポート確認
    gds_supported = check_gpu_direct_support() if use_gpu_direct else False
    
    # Ray初期化
    print("\n=== Ray初期化 ===")
    if not ray.is_initialized():
        ray.init(num_cpus=parallel_count)
        print(f"✅ Ray初期化完了 (CPUs: {parallel_count})")
    
    # RMM初期化（強制的に22GBメモリプールに再初期化）
    try:
        # 既存のRMMを一旦リセット
        if rmm.is_initialized():
            print("既存のRMMをリセット中...")
            try:
                # 既存のプールをクリア
                rmm.mr.get_current_device_resource().flush()
            except:
                pass
        
        # 22GBで再初期化
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=22*1024**3,  # 22GB
            maximum_pool_size=22*1024**3   # 22GB（最大も同じに設定）
        )
        print("✅ RMM メモリプール初期化完了 (22GB)")
        
    except Exception as e:
        print(f"❌ RMM初期化エラー: {e}")
        return
    
    start_total_time = time.time()
    
    # メタデータ取得
    print("\nメタデータを取得中...")
    start_meta_time = time.time()
    conn = psycopg.connect(dsn)
    try:
        columns = fetch_column_meta(conn, f"SELECT * FROM {TABLE_NAME}")
        meta_time = time.time() - start_meta_time
        print(f"メタデータ取得完了 ({meta_time:.4f}秒)")
        ncols = len(columns)
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
    
    total_rows_processed = 0
    total_copy_time = 0
    total_processing_time = 0
    total_gpu_transfer_time = 0
    total_data_size = 0
    
    # GPUメモリサイズを取得（CuPy経由）
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        print(f"\nGPU メモリプール状況:")
        print(f"  CuPy メモリプール: {mempool.used_bytes() / (1024**3):.2f} GB 使用中")
        print(f"  Pinned メモリプール: {pinned_mempool.used_bytes() / (1024**3):.2f} GB 使用中")
    except Exception as e:
        print(f"\nGPU メモリ情報取得エラー: {e}")
    
    # チャンクごとに処理
    for chunk_idx in range(CHUNK_COUNT):
        print(f"\n{'='*60}")
        print(f"チャンク {chunk_idx + 1}/{CHUNK_COUNT} 開始")
        print(f"{'='*60}")
        
        # チャンク処理前のメモリチェックとクリア
        mempool = cp.get_default_memory_pool()
        used_memory = mempool.used_bytes() / (1024**3)
        if used_memory > 1.0:  # 1GB以上使用している場合
            print(f"⚠️  メモリ使用量が高い ({used_memory:.2f} GB) - クリア実行")
            cp.get_default_memory_pool().free_all_blocks()
            cuda.synchronize()
        
        # 並列COPY処理
        print(f"\n{parallel_count}並列Ray処理開始（チャンク{chunk_idx + 1}）...")
        futures = []
        
        for i, (start_block, end_block) in enumerate(ranges):
            future = workers[i].process_ctid_range_chunked.remote(
                TABLE_NAME, start_block, end_block, chunk_idx, CHUNK_COUNT, limit_rows
            )
            futures.append(future)
        
        # 並列実行
        start_copy_time = time.time()
        print("⏳ Ray並列処理実行中...")
        results = ray.get(futures)
        copy_time = time.time() - start_copy_time
        total_copy_time += copy_time
        
        # 結果集計
        successful_results = [r for r in results if 'error' not in r]
        failed_count = len(results) - len(successful_results)
        
        if not successful_results:
            print(f"❌ チャンク {chunk_idx + 1}: すべてのワーカーが失敗")
            continue
        
        chunk_data_size = sum(r['data_size'] for r in successful_results)
        chunk_read_count = sum(r['read_count'] for r in successful_results)
        avg_speed = sum(r['speed'] for r in successful_results) / len(successful_results)
        total_data_size += chunk_data_size
        
        print(f"\n✅ チャンク {chunk_idx + 1} COPY完了 ({copy_time:.4f}秒)")
        print(f"  成功ワーカー数: {len(successful_results)}/{parallel_count}")
        print(f"  チャンクデータサイズ: {chunk_data_size / (1024*1024):.2f} MB")
        print(f"  並列転送速度: {chunk_data_size / (1024*1024) / copy_time:.2f} MB/sec")
        
        # GPU処理
        output_base = OUTPUT_PARQUET_PATH.replace('.parquet', '')
        rows_processed, processing_time, gpu_transfer_time = process_chunk_data(
            successful_results, columns, gds_supported, output_base, chunk_idx + 1
        )
        
        total_rows_processed += rows_processed
        total_processing_time += processing_time
        total_gpu_transfer_time += gpu_transfer_time
        
        # チャンク処理後のメモリ状況表示
        mempool = cp.get_default_memory_pool()
        print(f"チャンク {chunk_idx + 1} 後のメモリ使用: {mempool.used_bytes() / (1024**3):.2f} GB")
    
    # 総合結果
    total_time = time.time() - start_total_time
    
    print(f"\n{'='*60}")
    print(f"=== Ray並列ctid分割版（チャンク処理）ベンチマーク完了 ===")
    print(f"{'='*60}")
    print(f"総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得       : {meta_time:.4f} 秒")
    print(f"  {parallel_count}並列Ray処理（合計）: {total_copy_time:.4f} 秒")
    print(f"  GPU転送（合計）      : {total_gpu_transfer_time:.4f} 秒")
    print(f"  GPU処理（合計）      : {total_processing_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数（合計）    : {total_rows_processed:,} 行")
    print(f"  処理列数           : {len(columns)} 列")
    print(f"  並列数             : {parallel_count}")
    print(f"  チャンク数         : {CHUNK_COUNT}")
    print(f"  総データサイズ      : {total_data_size / (1024*1024):.2f} MB")
    
    # 性能評価
    baseline_speed = 155.6  # 単一接続の実測値
    
    # 全体スループット計算
    if total_data_size > 0 and total_copy_time > 0:
        # 実際のデータ処理にかかった時間（メタデータ取得を除く）
        data_processing_time = total_copy_time + total_gpu_transfer_time + total_processing_time
        overall_throughput = (total_data_size / (1024*1024)) / data_processing_time if data_processing_time > 0 else 0
        
        print("\n--- パフォーマンス指標 ---")
        print(f"  全体スループット    : {overall_throughput:.2f} MB/sec")
        print(f"    (COPY + GPU転送 + GPU処理の合計時間で計算)")
        
        # PostgreSQL COPY速度
        copy_speed = (total_data_size / (1024*1024)) / total_copy_time
        print(f"  PostgreSQL COPY速度 : {copy_speed:.2f} MB/sec")
        improvement_ratio = copy_speed / baseline_speed
        print(f"  COPY性能向上倍率    : {improvement_ratio:.1f}倍 (対 {baseline_speed} MB/sec)")
        
        # GPU転送速度
        if total_gpu_transfer_time > 0:
            gpu_transfer_speed = (total_data_size / (1024*1024)) / total_gpu_transfer_time
            print(f"  GPU転送速度        : {gpu_transfer_speed:.2f} MB/sec")
        
        # GPU処理速度
        if total_processing_time > 0:
            gpu_processing_speed = (total_data_size / (1024*1024)) / total_processing_time
            print(f"  GPU処理速度        : {gpu_processing_speed:.2f} MB/sec")
    
    print("--- Ray並列チャンク処理最適化効果 ---")
    print("  ✅ Ray分散処理: 真のマルチプロセス並列")
    print("  ✅ ctid分割: テーブル並列読み取り")
    print("  ✅ チャンク処理: GPUメモリ制限対応")
    print("  ✅ 大規模データ: 分割処理で対応可能")
    print("=========================================")
    
    # Ray終了
    ray.shutdown()
    print("\n✅ Ray終了")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU Ray並列 ctid分割版（チャンク処理）')
    parser.add_argument('--rows', type=int, default=10000000, help='処理行数制限')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, help='並列数')
    parser.add_argument('--chunks', type=int, default=2, help='チャンク数')
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
    run_ray_parallel_benchmark_chunked(
        limit_rows=final_limit_rows,
        parallel_count=args.parallel,
        use_gpu_direct=not args.no_gpu_direct
    )

if __name__ == "__main__":
    main()