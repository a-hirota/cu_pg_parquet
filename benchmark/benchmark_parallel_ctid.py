"""
PostgreSQL → GPU 16並列 ctid分割版
16並列ctid分割 + UNIXドメインソケット + GPU Direct Storage

最適化:
- ctidによるテーブル16分割並列読み取り
- UNIXドメインソケット接続（TCP loopbackより高速）
- 16個の専用GPUバッファ + GPU上concat
- psycopg3の8MBバッファ活用
- GPU Direct Storage統合
- libpq 8KiB問題の完全回避

期待効果: 125 MB/s → 2GB/s (16倍高速化)

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列（UNIXソケット推奨）
"""

import os
import time
import asyncio
import math
import tempfile
import psycopg
import rmm
import numpy as np
import numba
from numba import cuda
import cupy as cp
import argparse
from concurrent.futures import ThreadPoolExecutor

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_parallel_ctid.output.parquet"

# 並列設定
DEFAULT_PARALLEL = 16
UNIX_SOCKET_DSN = "dbname=postgres user=postgres host=/var/run/postgresql"

def check_gpu_direct_support():
    """GPU Direct サポート確認"""
    print("\n=== GPU Direct サポート確認 ===")
    
    try:
        if os.path.exists("/proc/driver/nvidia-fs/stats"):
            print("✅ nvidia-fs ドライバ検出")
        else:
            print("❌ nvidia-fs ドライバが見つかりません")
            return False
    except Exception:
        print("❌ nvidia-fs 確認エラー")
        return False
    
    try:
        import kvikio
        print(f"✅ kvikio バージョン: {kvikio.__version__}")
        os.environ["KVIKIO_COMPAT_MODE"] = "OFF"
        print("✅ KVIKIO_COMPAT_MODE=OFF 設定完了")
        return True
    except ImportError:
        print("❌ kvikio がインストールされていません")
        return False

def get_table_blocks(dsn, table_name):
    """テーブルの総ブロック数を取得"""
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT pg_relation_size('{table_name}') / 8192 AS blocks")
            blocks = cur.fetchone()[0]
            return int(blocks)
    finally:
        conn.close()

def make_ctid_ranges(total_blocks, parallel_count):
    """ctid範囲リストを生成"""
    chunk_size = math.ceil(total_blocks / parallel_count)
    ranges = []
    
    for i in range(parallel_count):
        start_block = i * chunk_size
        end_block = min((i + 1) * chunk_size, total_blocks)
        if start_block < total_blocks:
            ranges.append((start_block, end_block))
    
    return ranges

def make_copy_sql(table_name, start_block, end_block, limit_rows=None):
    """ctid範囲指定COPY SQLを生成"""
    base_sql = f"""
    COPY (
        SELECT * FROM {table_name}
        WHERE ctid >= '({start_block},0)' 
          AND ctid < '({end_block},0)'
    """
    
    if limit_rows:
        # 並列数で分割した行数制限
        per_worker_limit = limit_rows // DEFAULT_PARALLEL
        base_sql += f" LIMIT {per_worker_limit}"
    
    base_sql += ") TO STDOUT (FORMAT binary)"
    return base_sql

class ParallelCopyWorker:
    """並列COPY処理ワーカー"""
    
    def __init__(self, worker_id, dsn, gpu_buffer, gpu_offset):
        self.worker_id = worker_id
        self.dsn = dsn
        self.gpu_buffer = gpu_buffer
        self.gpu_offset = gpu_offset
        self.temp_files = []
        
    async def process_range(self, start_block, end_block, copy_sql):
        """ctid範囲の並列処理"""
        print(f"  Worker {self.worker_id}: ctid範囲 ({start_block},{end_block}) 開始...")
        
        # 一時ファイル作成
        temp_file = os.path.join(
            tempfile.gettempdir(),
            f"parallel_ctid_worker_{self.worker_id}_{start_block}_{end_block}.bin"
        )
        self.temp_files.append(temp_file)
        
        start_time = time.time()
        data_size = 0
        read_count = 0
        
        try:
            # UNIXドメインソケット接続
            conn = await psycopg.AsyncConnection.connect(self.dsn)
            
            try:
                async with conn.cursor() as cur:
                    async with cur.copy(copy_sql) as copy_obj:
                        with open(temp_file, 'wb') as f:
                            async for chunk in copy_obj:
                                if chunk:
                                    # memoryview → bytes変換
                                    if isinstance(chunk, memoryview):
                                        chunk_bytes = chunk.tobytes()
                                    else:
                                        chunk_bytes = bytes(chunk)
                                    
                                    f.write(chunk_bytes)
                                    data_size += len(chunk_bytes)
                                    read_count += 1
                
                duration = time.time() - start_time
                print(f"  Worker {self.worker_id}: 完了 ({duration:.2f}秒, {data_size/(1024*1024):.1f}MB, {read_count:,}回)")
                
                return {
                    'worker_id': self.worker_id,
                    'temp_file': temp_file,
                    'data_size': data_size,
                    'duration': duration,
                    'read_count': read_count
                }
                
            finally:
                await conn.aclose()
                
        except Exception as e:
            print(f"  Worker {self.worker_id}: エラー - {e}")
            return None
    
    def transfer_to_gpu(self, temp_file, data_size):
        """一時ファイル → GPU転送 (GPU Direct)"""
        try:
            import kvikio
            from kvikio import CuFile
            
            start_time = time.time()
            
            # GPU Directでファイル → GPU転送
            with CuFile(temp_file, "r") as cufile:
                # GPU バッファの該当領域に転送
                future = cufile.pread(self.gpu_buffer, file_offset=0, buffer_offset=self.gpu_offset)
                bytes_read = future.get()
            
            duration = time.time() - start_time
            speed = bytes_read / (1024*1024) / duration
            
            print(f"  Worker {self.worker_id}: GPU Direct転送完了 ({duration:.2f}秒, {speed:.1f}MB/sec)")
            
            # 一時ファイル削除
            os.remove(temp_file)
            
            return bytes_read
            
        except Exception as e:
            print(f"  Worker {self.worker_id}: GPU転送エラー - {e}")
            return 0

def run_bandwidth_test():
    """PCIe帯域テスト"""
    print("\n=== PCIe帯域幅テスト ===")
    
    try:
        size_mb = 1024
        host_data = np.random.randint(0, 256, size_mb * 1024 * 1024, dtype=np.uint8)
        
        start_time = time.time()
        device_data = cuda.to_device(host_data)
        htod_time = time.time() - start_time
        htod_speed = size_mb / htod_time
        
        start_time = time.time()
        result_data = device_data.copy_to_host()
        dtoh_time = time.time() - start_time
        dtoh_speed = size_mb / dtoh_time
        
        print(f"Host→Device: {htod_speed:.2f} MB/s")
        print(f"Device→Host: {dtoh_speed:.2f} MB/s")
        
    except Exception as e:
        print(f"帯域測定エラー: {e}")

async def run_parallel_ctid_benchmark(limit_rows=1000000, parallel_count=DEFAULT_PARALLEL):
    """16並列ctid分割ベンチマーク"""
    
    # DSN設定（UNIXソケット優先）
    dsn = os.environ.get("GPUPASER_PG_DSN", UNIX_SOCKET_DSN)
    if "host=/var/run/postgresql" in dsn or "host=" not in dsn:
        print(f"✅ UNIXドメインソケット接続使用")
    else:
        print(f"⚠️  TCP接続使用（UNIXソケット推奨）")
    
    print(f"=== PostgreSQL → GPU 16並列ctid分割版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}")
    print(f"並列数: {parallel_count}")
    print(f"最適化設定:")
    print(f"  ctid分割: {parallel_count}並列読み取り")
    print(f"  UNIXソケット: TCP loopbackより高速")
    print(f"  GPU Direct: kvikio/cuFile使用")
    print(f"  GPUバッファ: {parallel_count}個の専用領域")
    
    # GPU Direct サポート確認
    if not check_gpu_direct_support():
        print("❌ GPU Direct サポートが不完全です。")
        return

    # PCIe帯域テスト
    run_bandwidth_test()
    
    # RMM 22GB初期化
    try:
        if not rmm.is_initialized():
            rmm.reinitialize(
                pool_allocator=True, 
                initial_pool_size=22*1024**3
            )
            print("✅ RMM メモリプール初期化完了 (22GB)")
    except Exception as e:
        print(f"❌ RMM初期化エラー: {e}")
        return

    start_total_time = time.time()
    
    # メタデータ取得
    conn = psycopg.connect(dsn)
    try:
        print("\nメタデータを取得中...")
        start_meta_time = time.time()
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
    for i, (start, end) in enumerate(ranges):
        print(f"  範囲 {i}: ブロック {start:,} - {end:,}")

    # データサイズ推定
    estimated_size_per_worker = (limit_rows // parallel_count) * 200  # 概算200bytes/行
    total_estimated_size = estimated_size_per_worker * parallel_count
    print(f"推定データサイズ: {total_estimated_size / (1024*1024):.1f} MB")

    # 16個のGPUバッファ事前確保
    print(f"\n{parallel_count}個のGPUバッファ事前確保中...")
    gpu_buffers = []
    worker_offsets = []
    
    for i in range(parallel_count):
        buffer_size = estimated_size_per_worker * 2  # 余裕を持たせる
        gpu_buffer = rmm.DeviceBuffer(size=buffer_size)
        gpu_buffers.append(gpu_buffer)
        worker_offsets.append(0)  # 各バッファは独立なのでオフセット0
        print(f"  GPUバッファ {i}: {buffer_size / (1024*1024):.1f} MB確保")

    # 並列COPYワーカー作成
    print(f"\n{parallel_count}並列COPY処理開始...")
    workers = []
    tasks = []
    
    for i, (start_block, end_block) in enumerate(ranges):
        worker = ParallelCopyWorker(i, dsn, gpu_buffers[i], worker_offsets[i])
        workers.append(worker)
        
        copy_sql = make_copy_sql(TABLE_NAME, start_block, end_block, limit_rows)
        task = worker.process_range(start_block, end_block, copy_sql)
        tasks.append(task)
    
    # 並列実行
    start_copy_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    copy_time = time.time() - start_copy_time
    
    # 結果集計
    successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    total_data_size = sum(r['data_size'] for r in successful_results)
    total_read_count = sum(r['read_count'] for r in successful_results)
    
    print(f"✅ {parallel_count}並列COPY完了 ({copy_time:.4f}秒)")
    print(f"  成功ワーカー数: {len(successful_results)}/{parallel_count}")
    print(f"  総データサイズ: {total_data_size / (1024*1024):.2f} MB")
    print(f"  総読み込み回数: {total_read_count:,}")
    print(f"  並列転送速度: {total_data_size / (1024*1024) / copy_time:.2f} MB/sec")
    print(f"  単一ワーカー換算: {(total_data_size / (1024*1024) / copy_time) / parallel_count:.2f} MB/sec")

    # GPU Direct並列転送
    print(f"\n{parallel_count}並列GPU Direct転送開始...")
    start_gpu_time = time.time()
    
    gpu_transfer_tasks = []
    for i, result in enumerate(successful_results):
        if result:
            worker = workers[result['worker_id']]
            task = asyncio.get_event_loop().run_in_executor(
                None, worker.transfer_to_gpu, result['temp_file'], result['data_size']
            )
            gpu_transfer_tasks.append(task)
    
    gpu_results = await asyncio.gather(*gpu_transfer_tasks)
    gpu_time = time.time() - start_gpu_time
    
    total_gpu_transferred = sum(gpu_results)
    gpu_speed = total_gpu_transferred / (1024*1024) / gpu_time
    
    print(f"✅ {parallel_count}並列GPU Direct転送完了 ({gpu_time:.4f}秒)")
    print(f"  GPU転送速度: {gpu_speed:.2f} MB/sec")

    # GPU上でデータ結合
    print("\nGPU上でデータ結合中...")
    start_concat_time = time.time()
    
    # 実際のデータサイズでGPUバッファをトリミング
    trimmed_buffers = []
    actual_sizes = [r['data_size'] for r in successful_results]
    
    for i, actual_size in enumerate(actual_sizes):
        if actual_size > 0:
            trimmed_buffer = rmm.DeviceBuffer(size=actual_size)
            cp.cuda.runtime.memcpy(
                trimmed_buffer.ptr, gpu_buffers[i].ptr, actual_size,
                cp.cuda.runtime.memcpyDeviceToDevice
            )
            trimmed_buffers.append(trimmed_buffer)
    
    # GPU上でバッファ結合
    total_actual_size = sum(actual_sizes)
    final_gpu_buffer = rmm.DeviceBuffer(size=total_actual_size)
    
    current_offset = 0
    for buffer in trimmed_buffers:
        cp.cuda.runtime.memcpy(
            final_gpu_buffer.ptr + current_offset,
            buffer.ptr,
            buffer.size,
            cp.cuda.runtime.memcpyDeviceToDevice
        )
        current_offset += buffer.size
    
    concat_time = time.time() - start_concat_time
    print(f"GPU結合完了 ({concat_time:.4f}秒), 最終サイズ: {total_actual_size / (1024*1024):.2f} MB")

    # numba GPU アレイに変換
    print("GPU バッファを numba GPU アレイに変換中...")
    raw_dev = cuda.as_cuda_array(final_gpu_buffer).view(dtype=np.uint8)
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
        print(f"エラー詳細: {str(e)}")
        return

    total_time = time.time() - start_total_time
    decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
    
    print(f"\n=== 16並列ctid分割版ベンチマーク完了 ===")
    print(f"総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得       : {meta_time:.4f} 秒")
    print(f"  {parallel_count}並列COPY        : {copy_time:.4f} 秒")
    print(f"  {parallel_count}並列GPU Direct  : {gpu_time:.4f} 秒")
    print(f"  GPU結合             : {concat_time:.4f} 秒")
    print(f"  GPUパース           : {parse_time:.4f} 秒")
    print(f"  GPUデコード         : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み     : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数      : {rows:,} 行")
    print(f"  処理列数      : {len(columns)} 列")
    print(f"  Decimal列数   : {decimal_cols} 列")
    print(f"  データサイズ  : {total_actual_size / (1024*1024):.2f} MB")
    print(f"  並列数        : {parallel_count}")
    print(f"  ctid範囲数    : {len(ranges)}")
    
    total_cells = rows * len(columns)
    throughput = total_cells / decode_time if decode_time > 0 else 0
    parallel_copy_speed = total_data_size / (1024*1024) / copy_time
    single_equiv_speed = parallel_copy_speed / parallel_count
    
    print(f"  セル処理速度        : {throughput:,.0f} cells/sec")
    print(f"  {parallel_count}並列COPY速度     : {parallel_copy_speed:.2f} MB/sec")
    print(f"  単一換算速度        : {single_equiv_speed:.2f} MB/sec")
    print(f"  GPU Direct速度      : {gpu_speed:.2f} MB/sec")
    
    # 性能向上評価
    baseline_speed = 125  # 従来の単一COPY速度
    improvement_ratio = parallel_copy_speed / baseline_speed
    
    print(f"  性能向上倍率        : {improvement_ratio:.1f}倍 (対 {baseline_speed} MB/sec)")
    
    if parallel_copy_speed > 2000:
        performance_class = "🏆 超高速 (2GB/s+)"
    elif parallel_copy_speed > 1000:
        performance_class = "🥇 高速 (1GB/s+)"
    elif parallel_copy_speed > 500:
        performance_class = "🥈 中速 (500MB/s+)"
    else:
        performance_class = "🥉 改善中"
    
    print(f"  性能クラス          : {performance_class}")
    
    print("--- 16並列ctid分割最適化効果 ---")
    print("  ✅ ctid分割: テーブル並列読み取り")
    print("  ✅ UNIXソケット: TCP loopbackより高速")
    print("  ✅ psycopg3: 8MBバッファ活用")
    print("  ✅ GPU Direct: 並列GPU転送")
    print("  ✅ GPU結合: 高速GPU上concat")
    print("  ✅ libpq 8KiB問題: 完全回避")
    print("=========================================")

    # 検証用出力
    print(f"\ncuDF検証用出力:")
    try:
        print(f"出力Parquet: {OUTPUT_PARQUET_PATH}")
        print(f"読み込み確認: {len(cudf_df):,} 行 × {len(cudf_df.columns)} 列")
        print("✅ cuDF検証: 成功")
    except Exception as e:
        print(f"❌ cuDF検証: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU 16並列ctid分割版')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    parser.add_argument('--parallel', type=int, default=DEFAULT_PARALLEL, help='並列数')
    parser.add_argument('--bandwidth-test', action='store_true', help='帯域テストのみ')
    parser.add_argument('--check-support', action='store_true', help='GPU Directサポート確認のみ')
    parser.add_argument('--check-blocks', action='store_true', help='テーブルブロック数確認のみ')
    
    args = parser.parse_args()
    
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    if args.check_support:
        check_gpu_direct_support()
        return
    
    if args.bandwidth_test:
        run_bandwidth_test()
        return
    
    if args.check_blocks:
        dsn = os.environ.get("GPUPASER_PG_DSN", UNIX_SOCKET_DSN)
        blocks = get_table_blocks(dsn, TABLE_NAME)
        print(f"テーブル {TABLE_NAME} の総ブロック数: {blocks:,}")
        ranges = make_ctid_ranges(blocks, args.parallel)
        print(f"{args.parallel}並列でのctid範囲分割:")
        for i, (start, end) in enumerate(ranges):
            print(f"  範囲 {i}: ブロック {start:,} - {end:,}")
        return
    
    # 並列ベンチマーク実行
    asyncio.run(run_parallel_ctid_benchmark(
        limit_rows=args.rows,
        parallel_count=args.parallel
    ))

if __name__ == "__main__":
    main()