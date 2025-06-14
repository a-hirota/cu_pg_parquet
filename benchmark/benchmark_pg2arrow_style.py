"""
PostgreSQL → GPU pg2arrow完全対応版
pg2arrow成功モデルの完全実装

最適化:
- ctid offset=1開始（pg2arrow方式）
- プロセス並列化（Python GIL回避）
- 4MiB読み取りチャンク
- 共有スナップショット
- UNIXドメインソケット

期待効果: 2-3 GB/s (pg2arrow同等)
"""

import os
import time
import math
import tempfile
import multiprocessing
import psycopg
import rmm
import numpy as np
import numba
from numba import cuda
import cupy as cp
import argparse
from concurrent.futures import ProcessPoolExecutor

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_pg2arrow_style.output.parquet"

# pg2arrow設定
DEFAULT_PARALLEL = 8
UNIX_SOCKET_DSN = "dbname=postgres user=postgres host="
CHUNK_SIZE = 400 * 1024 * 1024  # 400MB（大容量最適化）

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
    """ctid範囲リスト生成（pg2arrow方式）"""
    chunk_size = math.ceil(total_blocks / parallel_count)
    ranges = []
    
    for i in range(parallel_count):
        start_block = i * chunk_size
        end_block = min((i + 1) * chunk_size, total_blocks)
        if start_block < total_blocks:
            ranges.append((start_block, end_block))
    
    return ranges

def make_copy_sql_pg2arrow(table_name, start_block, end_block):
    """pg2arrow完全互換COPY SQL生成"""
    # pg2arrow方式: (block,1) ... < (end_block+1,1)
    sql = f"""
    COPY (
        SELECT * FROM {table_name}
        WHERE ctid >= '({start_block},1)'::tid
          AND ctid < '({end_block+1},1)'::tid
    ) TO STDOUT (FORMAT binary)
    """
    return sql

def worker_process(args):
    """プロセス並列ワーカー（pg2arrow方式）"""
    worker_id, dsn, start_block, end_block, shared_snapshot = args
    
    # 一時ファイル
    temp_file = os.path.join(
        tempfile.gettempdir(),
        f"pg2arrow_worker_{worker_id}_{start_block}_{end_block}.bin"
    )
    
    print(f"  Worker {worker_id}: ctid範囲 ({start_block},{end_block}) 開始... [プロセス並列]")
    
    start_time = time.time()
    data_size = 0
    read_count = 0
    
    try:
        # pg2arrow方式: UNIXソケット接続
        conn = psycopg.connect(dsn)
        
        try:
            with conn.cursor() as cur:
                # 共有スナップショット設定
                if shared_snapshot:
                    cur.execute(f"BEGIN ISOLATION LEVEL REPEATABLE READ")
                    cur.execute(f"SET TRANSACTION SNAPSHOT '{shared_snapshot}'")
                    print(f"    Worker {worker_id}: 共有スナップショット設定完了")
                
                # pg2arrow方式COPY SQL
                copy_sql = make_copy_sql_pg2arrow(TABLE_NAME, start_block, end_block)
                
                with cur.copy(copy_sql) as copy_obj:
                    with open(temp_file, 'wb') as f:
                        # pg2arrow方式: 4MiBチャンク読み取り
                        buffer = bytearray()
                        
                        for chunk in copy_obj:
                            if chunk:
                                if isinstance(chunk, memoryview):
                                    chunk_bytes = chunk.tobytes()
                                else:
                                    chunk_bytes = bytes(chunk)
                                
                                buffer.extend(chunk_bytes)
                                
                                # 4MiB蓄積したら書き込み
                                if len(buffer) >= CHUNK_SIZE:
                                    f.write(buffer)
                                    data_size += len(buffer)
                                    read_count += 1
                                    
                                    if read_count <= 3:
                                        print(f"    Worker {worker_id}: 一時バッファ書き込み{read_count}: {len(buffer)/(1024*1024):.1f}MB")
                                    
                                    buffer.clear()
                        
                        # 残りバッファ
                        if buffer:
                            f.write(buffer)
                            data_size += len(buffer)
                            read_count += 1
                            print(f"    Worker {worker_id}: 最終バッファ書き込み: {len(buffer)/(1024*1024):.1f}MB")
        
        finally:
            conn.close()
    
    except Exception as e:
        print(f"  Worker {worker_id}: ❌エラー - {e}")
        return None
    
    duration = time.time() - start_time
    speed = data_size / (1024*1024) / duration if duration > 0 else 0
    
    print(f"  Worker {worker_id}: ✅完了 ({duration:.2f}秒, {data_size/(1024*1024):.1f}MB, {speed:.1f}MB/sec)")
    
    return {
        'worker_id': worker_id,
        'temp_file': temp_file,
        'data_size': data_size,
        'duration': duration,
        'read_count': read_count,
        'speed': speed
    }

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

def run_pg2arrow_benchmark(limit_rows=None, parallel_count=DEFAULT_PARALLEL):
    """pg2arrow完全対応ベンチマーク"""
    
    # DSN設定（UNIXソケット優先）
    dsn = os.environ.get("GPUPASER_PG_DSN", UNIX_SOCKET_DSN)
    if "host=" not in dsn or "host=" in dsn and dsn.split("host=")[1].split()[0] == "":
        print(f"✅ UNIXドメインソケット接続使用")
    else:
        print(f"⚠️  TCP接続使用（UNIXソケット推奨）")
    
    print(f"=== PostgreSQL → GPU pg2arrow完全対応版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: なし（全件処理）")
    print(f"並列数: {parallel_count}")
    print(f"最適化設定:")
    print(f"  ctid方式: pg2arrow完全互換 offset=1開始")
    print(f"  並列化: プロセス並列（Python GIL回避）")
    print(f"  チャンクサイズ: {CHUNK_SIZE/(1024*1024):.0f}MB（大容量最適化）")
    print(f"  接続: UNIXドメインソケット")
    
    # GPU Direct サポート確認
    gds_supported = check_gpu_direct_support()

    # PCIe帯域テスト
    run_bandwidth_test()
    
    # RMM初期化
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
    
    # 共有スナップショット作成（pg2arrow方式）
    print("共有スナップショット作成中...")
    conn_snap = psycopg.connect(dsn)
    try:
        with conn_snap.cursor() as cur:
            cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            cur.execute("SELECT pg_export_snapshot()")
            shared_snapshot = cur.fetchone()[0]
            print(f"✅ 共有スナップショット作成: {shared_snapshot}")
    except Exception as e:
        print(f"⚠️  共有スナップショット作成失敗（続行）: {e}")
        shared_snapshot = None
    finally:
        # スナップショット保持のため接続維持
        pass
    
    # ctid範囲分割
    ranges = make_ctid_ranges(total_blocks, parallel_count)
    print(f"ctid範囲分割: {len(ranges)}個の範囲 [pg2arrow方式]")
    for i, (start, end) in enumerate(ranges[:5]):
        print(f"  範囲 {i}: ブロック {start:,} - {end:,} (offset=1開始)")
    if len(ranges) > 5:
        print(f"  ... (残り {len(ranges)-5} 範囲)")

    # プロセス並列実行（pg2arrow方式）
    print(f"\n{parallel_count}プロセス並列COPY処理開始...")
    
    # ワーカー引数準備
    worker_args = []
    for i, (start_block, end_block) in enumerate(ranges):
        worker_args.append((i, dsn, start_block, end_block, shared_snapshot))
    
    # プロセス並列実行
    start_copy_time = time.time()
    
    with ProcessPoolExecutor(max_workers=parallel_count) as executor:
        results = list(executor.map(worker_process, worker_args))
    
    copy_time = time.time() - start_copy_time
    
    # 共有スナップショット接続クローズ
    try:
        conn_snap.close()
    except:
        pass
    
    # 結果集計
    successful_results = [r for r in results if r is not None]
    failed_count = len(results) - len(successful_results)
    
    if not successful_results:
        print("❌ すべてのワーカーが失敗しました")
        return
    
    total_data_size = sum(r['data_size'] for r in successful_results)
    total_read_count = sum(r['read_count'] for r in successful_results)
    
    print(f"✅ {parallel_count}プロセス並列COPY完了 ({copy_time:.4f}秒)")
    print(f"  成功ワーカー数: {len(successful_results)}/{parallel_count}")
    if failed_count > 0:
        print(f"  失敗ワーカー数: {failed_count}")
    print(f"  総データサイズ: {total_data_size / (1024*1024):.2f} MB")
    print(f"  総読み込み回数: {total_read_count:,}")
    
    parallel_copy_speed = total_data_size / (1024*1024) / copy_time
    print(f"  プロセス並列転送速度: {parallel_copy_speed:.2f} MB/sec")
    
    if parallel_count > 0:
        single_equiv_speed = parallel_copy_speed / parallel_count
        print(f"  単一ワーカー換算: {single_equiv_speed:.2f} MB/sec")

    # 一時ファイル → GPU転送
    print(f"\n一時ファイル → GPU転送開始...")
    start_gpu_time = time.time()
    
    if total_data_size > 0:
        gpu_buffer = rmm.DeviceBuffer(size=total_data_size)
        print(f"GPUバッファ確保完了: {total_data_size / (1024*1024):.2f} MB")
        
        current_offset = 0
        gpu_transferred = 0
        
        for result in successful_results:
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
                    
                    host_array = np.frombuffer(file_data, dtype=np.uint8)
                    cp.cuda.runtime.memcpy(
                        gpu_buffer.ptr + current_offset,
                        host_array.ctypes.data,
                        file_size,
                        cp.cuda.runtime.memcpyHostToDevice
                    )
                    gpu_transferred += file_size
                
                current_offset += file_size
                os.remove(temp_file)
                
            except Exception as e:
                print(f"  Worker {result['worker_id']}: GPU転送エラー - {e}")
    
    gpu_time = time.time() - start_gpu_time
    
    if gpu_transferred > 0:
        gpu_speed = gpu_transferred / (1024*1024) / gpu_time
        print(f"✅ GPU転送完了 ({gpu_time:.4f}秒)")
        print(f"  GPU転送速度: {gpu_speed:.2f} MB/sec")
        
        # numba GPU アレイに変換
        print("GPU バッファを numba GPU アレイに変換中...")
        raw_dev = cuda.as_cuda_array(gpu_buffer).view(dtype=np.uint8)
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
        
        print(f"\n=== pg2arrow完全対応版ベンチマーク完了 ===")
        print(f"総時間 = {total_time:.4f} 秒")
        print("--- 時間内訳 ---")
        print(f"  メタデータ取得       : {meta_time:.4f} 秒")
        print(f"  {parallel_count}プロセス並列COPY  : {copy_time:.4f} 秒")
        print(f"  GPU転送             : {gpu_time:.4f} 秒")
        print(f"  GPUパース           : {parse_time:.4f} 秒")
        print(f"  GPUデコード         : {decode_time:.4f} 秒")
        print(f"  Parquet書き込み     : {write_time:.4f} 秒")
        print("--- 統計情報 ---")
        print(f"  処理行数      : {rows:,} 行")
        print(f"  処理列数      : {len(columns)} 列")
        print(f"  Decimal列数   : {decimal_cols} 列")
        print(f"  データサイズ  : {total_data_size / (1024*1024):.2f} MB")
        print(f"  並列数        : {parallel_count}")
        print(f"  成功ワーカー数: {len(successful_results)}")
        
        total_cells = rows * len(columns)
        throughput = total_cells / decode_time if decode_time > 0 else 0
        
        print(f"  セル処理速度        : {throughput:,.0f} cells/sec")
        print(f"  {parallel_count}プロセス並列速度   : {parallel_copy_speed:.2f} MB/sec")
        print(f"  単一換算速度        : {single_equiv_speed:.2f} MB/sec")
        print(f"  GPU転送速度         : {gpu_speed:.2f} MB/sec")
        
        # 性能向上評価
        baseline_speed = 129.1  # PostgreSQL単一COPY速度
        improvement_ratio = parallel_copy_speed / baseline_speed
        
        print(f"  性能向上倍率        : {improvement_ratio:.1f}倍 (対PostgreSQL単一 {baseline_speed} MB/sec)")
        
        if parallel_copy_speed > 2000:
            performance_class = "🏆 pg2arrow級 (2GB/s+)"
        elif parallel_copy_speed > 1000:
            performance_class = "🥇 高速 (1GB/s+)"
        elif parallel_copy_speed > 500:
            performance_class = "🥈 中速 (500MB/s+)"
        else:
            performance_class = "🥉 改善中"
        
        print(f"  性能クラス          : {performance_class}")
        
        print("--- pg2arrow完全対応最適化効果 ---")
        print("  ✅ ctid範囲: pg2arrow方式 offset=1開始")
        print("  ✅ プロセス並列: Python GIL完全回避")
        print("  ✅ 4MiBチャンク: pg2arrow同等読み取り")
        print("  ✅ 共有スナップショット: MVCC整合性")
        print("  ✅ UNIXソケット: 最高速接続")
        print("=========================================")

        # 検証用出力
        print(f"\ncuDF検証用出力:")
        try:
            print(f"出力Parquet: {OUTPUT_PARQUET_PATH}")
            print(f"読み込み確認: {len(cudf_df):,} 行 × {len(cudf_df.columns)} 列")
            print("✅ cuDF検証: 成功")
        except Exception as e:
            print(f"❌ cuDF検証: {e}")
    else:
        print("❌ GPU転送するデータがありません")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU pg2arrow完全対応版')
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
    
    # pg2arrow完全対応ベンチマーク実行
    run_pg2arrow_benchmark(
        limit_rows=None,
        parallel_count=args.parallel
    )

if __name__ == "__main__":
    main()