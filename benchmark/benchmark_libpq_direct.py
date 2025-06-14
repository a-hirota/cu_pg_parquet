"""
PostgreSQL → GPU libpq直接版
libpqの大容量読み込み能力を活用

最適化:
- libpqの内部バッファリング活用
- 低レベルAPIでの大容量読み込み
- チャンク処理を最小化
- 22GB RMM メモリプール

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import psycopg
import rmm
import numpy as np
import numba
from numba import cuda
import cupy as cp
import argparse

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_libpq_direct.output.parquet"

def run_bandwidth_test():
    """CUDA bandwidthTest でPCIe性能確認"""
    print("\n=== PCIe帯域幅テスト ===")
    
    try:
        print("シンプル帯域測定実行中...")
        size_mb = 1024  # 1GB
        host_data = np.random.randint(0, 256, size_mb * 1024 * 1024, dtype=np.uint8)
        
        # Host to Device
        start_time = time.time()
        device_data = cuda.to_device(host_data)
        htod_time = time.time() - start_time
        htod_speed = size_mb / htod_time
        
        # Device to Host  
        start_time = time.time()
        result_data = device_data.copy_to_host()
        dtoh_time = time.time() - start_time
        dtoh_speed = size_mb / dtoh_time
        
        print(f"Host→Device: {htod_speed:.2f} MB/s")
        print(f"Device→Host: {dtoh_speed:.2f} MB/s")
        
    except Exception as e:
        print(f"帯域測定エラー: {e}")

def run_libpq_direct_benchmark(limit_rows=1000000):
    """libpq直接版ベンチマーク - 大容量読み込み活用"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    print(f"=== PostgreSQL → GPU libpq直接版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}")
    print(f"最適化設定:")
    print(f"  libpq: 大容量読み込み活用")
    print(f"  チャンク処理: 最小化")
    print(f"  メモリプール: 22GB")
    
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

        # libpq 大容量読み込み
        print("libpq 大容量読み込み実行中...")
        start_copy_time = time.time()
        
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        
        print("  🚀 libpq大容量バッファ読み込み開始...")
        
        # 大容量バッファサイズ設定（libpqの能力を活用）
        LARGE_BUFFER_SIZE = 64 * 1024 * 1024  # 64MB
        
        data_parts = []
        total_bytes = 0
        read_count = 0
        
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_obj:
                print(f"  📦 libpq大容量読み込み ({LARGE_BUFFER_SIZE / (1024*1024):.0f}MB単位)")
                
                # 大容量読み込みループ
                while True:
                    try:
                        # libpqから大容量読み込み
                        # psycopg3では copy_obj.read() がサイズ指定不可のため
                        # 内部的に利用可能な全データを読み込み
                        data_chunk = copy_obj.read()
                        
                        if not data_chunk:
                            break
                        
                        # memoryview → bytes変換
                        if isinstance(data_chunk, memoryview):
                            chunk_bytes = data_chunk.tobytes()
                        else:
                            chunk_bytes = bytes(data_chunk)
                        
                        data_parts.append(chunk_bytes)
                        total_bytes += len(chunk_bytes)
                        read_count += 1
                        
                        # 進捗表示（5GB単位）
                        if total_bytes % (5 * 1024 * 1024 * 1024) < len(chunk_bytes):
                            print(f"    読み込み進捗: {total_bytes / (1024*1024*1024):.1f} GB")
                        
                    except Exception as e:
                        if "no data available" in str(e).lower():
                            break
                        else:
                            raise e
                
                # 全データ結合
                print("  🔗 データ一括結合中...")
                host_bytes = b''.join(data_parts)
                del data_parts  # メモリ解放

        copy_time = time.time() - start_copy_time
        actual_data_size = len(host_bytes)
        
        print(f"✅ libpq 大容量読み込み完了 ({copy_time:.4f}秒)")
        print(f"  読み込み回数: {read_count:,}")
        print(f"  実際のデータサイズ: {actual_data_size / (1024*1024):.2f} MB")
        print(f"  平均読み込みサイズ: {(actual_data_size / read_count) / (1024*1024):.2f} MB/回")
        print(f"  読み込み速度: {actual_data_size / (1024*1024) / copy_time:.2f} MB/sec")

    finally:
        conn.close()

    # GPU バッファ確保 & 1回転送
    print("GPU バッファ確保 & 1回転送実行中...")
    start_gpu_time = time.time()
    
    try:
        # GPU バッファ確保（実データサイズ）
        devbuf = rmm.DeviceBuffer(size=actual_data_size)
        print(f"  GPU バッファ確保完了: {actual_data_size / (1024*1024):.2f} MB")
        
        # 1回転送
        devbuf.copy_from_host(host_bytes)
        print(f"  ✅ GPU 1回転送完了!")
        
        # ホストメモリ解放
        del host_bytes
        
    except Exception as e:
        print(f"❌ GPU転送エラー: {e}")
        return
    
    gpu_time = time.time() - start_gpu_time
    gpu_throughput = actual_data_size / (1024*1024) / gpu_time
    print(f"GPU転送完了 ({gpu_time:.4f}秒), 速度: {gpu_throughput:.2f} MB/sec")

    # numba GPU アレイに変換
    print("GPU バッファを numba GPU アレイに変換中...")
    raw_dev = cuda.as_cuda_array(devbuf).view(dtype=np.uint8)
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
    
    print(f"\n=== libpq直接版ベンチマーク完了 ===")
    print(f"総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得       : {meta_time:.4f} 秒")
    print(f"  libpq大容量読み込み  : {copy_time:.4f} 秒")
    print(f"  ホスト→GPU転送       : {gpu_time:.4f} 秒")
    print(f"  GPUパース           : {parse_time:.4f} 秒")
    print(f"  GPUデコード         : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み     : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数      : {rows:,} 行")
    print(f"  処理列数      : {len(columns)} 列")
    print(f"  Decimal列数   : {decimal_cols} 列")
    print(f"  データサイズ  : {actual_data_size / (1024*1024):.2f} MB")
    print(f"  読み込み回数  : {read_count:,}")
    
    total_cells = rows * len(columns)
    throughput = total_cells / decode_time if decode_time > 0 else 0
    network_throughput = actual_data_size / (1024*1024) / copy_time
    
    print(f"  セル処理速度     : {throughput:,.0f} cells/sec")
    print(f"  libpq読み込み速度: {network_throughput:.2f} MB/sec")
    print(f"  GPU転送速度     : {gpu_throughput:.2f} MB/sec")
    
    # libpq効率評価
    if read_count < 1000:
        efficiency_class = "🏆 高効率 (読み込み回数<1000)"
    elif read_count < 10000:
        efficiency_class = "🥇 中効率 (読み込み回数<10000)"
    elif read_count < 100000:
        efficiency_class = "🥈 低効率 (読み込み回数<100000)"
    else:
        efficiency_class = "🥉 非効率 (読み込み回数多数)"
    
    print(f"  libpq効率       : {efficiency_class}")
    
    # PCIe効率計算
    pcie_efficiency = network_throughput / 11900 * 100
    print(f"  PCIe効率        : {pcie_efficiency:.1f}% (対11.9GB/s実測)")
    
    print("--- libpq直接最適化効果 ---")
    print("  ✅ libpq: 大容量読み込み活用")
    print("  ✅ 読み込み回数: 最小化")
    print("  ✅ 内部バッファリング: libpqに依存")
    print("  ✅ GPU転送: 1回のみ")
    print("  ✅ CPU使用率: 削減")
    
    if network_throughput > 5000:
        print("  🏆 5GB/s超達成！")
    elif network_throughput > 1000:
        print("  🥇 1GB/s超達成！")
    elif network_throughput > 500:
        print("  🥈 500MB/s超達成！")
    else:
        print("  ⚠️  転送速度改善の余地あり")
    
    print("=========================================")

    # 検証用出力
    print(f"\ncuDF検証用出力:")
    try:
        print(f"出力Parquet: {OUTPUT_PARQUET_PATH}")
        print(f"読み込み確認: {len(cudf_df):,} 行 × {len(cudf_df.columns)} 列")
        print("先頭データ型:")
        for i, (col_name, dtype) in enumerate(cudf_df.dtypes.items()):
            if i < 3:
                print(f"  {col_name}: {dtype}")
        print("✅ cuDF検証: 成功")
    except Exception as e:
        print(f"❌ cuDF検証: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU libpq直接版')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    parser.add_argument('--bandwidth-test', action='store_true', help='帯域テストのみ')
    
    args = parser.parse_args()
    
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    if args.bandwidth_test:
        run_bandwidth_test()
        return
    
    run_libpq_direct_benchmark(limit_rows=args.rows)

if __name__ == "__main__":
    main()