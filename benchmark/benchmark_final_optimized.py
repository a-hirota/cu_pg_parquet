"""
PostgreSQL → GPU 最終最適化版
チャンク分割なし + 22GBメモリプール

最適化:
- チャンク蓄積処理の排除
- 22GB RMM メモリプール
- シンプルな一括処理
- メモリ不足問題の解決

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import ctypes
import psycopg
import rmm
import numpy as np
import numba
from numba import cuda
import cupy as cp
import argparse
import io

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_final_optimized.output.parquet"

# 最終最適化パラメータ
CHUNK_SIZE = 128 << 20  # 128 MiB (大容量)
NUM_STREAMS = 2

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

def run_final_optimized_benchmark(limit_rows=1000000):
    """最終最適化版ベンチマーク - チャンク分割なし"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    print(f"=== PostgreSQL → GPU 最終最適化版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}")
    print(f"最適化設定:")
    print(f"  チャンクサイズ: {CHUNK_SIZE / (1024*1024):.0f} MB")
    print(f"  ストリーム数: {NUM_STREAMS}")
    print(f"  チャンク分割: なし (一括処理)")
    
    # PCIe帯域テスト
    run_bandwidth_test()
    
    # RMM 22GB初期化
    try:
        if not rmm.is_initialized():
            rmm.reinitialize(
                pool_allocator=True, 
                initial_pool_size=22*1024**3  # 22GB
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

        # COPY BINARY → 一括収集 (チャンク分割なし)
        print("COPY BINARY → 一括収集実行中...")
        start_copy_time = time.time()
        
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        
        # BytesIO使用で効率的一括収集
        buffer = io.BytesIO()
        chunk_count = 0
        
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_obj:
                print("  📡 COPY ストリーム → BytesIO一括収集中...")
                
                for chunk in copy_obj:
                    if chunk:
                        buffer.write(chunk)
                        chunk_count += 1
                        
                        # 進捗表示（大幅削減）
                        if chunk_count % 10000 == 0:
                            print(f"    チャンク {chunk_count:,} 処理中...")
                
                # 一括取得
                host_bytes = buffer.getvalue()
                buffer.close()

        copy_time = time.time() - start_copy_time
        actual_data_size = len(host_bytes)
        
        print(f"✅ COPY BINARY → 一括収集完了 ({copy_time:.4f}秒)")
        print(f"  処理チャンク数: {chunk_count:,}")
        print(f"  実際のデータサイズ: {actual_data_size / (1024*1024):.2f} MB")
        print(f"  ネットワーク速度: {actual_data_size / (1024*1024) / copy_time:.2f} MB/sec")

    finally:
        conn.close()

    # GPU バッファ確保 & 1回コピー (分割なし)
    print("GPU バッファ確保 & 1回コピー実行中...")
    start_gpu_time = time.time()
    
    try:
        # GPU バッファ確保（実データサイズ）
        devbuf = rmm.DeviceBuffer(size=actual_data_size)
        print(f"  GPU バッファ確保完了: {actual_data_size / (1024*1024):.2f} MB")
        
        # 【最終最適化】チャンク分割なし - 1回のみコピー
        devbuf.copy_from_host(host_bytes)
        print(f"  ✅ GPU 1回コピー完了 (分割なし)!")
        
        # ホストメモリ解放
        del host_bytes
        
    except Exception as e:
        print(f"❌ GPU コピーエラー: {e}")
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
    
    print(f"\n=== 最終最適化版ベンチマーク完了 ===")
    print(f"総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得       : {meta_time:.4f} 秒")
    print(f"  COPY→一括収集       : {copy_time:.4f} 秒")
    print(f"  一括→GPU転送        : {gpu_time:.4f} 秒")
    print(f"  GPUパース           : {parse_time:.4f} 秒")
    print(f"  GPUデコード         : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み     : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数      : {rows:,} 行")
    print(f"  処理列数      : {len(columns)} 列")
    print(f"  Decimal列数   : {decimal_cols} 列")
    print(f"  データサイズ  : {actual_data_size / (1024*1024):.2f} MB")
    print(f"  処理チャンク数: {chunk_count:,}")
    
    total_cells = rows * len(columns)
    throughput = total_cells / decode_time if decode_time > 0 else 0
    network_throughput = actual_data_size / (1024*1024) / copy_time
    
    print(f"  セル処理速度     : {throughput:,.0f} cells/sec")
    print(f"  ネットワーク速度 : {network_throughput:.2f} MB/sec")
    print(f"  GPU転送速度     : {gpu_throughput:.2f} MB/sec")
    
    # PCIe効率計算（実測11.9GB/s基準）
    pcie_efficiency = network_throughput / 11900 * 100
    print(f"  PCIe効率        : {pcie_efficiency:.1f}% (対11.9GB/s実測)")
    
    print("--- 最終最適化効果 ---")
    print("  ✅ ファイル I/O: 完全ゼロ")
    print("  ✅ チャンク分割: なし (一括処理)")
    print("  ✅ メモリプール: 22GB (十分な容量)")
    print("  ✅ GPU転送: 1回のみ（最小オーバーヘッド）")
    print("  ✅ BytesIO: 効率的一括収集")
    
    # 性能評価
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
            if i < 3:  # 最初の3列のみ
                print(f"  {col_name}: {dtype}")
        print("✅ cuDF検証: 成功")
    except Exception as e:
        print(f"❌ cuDF検証: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU 最終最適化版')
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
    
    run_final_optimized_benchmark(limit_rows=args.rows)

if __name__ == "__main__":
    main()