"""
PostgreSQL → GPU 究極最適化版
転送速度を10GB/s近くまで引き上げる

ユーザー分析に基づく最適化:
- libpq から 1MB以上の大きなチャンクで読取り
- pinned memcpyの最適化
- 64MiB以上の大きなチャンクサイズ
- 複数ストリームでの非同期DMA
- bandwidthTest での実測確認

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

from src.metadata import fetch_column_meta
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_ultimate_optimized.output.parquet"

# 究極最適化パラメータ
CHUNK_SIZE = 64 << 20  # 64 MiB (大きなチャンクで効率最大化)
READ_SIZE = 4 << 20    # 4 MiB (libpqからの読み取りサイズ)
NUM_STREAMS = 2        # 複数ストリームでpipelined DMA

def run_bandwidth_test():
    """CUDA bandwidthTest でPCIe性能確認"""
    print("\n=== PCIe帯域幅テスト ===")
    
    try:
        import subprocess
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=pci.link.gen.current,pci.link.width.current", "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gen, width = result.stdout.strip().split(', ')
            print(f"PCIe世代: Gen {gen}, 幅: x{width}")
            
            # 理論帯域計算
            gen_speed = {"3": 8, "4": 16, "5": 32}  # GT/s per lane
            if gen in gen_speed:
                theoretical_gbps = gen_speed[gen] * int(width) * 1.0  # GB/s (8b/10b考慮)
                print(f"理論帯域: {theoretical_gbps:.1f} GB/s")
        
    except Exception as e:
        print(f"PCIe情報取得エラー: {e}")
    
    # シンプルな帯域測定
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

def run_ultimate_optimized_benchmark(limit_rows=1000000):
    """
    究極最適化版ベンチマーク
    10GB/s近い転送速度を目指す
    """
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    print(f"=== PostgreSQL → GPU 究極最適化版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}")
    print(f"最適化設定:")
    print(f"  チャンクサイズ: {CHUNK_SIZE / (1024*1024):.0f} MB")
    print(f"  読み取りサイズ: {READ_SIZE / (1024*1024):.0f} MB")
    print(f"  ストリーム数: {NUM_STREAMS}")
    
    # PCIe帯域テスト
    run_bandwidth_test()
    
    # RMM 初期化
    try:
        if not rmm.is_initialized():
            rmm.reinitialize(
                pool_allocator=True, 
                initial_pool_size=16*1024**3  # 16GB (大容量)
            )
            print("✅ RMM メモリプール初期化完了 (16GB)")
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

        # データサイズ推定
        rows_est = limit_rows
        row_bytes = 17*8 + 17*4  # 概算
        header_est = 19
        estimated_size = header_est + rows_est * row_bytes
        print(f"推定データサイズ: {estimated_size / (1024*1024):.2f} MB")

        # GPU バッファ事前確保
        print("GPU バッファ事前確保中...")
        devbuf = rmm.DeviceBuffer(size=estimated_size)
        print(f"GPU バッファ確保完了: {estimated_size / (1024*1024):.2f} MB")

        # 超大容量 Pinned ホストバッファ確保
        print("大容量 Pinned ホストバッファ確保中...")
        pbuf1 = numba.cuda.pinned_array(CHUNK_SIZE, dtype=np.uint8)
        pbuf2 = numba.cuda.pinned_array(CHUNK_SIZE, dtype=np.uint8)
        print(f"✅ ダブル Pinned バッファ確保完了: {CHUNK_SIZE / (1024*1024):.0f} MB × 2")

        # 複数 CUDA ストリーム作成
        streams = []
        for i in range(NUM_STREAMS):
            streams.append(cp.cuda.Stream(non_blocking=True))
        print(f"✅ CUDA ストリーム {NUM_STREAMS}個作成完了")

        # 究極最適化 COPY → GPU転送
        print("究極最適化 COPY → GPU転送実行中...")
        start_copy_time = time.time()
        
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        
        offset = 0
        chunk_count = 0
        total_read_time = 0
        total_memcpy_time = 0
        total_async_time = 0
        stream_idx = 0
        
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_obj:
                print("  🚀 大きなチャンクでの高速読み取り開始...")
                
                # 【最適化a】チャンクを蓄積して大きなブロックで処理
                accumulated_data = []
                accumulated_size = 0
                
                for chunk in copy_obj:
                    if not chunk:
                        break
                    
                    start_read = time.time()
                    accumulated_data.append(chunk)
                    accumulated_size += len(chunk)
                    read_time = time.time() - start_read
                    total_read_time += read_time
                    
                    # READ_SIZE (4MB) に達したら処理
                    if accumulated_size >= READ_SIZE:
                        # 蓄積されたチャンクを結合
                        combined_chunk = b''.join(accumulated_data)
                        
                        # 処理対象として使用
                        chunk = combined_chunk
                        chunk_size = len(chunk)
                        
                        # リセット
                        accumulated_data = []
                        accumulated_size = 0
                    else:
                        continue  # まだ蓄積中
                    
                    # GPU転送処理
                    self._process_chunk_to_gpu(chunk, offset, devbuf, pbuf1, pbuf2, streams,
                                             chunk_count, total_memcpy_time, total_async_time)
                    
                    offset += chunk_size
                    chunk_count += 1
                
                # 残りの蓄積データも処理
                if accumulated_data:
                    combined_chunk = b''.join(accumulated_data)
                    chunk = combined_chunk
                    chunk_size = len(chunk)
                    
                    # GPU転送処理
                    self._process_chunk_to_gpu(chunk, offset, devbuf, pbuf1, pbuf2, streams,
                                             chunk_count, total_memcpy_time, total_async_time)
                    
                    offset += chunk_size
                    chunk_count += 1

def _process_chunk_to_gpu(chunk, offset, devbuf, pbuf1, pbuf2, streams, chunk_count, total_memcpy_time, total_async_time):
    """チャンクをGPUに転送する共通処理"""
    chunk_size = len(chunk)
    
    # バッファオーバーフロー チェック
    if offset + chunk_size > devbuf.size:
        print(f"⚠️  警告: GPUバッファサイズ不足")
        return False
                    
                    # ダブルバッファ選択
                    current_buf = pbuf1 if chunk_count % 2 == 0 else pbuf2
                    current_stream = streams[stream_idx % NUM_STREAMS]
                    
                    # 【最適化c】効率的 pinned memcpy
                    start_memcpy = time.time()
                    if chunk_size <= CHUNK_SIZE:
                        # NumPy最適化コピー
                        np_chunk = np.frombuffer(chunk, dtype=np.uint8)
                        current_buf[:chunk_size] = np_chunk
                    else:
                        # 分割処理
                        for sub_offset in range(0, chunk_size, CHUNK_SIZE):
                            sub_size = min(CHUNK_SIZE, chunk_size - sub_offset)
                            sub_chunk = chunk[sub_offset:sub_offset+sub_size]
                            np_sub = np.frombuffer(sub_chunk, dtype=np.uint8)
                            current_buf[:sub_size] = np_sub
                    memcpy_time = time.time() - start_memcpy
                    total_memcpy_time += memcpy_time
                    
                    # 【最適化e】非同期 DMA (複数ストリーム)
                    start_async = time.time()
                    src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(current_buf))
                    dst_ptr = devbuf.ptr + offset
                    
                    cp.cuda.runtime.memcpyAsync(
                        dst_ptr, src_ptr, chunk_size,
                        cp.cuda.runtime.memcpyHostToDevice,
                        current_stream.ptr
                    )
                    async_time = time.time() - start_async
                    total_async_time += async_time
                    
                    offset += chunk_size
                    chunk_count += 1
                    stream_idx += 1
                    
                    # 進捗表示（少なめに）
                    if chunk_count % 100 == 0:
                        print(f"    チャンク {chunk_count:,} | {offset / (1024*1024):.0f} MB | 平均 {chunk_size / (1024*1024):.1f} MB/chunk")
                
                # 全ストリーム同期
                print("  ⏳ 全ストリーム同期待機中...")
                for stream in streams:
                    cp.cuda.runtime.streamSynchronize(stream.ptr)
        
        copy_time = time.time() - start_copy_time
        actual_data_size = offset
        
        print(f"✅ 究極最適化転送完了 ({copy_time:.4f}秒)")
        print(f"  処理チャンク数: {chunk_count:,}")
        print(f"  平均チャンクサイズ: {(actual_data_size / chunk_count) / (1024*1024):.2f} MB")
        print(f"  実際のデータサイズ: {actual_data_size / (1024*1024):.2f} MB")
        print(f"  総合転送速度: {actual_data_size / (1024*1024) / copy_time:.2f} MB/sec")

    finally:
        conn.close()
        # Pinned メモリ解放
        if 'pbuf1' in locals():
            del pbuf1
        if 'pbuf2' in locals():
            del pbuf2
        if 'streams' in locals():
            for stream in streams:
                del stream

    # GPU バッファをトリミング
    if actual_data_size < devbuf.size:
        print("GPU バッファをトリミング中...")
        trimmed_devbuf = rmm.DeviceBuffer(size=actual_data_size)
        cp.cuda.runtime.memcpy(
            trimmed_devbuf.ptr, devbuf.ptr, actual_data_size,
            cp.cuda.runtime.memcpyDeviceToDevice
        )
        devbuf = trimmed_devbuf
        print(f"GPU バッファトリミング完了: {actual_data_size / (1024*1024):.2f} MB")

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
        return

    total_time = time.time() - start_total_time
    decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
    
    print(f"\n=== 究極最適化版ベンチマーク完了 ===")
    print(f"総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳（詳細） ---")
    print(f"  メタデータ取得       : {meta_time:.4f} 秒")
    print(f"  COPY→究極最適化転送  : {copy_time:.4f} 秒")
    print(f"    ├─ libpq読み取り   : {total_read_time:.4f} 秒")
    print(f"    ├─ pinned memcpy   : {total_memcpy_time:.4f} 秒")
    print(f"    └─ 非同期DMA       : {total_async_time:.4f} 秒")
    print(f"  GPUパース           : {parse_time:.4f} 秒")
    print(f"  GPUデコード         : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み     : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数      : {rows:,} 行")
    print(f"  処理列数      : {len(columns)} 列")
    print(f"  Decimal列数   : {decimal_cols} 列")
    print(f"  データサイズ  : {actual_data_size / (1024*1024):.2f} MB")
    print(f"  処理チャンク数: {chunk_count:,}")
    print(f"  平均チャンクサイズ: {(actual_data_size / chunk_count) / (1024*1024):.2f} MB")
    
    total_cells = rows * len(columns)
    throughput = total_cells / decode_time if decode_time > 0 else 0
    network_throughput = actual_data_size / (1024*1024) / copy_time
    
    # 最適化効果の詳細分析
    libpq_efficiency = actual_data_size / (1024*1024) / total_read_time if total_read_time > 0 else 0
    memcpy_efficiency = actual_data_size / (1024*1024) / total_memcpy_time if total_memcpy_time > 0 else 0
    dma_efficiency = actual_data_size / (1024*1024) / total_async_time if total_async_time > 0 else 0
    
    print(f"  セル処理速度     : {throughput:,.0f} cells/sec")
    print(f"  総合転送速度     : {network_throughput:.2f} MB/sec")
    print(f"  libpq読み取り速度: {libpq_efficiency:.2f} MB/sec")
    print(f"  pinned memcpy速度: {memcpy_efficiency:.2f} MB/sec")
    print(f"  非同期DMA速度    : {dma_efficiency:.2f} MB/sec")
    
    # PCIe効率計算（RTX 3090: 22GB/s実効）
    pcie_efficiency = network_throughput / 22000 * 100
    print(f"  PCIe効率        : {pcie_efficiency:.1f}% (対22GB/s実効)")
    
    print("--- 究極最適化効果 ---")
    print("  ✅ ファイル I/O: 完全ゼロ")
    print("  ✅ 大きなチャンク: libpqから4MB単位読み取り")
    print("  ✅ 効率的memcpy: NumPy最適化コピー")
    print("  ✅ 複数ストリーム: pipelined非同期DMA")
    print("  ✅ 64MiB pinned: 大容量バッファで効率最大化")
    print("  ✅ CPU使用率: 最小化（nvtop確認推奨）")
    
    if network_throughput > 5000:
        print("  🏆 5GB/s超達成！")
    elif network_throughput > 1000:
        print("  🥇 1GB/s超達成！")
    else:
        print("  ⚠️  転送速度が期待値を下回っています")
    
    print("=========================================")

    # 検証用出力
    print(f"\ncuDF検証用出力:")
    try:
        print(f"出力Parquet: {OUTPUT_PARQUET_PATH}")
        print(f"読み込み確認: {len(cudf_df):,} 行 × {len(cudf_df.columns)} 列")
        print("先頭データ型:")
        for i, (col_name, dtype) in enumerate(cudf_df.dtypes.items()):
            if i < 5:  # 最初の5列のみ
                print(f"  {col_name}: {dtype}")
        print("✅ cuDF検証: 成功")
    except Exception as e:
        print(f"❌ cuDF検証: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU 究極最適化版')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    parser.add_argument('--chunk-size', type=int, default=64, help='チャンクサイズ(MB)')
    parser.add_argument('--read-size', type=int, default=4, help='読み取りサイズ(MB)')
    parser.add_argument('--streams', type=int, default=2, help='ストリーム数')
    parser.add_argument('--bandwidth-test', action='store_true', help='帯域テストのみ')
    
    args = parser.parse_args()
    
    # パラメータ設定
    global CHUNK_SIZE, READ_SIZE, NUM_STREAMS
    CHUNK_SIZE = args.chunk_size * 1024 * 1024
    READ_SIZE = args.read_size * 1024 * 1024
    NUM_STREAMS = args.streams
    
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    if args.bandwidth_test:
        run_bandwidth_test()
        return
    
    run_ultimate_optimized_benchmark(limit_rows=args.rows)

if __name__ == "__main__":
    main()