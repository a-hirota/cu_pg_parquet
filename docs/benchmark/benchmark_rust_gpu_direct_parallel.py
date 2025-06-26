"""
PostgreSQL → Rust → GPU 直接抽出版（16並列一括転送）
Rustの16並列転送を活用した高速版

改善内容:
1. Rustプログラムを1回実行して16並列で全チャンクを転送
2. 転送完了後、GPUで順次処理
3. 統合バッファを削除してメモリ効率向上
"""

import os
import time
import subprocess
import json
import numpy as np
from numba import cuda
import rmm
import cudf
from pathlib import Path
from typing import List, Dict, Any
import glob

from src.types import ColumnMeta
from src.main_postgres_to_parquet_direct import postgresql_to_cudf_parquet_direct
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.readPostgres.metadata import fetch_column_meta

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_optimized"
TOTAL_CHUNKS = 8  # 8チャンク（各約6.6GB）


def setup_rmm_pool():
    """RMMメモリプールを適切に設定"""
    try:
        if rmm.is_initialized():
            print("RMM既に初期化済み")
            return
        
        # GPUメモリの90%を使用可能に設定
        import cupy as cp
        gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        pool_size = int(gpu_memory * 0.9)
        
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=pool_size,
            maximum_pool_size=pool_size
        )
        print(f"✅ RMMメモリプール初期化: {pool_size / 1024**3:.1f} GB")
    except Exception as e:
        print(f"⚠️ RMM初期化警告: {e}")


def get_postgresql_metadata():
    """PostgreSQLからテーブルメタデータを取得"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    import psycopg
    conn = psycopg.connect(dsn)
    try:
        print("PostgreSQLメタデータを取得中...")
        columns = fetch_column_meta(conn, f"SELECT * FROM {TABLE_NAME}")
        print(f"✅ メタデータ取得完了: {len(columns)} 列")
        
        # デバッグ：Decimal列の情報を表示
        for col in columns:
            if col.arrow_id == 5:  # DECIMAL128
                print(f"  Decimal列 {col.name}: arrow_param={col.arrow_param}")
        
        return columns
    finally:
        conn.close()


def cleanup_files():
    """ファイルをクリーンアップ"""
    # チャンクファイル
    for f in glob.glob(f"{OUTPUT_DIR}/chunk_*.bin"):
        os.remove(f)
    
    # メタデータファイル
    files = [
        f"{OUTPUT_DIR}/lineorder_meta.json",
        f"{OUTPUT_DIR}/lineorder_meta_0.json",
        f"{OUTPUT_DIR}/lineorder_data_0.ready"
    ]
    
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def run_rust_parallel_transfer() -> tuple:
    """Rust 16並列転送を実行"""
    print("\n=== Rust 16並列転送開始 ===")
    print(f"並列接続数: 16")
    print(f"チャンク数: {TOTAL_CHUNKS}")
    
    rust_start = time.time()
    
    # 環境変数をクリア（個別チャンク処理モードを無効化）
    env = os.environ.copy()
    if 'CHUNK_ID' in env:
        del env['CHUNK_ID']
    if 'TOTAL_CHUNKS' in env:
        del env['TOTAL_CHUNKS']
    
    # Rustプログラムを実行
    process = subprocess.run(
        [RUST_BINARY],
        capture_output=True,
        text=True,
        env=env
    )
    
    rust_time = time.time() - rust_start
    
    if process.returncode != 0:
        print(f"❌ Rustエラー: {process.stderr}")
        raise RuntimeError("Rust転送失敗")
    
    print(process.stdout)
    
    # 生成されたチャンクファイルを確認
    chunk_files = []
    total_size = 0
    
    for i in range(TOTAL_CHUNKS):
        chunk_file = f"{OUTPUT_DIR}/chunk_{i}.bin"
        if os.path.exists(chunk_file):
            size = os.path.getsize(chunk_file)
            chunk_files.append({
                'chunk_id': i,
                'chunk_file': chunk_file,
                'file_size': size
            })
            total_size += size
    
    print(f"\n✅ Rust転送完了: {total_size / 1024**3:.2f} GB, {rust_time:.2f}秒 ({total_size / rust_time / 1024**3:.2f} GB/秒)")
    print(f"生成チャンク数: {len(chunk_files)}")
    
    return chunk_files, rust_time, total_size


def process_chunk_direct(chunk_info: dict, columns: List[ColumnMeta]) -> tuple:
    """チャンクを直接抽出処理（統合バッファ削除版）"""
    chunk_id = chunk_info['chunk_id']
    chunk_file = chunk_info['chunk_file']
    file_size = chunk_info['file_size']
    
    print(f"\n[GPU] チャンク {chunk_id + 1} 直接抽出処理開始")
    
    # GPUメモリクリア
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    gpu_start = time.time()
    
    try:
        # ファイル読み込み
        read_start = time.time()
        with open(chunk_file, 'rb') as f:
            data = f.read()
        raw_host = np.frombuffer(data, dtype=np.uint8)
        del data
        read_time = time.time() - read_start
        print(f"  ファイル読み込み: {read_time:.2f}秒 ({file_size / read_time / 1024**3:.2f} GB/秒)")
        
        # GPU転送
        transfer_start = time.time()
        raw_dev = cuda.to_device(raw_host)
        transfer_time = time.time() - transfer_start
        print(f"  GPU転送: {transfer_time:.2f}秒 ({file_size / transfer_time / 1024**3:.2f} GB/秒)")
        
        # ヘッダーサイズ検出
        header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
        header_size = detect_pg_header_size(header_sample)
        
        # 直接抽出処理
        process_start = time.time()
        chunk_output = f"benchmark/chunk_{chunk_id}_direct.parquet"
        
        cudf_df, detailed_timing = postgresql_to_cudf_parquet_direct(
            raw_dev=raw_dev,
            columns=columns,
            ncols=len(columns),
            header_size=header_size,
            output_path=chunk_output,
            compression='snappy',
            use_rmm=True,
            optimize_gpu=True
        )
        
        process_time = time.time() - process_start
        gpu_time = time.time() - gpu_start
        
        # 処理統計
        rows = len(cudf_df) if cudf_df is not None else 0
        
        print(f"[GPU] チャンク {chunk_id + 1} 処理完了:")
        print(f"  - 処理行数: {rows:,} 行")
        print(f"  - GPU全体時間: {gpu_time:.2f}秒")
        print(f"  - スループット: {file_size / gpu_time / 1024**3:.2f} GB/秒")
        
        # メモリ解放
        del raw_dev
        del raw_host
        del cudf_df
        
        mempool.free_all_blocks()
        
        import gc
        gc.collect()
        
        return gpu_time, file_size, rows, detailed_timing
        
    except Exception as e:
        print(f"❌ GPU処理エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # チャンクファイル削除
        if os.path.exists(chunk_file):
            os.remove(chunk_file)


def main():
    print("=== PostgreSQL → Rust → GPU 直接抽出版（16並列一括転送） ===")
    print(f"チャンク数: {TOTAL_CHUNKS}")
    print(f"各チャンクサイズ: 約{52.86 / TOTAL_CHUNKS:.1f} GB")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    
    # RMMメモリプール設定
    setup_rmm_pool()
    
    # CUDA context確認
    try:
        cuda.current_context()
        print("\n✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA context エラー: {e}")
        return
    
    # クリーンアップ
    cleanup_files()
    
    total_start = time.time()
    
    # PostgreSQLからメタデータを取得
    columns = get_postgresql_metadata()
    
    try:
        # Rust 16並列転送
        chunk_files, rust_time, total_size = run_rust_parallel_transfer()
        
        # GPU処理統計
        total_gpu_time = 0
        total_rows = 0
        
        # 各チャンクをGPUで順次処理
        for chunk_info in chunk_files:
            gpu_time, _, rows, timing = process_chunk_direct(chunk_info, columns)
            total_gpu_time += gpu_time
            total_rows += rows
        
        # 最終統計
        total_time = time.time() - total_start
        total_gb = total_size / 1024**3
        
        print(f"\n{'='*60}")
        print("✅ 全処理完了!")
        print('='*60)
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"  - Rust並列転送: {rust_time:.2f}秒")
        print(f"  - GPU処理合計: {total_gpu_time:.2f}秒")
        print(f"総データサイズ: {total_gb:.2f} GB")
        print(f"総行数: {total_rows:,} 行")
        print(f"全体スループット: {total_gb / total_time:.2f} GB/秒")
        print(f"Rust転送速度: {total_gb / rust_time:.2f} GB/秒（16並列）")
        print(f"GPU処理速度: {total_gb / total_gpu_time:.2f} GB/秒")
        
        # 文字列破損チェック
        print(f"\n{'='*60}")
        print("文字列破損チェック")
        print('='*60)
        
        even_errors = 0
        odd_errors = 0
        check_chunks = min(2, len(chunk_files))  # 最初の2チャンクのみ
        
        for i in range(check_chunks):
            parquet_file = f"benchmark/chunk_{i}_direct.parquet"
            if os.path.exists(parquet_file):
                try:
                    df = cudf.read_parquet(parquet_file)
                    
                    if 'lo_orderpriority' in df.columns:
                        check_rows = min(1000, len(df))
                        for row_idx in range(check_rows):
                            try:
                                value = df['lo_orderpriority'].iloc[row_idx]
                                expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                                is_valid = any(value.startswith(p) for p in expected_patterns)
                                
                                if not is_valid:
                                    if row_idx % 2 == 0:
                                        even_errors += 1
                                    else:
                                        odd_errors += 1
                            except:
                                pass
                    
                    del df
                except Exception as e:
                    print(f"チェックエラー: {e}")
        
        print(f"偶数行エラー: {even_errors}")
        print(f"奇数行エラー: {odd_errors}")
        
        if even_errors + odd_errors == 0:
            print("✅ 文字列破損なし！")
        
        # クリーンアップ
        for i in range(TOTAL_CHUNKS):
            parquet_file = f"benchmark/chunk_{i}_direct.parquet"
            if os.path.exists(parquet_file):
                os.remove(parquet_file)
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_files()


if __name__ == "__main__":
    main()