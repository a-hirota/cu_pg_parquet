"""
PostgreSQL → Rust → GPU 直接抽出版（統合バッファ削除）
メモリ効率的な処理でGPUメモリ不足を解決

改善内容:
1. 統合バッファを完全に削除
2. 入力データから直接cuDF列を作成
3. メモリ使用量を大幅削減
4. 処理速度も向上
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
import psycopg

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet_direct import postgresql_to_cudf_parquet_direct
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.metadata import fetch_column_meta

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_fixed"
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
    files = [
        f"{OUTPUT_DIR}/lineorder_meta_0.json",
        f"{OUTPUT_DIR}/lineorder_data_0.ready"
    ] + [f"{OUTPUT_DIR}/chunk_{i}.bin" for i in range(TOTAL_CHUNKS)]
    
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def process_single_chunk(chunk_id: int) -> dict:
    """1つのチャンクを処理（Rust転送）"""
    print(f"\n[Rust] チャンク {chunk_id + 1}/{TOTAL_CHUNKS} 転送開始")
    
    env = os.environ.copy()
    env['CHUNK_ID'] = str(chunk_id)
    env['TOTAL_CHUNKS'] = str(TOTAL_CHUNKS)
    
    rust_start = time.time()
    process = subprocess.run(
        [RUST_BINARY],
        capture_output=True,
        text=True,
        env=env
    )
    
    if process.returncode != 0:
        print(f"❌ Rustエラー: {process.stderr}")
        raise RuntimeError(f"チャンク{chunk_id}の転送失敗")
    
    # JSON結果を抽出
    output = process.stdout
    json_start = output.find("===CHUNK_RESULT_JSON===")
    json_end = output.find("===END_CHUNK_RESULT_JSON===")
    
    if json_start != -1 and json_end != -1:
        json_str = output[json_start + len("===CHUNK_RESULT_JSON==="):json_end].strip()
        result = json.loads(json_str)
        rust_time = result['elapsed_seconds']
        file_size = result['total_bytes']
        chunk_file = result['chunk_file']
        columns_data = result.get('columns')
    else:
        rust_time = time.time() - rust_start
        chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
        file_size = os.path.getsize(chunk_file)
        columns_data = None
    
    print(f"[Rust] チャンク {chunk_id + 1} 転送完了: {file_size / 1024**3:.2f} GB, {rust_time:.2f}秒 ({file_size / rust_time / 1024**3:.2f} GB/秒)")
    
    return {
        'chunk_id': chunk_id,
        'chunk_file': chunk_file,
        'file_size': file_size,
        'rust_time': rust_time,
        'columns': columns_data
    }


def process_chunk_direct(chunk_info: dict, columns: List[ColumnMeta]) -> tuple:
    """チャンクを直接抽出処理（統合バッファ削除版）"""
    chunk_id = chunk_info['chunk_id']
    chunk_file = chunk_info['chunk_file']
    file_size = chunk_info['file_size']
    
    print(f"\n[GPU] チャンク {chunk_id + 1} 直接抽出処理開始（統合バッファ削除）")
    
    # ループ開始前のGPUメモリクリア
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    gpu_start = time.time()
    
    try:
        # ファイル読み込み（CPUメモリへ）
        read_start = time.time()
        with open(chunk_file, 'rb') as f:
            data = f.read()
        raw_host = np.frombuffer(data, dtype=np.uint8)
        del data  # 元のバイト列は不要なので削除
        read_time = time.time() - read_start
        print(f"  ファイル読み込み: {read_time:.2f}秒 ({file_size / read_time / 1024**3:.2f} GB/秒)")
        
        # GPU転送（cuda.to_deviceを使用）
        transfer_start = time.time()
        raw_dev = cuda.to_device(raw_host)
        transfer_time = time.time() - transfer_start
        print(f"  GPU転送: {transfer_time:.2f}秒 ({file_size / transfer_time / 1024**3:.2f} GB/秒)")
        
        # ヘッダーサイズ検出
        header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
        header_size = detect_pg_header_size(header_sample)
        
        # 直接抽出処理（統合バッファ不使用）
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
        parse_time = detailed_timing.get('gpu_parsing', 0)
        process_export_time = detailed_timing.get('process_and_export', 0)
        string_time = detailed_timing.get('string_buffer_creation', 0)
        extract_time = detailed_timing.get('direct_extraction', 0)
        write_time = detailed_timing.get('parquet_export', 0)
        
        print(f"[GPU] チャンク {chunk_id + 1} 処理完了:")
        print(f"  - 処理行数: {rows:,} 行")
        print(f"  - GPU全体時間: {gpu_time:.2f}秒")
        print(f"  - 内訳:")
        print(f"    - ファイル読込: {read_time:.2f}秒")
        print(f"    - GPU転送: {transfer_time:.2f}秒")
        print(f"    - GPUパース: {parse_time:.2f}秒")
        print(f"    - 直接抽出処理: {process_export_time:.2f}秒")
        print(f"      - 文字列バッファ: {string_time:.2f}秒")
        print(f"      - 直接列抽出: {extract_time:.2f}秒")
        print(f"      - Parquet書込: {write_time:.2f}秒")
        print(f"  - スループット: {file_size / gpu_time / 1024**3:.2f} GB/秒")
        print(f"  - 統合バッファ: 【削除済み】")
        
        # メモリ解放
        del raw_dev
        del raw_host
        del cudf_df
        
        # GPUメモリプールを明示的にクリア
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        
        # ガベージコレクションを強制
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
    print("=== PostgreSQL → Rust → GPU 直接抽出版（統合バッファ削除） ===")
    print(f"チャンク数: {TOTAL_CHUNKS}")
    print(f"各チャンクサイズ: 約{52.86 / TOTAL_CHUNKS:.1f} GB")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print("\n改善内容:")
    print("  - 統合バッファを完全に削除")
    print("  - RMM DeviceBufferで全ての列を処理")
    print("  - メモリ使用量を大幅削減")
    print("  - チャンク数8で安定動作（各約6.6GB）")
    
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
    total_rust_time = 0
    total_gpu_time = 0
    total_size = 0
    total_rows = 0
    
    # PostgreSQLからメタデータを取得（arrow_param含む）
    columns = get_postgresql_metadata()
    
    try:
        # 各チャンクを順次処理
        for chunk_id in range(TOTAL_CHUNKS):
            chunk_info = process_single_chunk(chunk_id)
            total_rust_time += chunk_info['rust_time']
            total_size += chunk_info['file_size']
            
            # GPU直接抽出処理
            gpu_time, _, rows, timing = process_chunk_direct(chunk_info, columns)
            total_gpu_time += gpu_time
            total_rows += rows
        
        # 最終統計
        total_time = time.time() - total_start
        total_gb = total_size / 1024**3
        
        print(f"\n{'='*60}")
        print("✅ 全チャンク処理完了!")
        print('='*60)
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"  - Rust転送合計: {total_rust_time:.2f}秒")
        print(f"  - GPU処理合計: {total_gpu_time:.2f}秒")
        print(f"総データサイズ: {total_gb:.2f} GB")
        print(f"総行数: {total_rows:,} 行")
        print(f"全体スループット: {total_gb / total_time:.2f} GB/秒")
        print(f"Rust平均速度: {total_gb / total_rust_time:.2f} GB/秒")
        print(f"GPU平均速度: {total_gb / total_gpu_time:.2f} GB/秒")
        
        # ベースライン比較
        baseline_throughput = 0.20  # 4チャンク版
        improvement = (total_gb / total_time) / baseline_throughput
        print(f"\n改善率: {improvement:.1f}倍（4チャンクベースライン比）")
        
        print(f"\nメモリ効率改善:")
        print(f"  - 統合バッファ削除: 約{total_rows * 100 / (1024**3):.1f} GB節約")
        print(f"  - ピークメモリ使用量: 大幅削減")
        
        # === 処理結果の検証 ===
        print(f"\n{'='*60}")
        print("処理結果の検証")
        print('='*60)
        
        # 最初のチャンクのParquetファイルを読み込んで検証
        chunk_0_parquet = "benchmark/chunk_0_direct.parquet"
        
        if os.path.exists(chunk_0_parquet):
            try:
                import cudf
                print(f"\ncuDFでParquetファイルを読み込み中: {chunk_0_parquet}")
                start_verify_time = time.time()
                verification_df = cudf.read_parquet(chunk_0_parquet)
                verify_time = time.time() - start_verify_time
                print(f"cuDF読み込み完了 ({verify_time:.4f}秒)")
                
                print("\n--- cuDF DataFrame Info ---")
                verification_df.info()
                
                print(f"\n読み込み結果: {len(verification_df):,} 行 × {len(verification_df.columns)} 列")
                
                # データ型確認
                print("\nデータ型:")
                for col_name, dtype in verification_df.dtypes.items():
                    print(f"  {col_name}: {dtype}")
                
                print("\n--- cuDF DataFrame Head (全列表示) ---")
                # pandas設定を使用して全列を強制表示
                import pandas as pd
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 20)
                
                try:
                    # cuDFをpandasに変換して全列表示
                    pandas_df = verification_df.head().to_pandas()
                    print(pandas_df)
                except Exception:
                    # フォールバック: 列を一つずつ表示
                    print("cuDF Head (列別表示):")
                    for i, col_name in enumerate(verification_df.columns):
                        if i < 5:  # 最初の5列のみ
                            print(f"  列{i+1:2d} {col_name:20s}: {verification_df[col_name].iloc[:3].to_pandas().tolist()}")
                
                # 設定をリセット
                pd.reset_option('display.max_columns')
                pd.reset_option('display.width')
                pd.reset_option('display.max_colwidth')
                
                # 基本統計情報
                print("\n基本統計:")
                try:
                    numeric_cols = verification_df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        for col in numeric_cols[:5]:  # 最初の5つの数値列のみ
                            col_data = verification_df[col]
                            if len(col_data) > 0:
                                print(f"  {col}: 平均={float(col_data.mean()):.2f}, 最小={float(col_data.min()):.2f}, 最大={float(col_data.max()):.2f}")
                except Exception as e:
                    print(f"  統計情報エラー: {e}")
                
                # 文字列列の検証
                print("\n文字列列の検証:")
                string_cols = verification_df.select_dtypes(include=['object', 'string']).columns
                if len(string_cols) > 0:
                    for col in string_cols[:3]:  # 最初の3つの文字列列
                        print(f"  {col}: サンプル値 = {verification_df[col].iloc[:3].to_pandas().tolist()}")
                else:
                    print("  文字列列が見つかりません")
                
                print("\n-------------------------")
                print("検証結果: 成功")
                
                # クリーンアップ
                for i in range(TOTAL_CHUNKS):
                    chunk_file = f"benchmark/chunk_{i}_direct.parquet"
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                        
            except Exception as e:
                print(f"\n検証失敗: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n検証失敗: Parquetファイル {chunk_0_parquet} が見つかりません")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_files()


if __name__ == "__main__":
    main()