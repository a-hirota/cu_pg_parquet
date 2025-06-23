"""
PostgreSQL → Rust → GPU 統一処理版
cuDF DataFrame作成のCPUシングルスレッド問題を解決

改善内容:
1. cuda.to_device()で直接GPU転送
2. postgresql_to_cudf_parquet()で一括処理
3. benchmark_lineorder_5m.pyと同じ効率的な処理フロー
"""

import os
import time
import subprocess
import json
import numpy as np
from numba import cuda
import rmm
from pathlib import Path
from typing import List, Dict, Any

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
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


def process_chunk_unified(chunk_info: dict, columns: List[ColumnMeta]) -> tuple:
    """チャンクを統一処理（benchmark_lineorder_5m.pyと同じアプローチ）"""
    chunk_id = chunk_info['chunk_id']
    chunk_file = chunk_info['chunk_file']
    file_size = chunk_info['file_size']
    
    print(f"\n[GPU] チャンク {chunk_id + 1} 統一処理開始")
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
        
        # 統一処理（postgresql_to_cudf_parquet）
        process_start = time.time()
        chunk_output = f"benchmark/chunk_{chunk_id}_unified.parquet"
        
        cudf_df, detailed_timing = postgresql_to_cudf_parquet(
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
        decode_time = detailed_timing.get('decode_and_export', 0)
        cudf_time = detailed_timing.get('cudf_creation', 0)
        write_time = detailed_timing.get('parquet_export', 0)
        
        print(f"[GPU] チャンク {chunk_id + 1} 処理完了:")
        print(f"  - 処理行数: {rows:,} 行")
        print(f"  - GPU全体時間: {gpu_time:.2f}秒")
        print(f"  - 内訳:")
        print(f"    - ファイル読込: {read_time:.2f}秒")
        print(f"    - GPU転送: {transfer_time:.2f}秒")
        print(f"    - GPUパース: {parse_time:.2f}秒")
        print(f"    - デコード+cuDF: {decode_time:.2f}秒 (cuDF作成: {cudf_time:.2f}秒)")
        print(f"    - Parquet書込: {write_time:.2f}秒")
        print(f"  - スループット: {file_size / gpu_time / 1024**3:.2f} GB/秒")
        
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
    print("=== PostgreSQL → Rust → GPU 統一処理版 ===")
    print(f"チャンク数: {TOTAL_CHUNKS}")
    print(f"各チャンクサイズ: 約{52.86 / TOTAL_CHUNKS:.1f} GB")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print("\n改善内容:")
    print("  - cuda.to_device()で直接GPU転送")
    print("  - postgresql_to_cudf_parquet()で一括処理")
    print("  - cuDF DataFrame作成のCPUシングルスレッド問題を解決")
    
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
    
    try:
        # 各チャンクを順次処理
        columns = None
        for chunk_id in range(TOTAL_CHUNKS):
            chunk_info = process_single_chunk(chunk_id)
            total_rust_time += chunk_info['rust_time']
            total_size += chunk_info['file_size']
            
            # 初回のみカラム情報を取得
            if chunk_id == 0 and chunk_info['columns']:
                columns = []
                for col in chunk_info['columns']:
                    pg_oid = col['pg_oid']
                    arrow_info = PG_OID_TO_ARROW.get(pg_oid, (UNKNOWN, None))
                    arrow_id, elem_size = arrow_info
                    
                    column_meta = ColumnMeta(
                        name=col['name'],
                        pg_oid=pg_oid,
                        pg_typmod=-1,
                        arrow_id=arrow_id,
                        elem_size=elem_size if elem_size is not None else 0
                    )
                    columns.append(column_meta)
                print(f"カラム数: {len(columns)}")
            
            if not columns:
                raise RuntimeError("カラム情報が取得できませんでした")
            
            # GPU統一処理
            gpu_time, _, rows, timing = process_chunk_unified(chunk_info, columns)
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
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_files()


if __name__ == "__main__":
    main()