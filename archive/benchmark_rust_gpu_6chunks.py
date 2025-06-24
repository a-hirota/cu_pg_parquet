"""
PostgreSQL → Rust → GPU 6チャンク版
最適なバランス: 各チャンク約8.8GB

改善内容:
1. 6チャンクでGPUメモリ効率と処理効率のバランス
2. パイプライン処理で並列化
3. より詳細なパフォーマンス統計
"""

import os
import time
import subprocess
import json
import cupy as cp
import numpy as np
import mmap
import rmm
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
TOTAL_CHUNKS = 6  # 6チャンク（各約8.8GB）


def setup_rmm_pool():
    """RMMメモリプールを適切に設定"""
    try:
        if rmm.is_initialized():
            print("RMM既に初期化済み")
            return
        
        # GPUメモリの90%を使用可能に設定
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
    """1つのチャンクを処理"""
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


def process_chunk_on_gpu(chunk_info: dict, columns: List[ColumnMeta]) -> tuple:
    """チャンクをGPUで処理"""
    chunk_id = chunk_info['chunk_id']
    chunk_file = chunk_info['chunk_file']
    file_size = chunk_info['file_size']
    
    print(f"\n[GPU] チャンク {chunk_id + 1} 処理開始")
    gpu_start = time.time()
    
    # GPUメモリ状況確認
    mempool = cp.get_default_memory_pool()
    print(f"  GPUメモリプール: 使用 {mempool.used_bytes() / 1024**3:.2f} GB / 総計 {mempool.total_bytes() / 1024**3:.2f} GB")
    
    try:
        # mmapを使用してチャンク全体を読み込み
        transfer_start = time.time()
        with open(chunk_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # チャンク全体をGPUに転送
                gpu_data = cp.asarray(np.frombuffer(mm, dtype=np.uint8))
        
        transfer_time = time.time() - transfer_start
        print(f"  GPU転送完了: {transfer_time:.2f}秒 ({file_size / transfer_time / 1024**3:.2f} GB/秒)")
        
        # ヘッダーサイズ検出
        from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size
        header_size = detect_pg_header_size(gpu_data[:100].get())
        
        # PostgreSQL → Parquet変換
        postgres_start = time.time()
        chunk_output = f"benchmark/chunk_{chunk_id}_6chunks.parquet"
        
        df, metrics = postgresql_to_cudf_parquet(
            raw_dev=gpu_data,
            columns=columns,
            ncols=len(columns),
            header_size=header_size,
            output_path=chunk_output
        )
        
        postgres_time = time.time() - postgres_start
        gpu_time = time.time() - gpu_start
        
        # 処理統計
        actual_rows = len(df) if df is not None else 0
        print(f"[GPU] チャンク {chunk_id + 1} 処理完了:")
        print(f"  - 処理行数: {actual_rows:,} 行")
        print(f"  - GPU全体時間: {gpu_time:.2f}秒")
        print(f"  - 内訳: 転送 {transfer_time:.1f}秒 + 変換 {postgres_time:.1f}秒")
        print(f"  - スループット: {file_size / gpu_time / 1024**3:.2f} GB/秒")
        
        # メモリ解放
        del gpu_data
        del df
        mempool.free_all_blocks()
        
        return gpu_time, file_size, actual_rows
        
    except Exception as e:
        print(f"❌ GPU処理エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # チャンクファイル削除
        if os.path.exists(chunk_file):
            os.remove(chunk_file)


async def pipeline_processing(columns: List[ColumnMeta]):
    """パイプライン処理（Rust転送とGPU処理を並列化）"""
    print("\n=== パイプライン処理開始 ===")
    
    total_rust_time = 0
    total_gpu_time = 0
    total_size = 0
    total_rows = 0
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 最初のRust転送を開始
        rust_future = executor.submit(process_single_chunk, 0)
        
        for chunk_id in range(TOTAL_CHUNKS):
            # 現在のRust転送完了を待つ
            chunk_info = rust_future.result()
            total_rust_time += chunk_info['rust_time']
            total_size += chunk_info['file_size']
            
            # 次のRust転送を非同期で開始（最後のチャンクでない場合）
            if chunk_id < TOTAL_CHUNKS - 1:
                rust_future = executor.submit(process_single_chunk, chunk_id + 1)
            
            # 現在のチャンクをGPUで処理
            gpu_time, _, rows = process_chunk_on_gpu(chunk_info, columns)
            total_gpu_time += gpu_time
            total_rows += rows
    
    return total_rust_time, total_gpu_time, total_size, total_rows


def main():
    print("=== PostgreSQL → Rust → GPU 6チャンク版 ===")
    print(f"チャンク数: {TOTAL_CHUNKS}")
    print(f"各チャンクサイズ: 約{52.86 / TOTAL_CHUNKS:.1f} GB")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print("\n最適化内容:")
    print("  - 6チャンクでメモリ効率と処理効率のバランス")
    print("  - パイプライン処理で並列化")
    print("  - mmapによるゼロコピー転送")
    
    # RMMメモリプール設定
    setup_rmm_pool()
    
    # CUDA context確認
    print("\n✅ CUDA context OK")
    
    # クリーンアップ
    cleanup_files()
    
    total_start = time.time()
    
    try:
        # 最初のチャンクでカラム情報を取得
        print("\n=== メタデータ取得 ===")
        first_chunk_info = process_single_chunk(0)
        
        if first_chunk_info['columns']:
            columns = []
            for col in first_chunk_info['columns']:
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
        else:
            raise RuntimeError("カラム情報が取得できませんでした")
        
        # 最初のチャンクを処理
        gpu_time, _, rows = process_chunk_on_gpu(first_chunk_info, columns)
        total_rust_time = first_chunk_info['rust_time']
        total_gpu_time = gpu_time
        total_size = first_chunk_info['file_size']
        total_rows = rows
        
        # 残りのチャンクをパイプライン処理
        if TOTAL_CHUNKS > 1:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            rust_time, gpu_time, size, rows = loop.run_until_complete(
                pipeline_processing(columns)
            )
            total_rust_time += rust_time
            total_gpu_time += gpu_time
            total_size += size
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
        print(f"  - 並列化による時間削減: {(total_rust_time + total_gpu_time - total_time):.2f}秒")
        print(f"総データサイズ: {total_gb:.2f} GB")
        print(f"総行数: {total_rows:,} 行")
        print(f"全体スループット: {total_gb / total_time:.2f} GB/秒")
        print(f"Rust平均速度: {total_gb / total_rust_time:.2f} GB/秒")
        print(f"GPU平均速度: {total_gb / total_gpu_time:.2f} GB/秒")
        
        # ベースライン比較（4チャンク版: 52.86GB in 263.49秒 = 0.20 GB/秒）
        baseline_throughput = 0.20
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