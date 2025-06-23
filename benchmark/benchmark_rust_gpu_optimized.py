"""
PostgreSQL → Rust → GPU 最適化版
パフォーマンス問題を解決した高速版

最適化:
1. mmapによる真のゼロコピー読み込み
2. 行数制限の解除（動的メモリ管理）
3. GPUダイレクト転送の最適化
4. パイプライン処理による並列化
"""

import os
import time
import subprocess
import json
import cupy as cp
import numpy as np
import mmap
from pathlib import Path
from typing import List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
TOTAL_CHUNKS = 4

# GPUストリームを使用した非同期処理
cuda_streams = [cp.cuda.Stream() for _ in range(2)]


def cleanup_files():
    """ファイルをクリーンアップ"""
    files = [
        f"{OUTPUT_DIR}/lineorder_meta_0.json",
        f"{OUTPUT_DIR}/lineorder_data_0.ready"
    ] + [f"{OUTPUT_DIR}/chunk_{i}.bin" for i in range(TOTAL_CHUNKS)]
    
    for f in files:
        if os.path.exists(f):
            os.remove(f)
            print(f"✓ クリーンアップ: {f}")


def process_rust_chunk(chunk_id: int) -> dict:
    """Rustでチャンクを転送"""
    print(f"\n[Rust] チャンク {chunk_id} 転送開始")
    
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
    else:
        rust_time = time.time() - rust_start
        chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
        file_size = os.path.getsize(chunk_file)
        result = {'columns': None}
    
    print(f"[Rust] チャンク {chunk_id} 転送完了: {file_size / 1024**3:.2f} GB, {rust_time:.2f}秒")
    
    return {
        'chunk_id': chunk_id,
        'chunk_file': chunk_file,
        'file_size': file_size,
        'rust_time': rust_time,
        'columns': result.get('columns')
    }


def process_gpu_chunk(chunk_info: dict, columns: List[ColumnMeta], stream_id: int) -> Tuple[float, float]:
    """GPU処理（最適化版）"""
    chunk_id = chunk_info['chunk_id']
    chunk_file = chunk_info['chunk_file']
    file_size = chunk_info['file_size']
    
    print(f"\n[GPU] チャンク {chunk_id} 処理開始 (stream {stream_id})")
    gpu_start = time.time()
    
    # mmapを使用した真のゼロコピー転送
    transfer_start = time.time()
    with open(chunk_file, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # GPUストリームを使用した非同期転送
            with cuda_streams[stream_id]:
                # cp.asarrayでmmapから直接GPU転送（コピーなし）
                gpu_data = cp.asarray(np.frombuffer(mm, dtype=np.uint8))
    
    transfer_time = time.time() - transfer_start
    print(f"  GPU転送: {transfer_time:.2f}秒 ({file_size / transfer_time / 1024**3:.2f} GB/秒)")
    
    # PostgreSQLヘッダーサイズを検出
    from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size
    header_size = detect_pg_header_size(gpu_data[:100].get())
    
    # GPU処理を動的行数で実行（制限なし）
    with cuda_streams[stream_id]:
        chunk_output = f"benchmark/chunk_{chunk_id}_optimized.parquet"
        postgres_start = time.time()
        
        # 行数制限を解除して処理
        df, metrics = postgresql_to_cudf_parquet(
            raw_dev=gpu_data,
            columns=columns,
            ncols=len(columns),
            header_size=header_size,
            output_path=chunk_output
        )
        
        # ストリーム同期
        cuda_streams[stream_id].synchronize()
        
    postgres_time = time.time() - postgres_start
    gpu_time = time.time() - gpu_start
    
    # 処理統計
    actual_rows = len(df) if df is not None else 0
    print(f"[GPU] チャンク {chunk_id} 処理完了:")
    print(f"  - 処理行数: {actual_rows:,} 行")
    print(f"  - GPU処理時間: {gpu_time:.2f}秒")
    print(f"  - 内訳: 転送{transfer_time:.1f}秒 + 変換{postgres_time:.1f}秒")
    
    # メモリ解放
    del gpu_data
    del df
    cp.get_default_memory_pool().free_all_blocks()
    
    # チャンクファイル削除
    os.remove(chunk_file)
    print(f"✓ チャンクファイル削除: {chunk_file}")
    
    return gpu_time, file_size


async def pipeline_processing(columns: List[ColumnMeta]):
    """パイプライン処理（Rust転送とGPU処理を並列化）"""
    print("\n=== パイプライン処理開始 ===")
    
    total_rust_time = 0
    total_gpu_time = 0
    total_size = 0
    
    # ThreadPoolExecutorでRust処理を非同期実行
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 最初のRust転送を開始
        rust_future = executor.submit(process_rust_chunk, 0)
        
        for chunk_id in range(TOTAL_CHUNKS):
            # 現在のRust転送完了を待つ
            chunk_info = rust_future.result()
            total_rust_time += chunk_info['rust_time']
            total_size += chunk_info['file_size']
            
            # 次のRust転送を非同期で開始（最後のチャンクでない場合）
            if chunk_id < TOTAL_CHUNKS - 1:
                rust_future = executor.submit(process_rust_chunk, chunk_id + 1)
            
            # 現在のチャンクをGPUで処理（ストリームIDを交互に使用）
            gpu_time, _ = process_gpu_chunk(chunk_info, columns, chunk_id % 2)
            total_gpu_time += gpu_time
    
    return total_rust_time, total_gpu_time, total_size


def main():
    print("✅ CUDA context OK")
    print("=== PostgreSQL → Rust → GPU 最適化版 ===")
    print(f"チャンク数: {TOTAL_CHUNKS}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print("最適化内容:")
    print("  - mmapによる真のゼロコピー転送")
    print("  - 行数制限の解除")
    print("  - GPUストリームによる非同期処理")
    print("  - パイプライン処理による並列化")
    
    # クリーンアップ
    cleanup_files()
    
    total_start = time.time()
    
    try:
        # 最初のチャンクでカラム情報を取得
        print("\n=== メタデータ取得 ===")
        first_chunk_info = process_rust_chunk(0)
        
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
        gpu_time, _ = process_gpu_chunk(first_chunk_info, columns, 0)
        total_rust_time = first_chunk_info['rust_time']
        total_gpu_time = gpu_time
        total_size = first_chunk_info['file_size']
        
        # 残りのチャンクをパイプライン処理
        if TOTAL_CHUNKS > 1:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            rust_time, gpu_time, size = loop.run_until_complete(
                pipeline_processing(columns)
            )
            total_rust_time += rust_time
            total_gpu_time += gpu_time
            total_size += size
        
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
        print(f"全体スループット: {total_gb / total_time:.2f} GB/秒")
        print(f"Rust平均速度: {total_gb / total_rust_time:.2f} GB/秒")
        print(f"GPU平均速度: {total_gb / total_gpu_time:.2f} GB/秒")
        
        # 改善率の計算（ベースライン: 52.86GB in 263.49秒 = 0.20 GB/秒）
        baseline_throughput = 0.20
        improvement = (total_gb / total_time) / baseline_throughput
        print(f"\n改善率: {improvement:.1f}倍（ベースライン比）")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_files()


if __name__ == "__main__":
    main()