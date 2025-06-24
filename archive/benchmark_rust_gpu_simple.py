"""
PostgreSQL → Rust → GPU シンプル版
1チャンクずつ処理してメモリ制限を回避

処理フロー:
1. Rust環境変数でチャンク0のみ処理
2. GPU処理→削除
3. Rust環境変数でチャンク1のみ処理
4. GPU処理→削除
（以下繰り返し）
"""

import os
import time
import subprocess
import json
import cupy as cp
import numpy as np
from pathlib import Path
from typing import List

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
TOTAL_CHUNKS = 4

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

def process_single_chunk(chunk_id: int) -> tuple:
    """1チャンクを処理"""
    print(f"\n{'='*60}")
    print(f"チャンク {chunk_id + 1}/{TOTAL_CHUNKS} を処理")
    print('='*60)
    
    # Rust実行
    print(f"\n[1] Rustデータ転送")
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
        print("❌ Rustエラー:")
        print(process.stderr)
        raise RuntimeError(f"チャンク{chunk_id}の転送失敗")
    
    # JSON結果を抽出
    output = process.stdout
    json_start = output.find("===CHUNK_RESULT_JSON===")
    json_end = output.find("===END_CHUNK_RESULT_JSON===")
    
    if json_start == -1 or json_end == -1:
        # 通常の実行の場合
        rust_time = time.time() - rust_start
        chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
        file_size = os.path.getsize(chunk_file)
        print(f"✓ Rust転送完了: {file_size / 1024**3:.2f} GB, {rust_time:.2f}秒")
        result = None
    else:
        json_str = output[json_start + len("===CHUNK_RESULT_JSON==="):json_end].strip()
        result = json.loads(json_str)
        rust_time = result['elapsed_seconds']
        file_size = result['total_bytes']
        chunk_file = result['chunk_file']
        print(f"✓ Rust転送完了: {file_size / 1024**3:.2f} GB, {rust_time:.2f}秒")
    
    # メタデータ取得（初回のみ）
    if chunk_id == 0:
        if result and 'columns' in result:
            # JSON結果から直接取得
            columns = []
            for col in result['columns']:
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
        else:
            raise RuntimeError("初回実行でカラム情報が取得できませんでした")
    else:
        # 前回のメタデータを使用
        columns = globals().get('saved_columns', [])
    
    # GPU処理
    print(f"\n[2] GPU処理")
    gpu_start = time.time()
    
    # mmapを使用したゼロコピー読み込み
    import mmap
    read_start = time.time()
    with open(chunk_file, 'rb') as f:
        # mmapでファイルをメモリマップ
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # GPU転送（真のゼロコピー）
            transfer_start = time.time()
            # cp.asarrayでmmapから直接GPU転送
            gpu_data = cp.asarray(np.frombuffer(mm, dtype=np.uint8))
            transfer_time = time.time() - transfer_start
    
    read_time = time.time() - read_start
    print(f"  ファイル読み込み+GPU転送: {read_time:.2f}秒 ({file_size / read_time / 1024**3:.2f} GB/秒)")
    print(f"    - mmap作成: {transfer_start - read_start:.2f}秒")
    print(f"    - GPU転送: {transfer_time:.2f}秒")
    
    # PostgreSQLヘッダーサイズを検出（最初の100バイトのみCPUで確認）
    from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
    header_size = detect_pg_header_size(gpu_data[:100].get())
    print(f"PostgreSQLヘッダーサイズ: {header_size} bytes")
    
    # PostgreSQL binary → Parquet変換
    chunk_output = f"benchmark/chunk_{chunk_id}.parquet"
    postgres_start = time.time()
    df, metrics = postgresql_to_cudf_parquet(
        raw_dev=gpu_data,
        columns=columns,
        ncols=len(columns),
        header_size=header_size,
        output_path=chunk_output
    )
    postgres_time = time.time() - postgres_start
    print(f"  PostgreSQL→Parquet変換: {postgres_time:.2f}秒")
    
    gpu_time = time.time() - gpu_start
    print(f"✓ GPU処理完了: {gpu_time:.2f}秒")
    print(f"  内訳: 読込{read_time:.1f}秒 + 転送{transfer_time:.1f}秒 + 変換{postgres_time:.1f}秒")
    
    # GPU/CPUメモリ解放
    del gpu_data
    cp.get_default_memory_pool().free_all_blocks()
    
    # チャンクファイル削除
    os.remove(chunk_file)
    print(f"✓ チャンクファイル削除: {chunk_file}")
    
    # カラム情報を保存（次回用）
    globals()['saved_columns'] = columns
    
    return rust_time, gpu_time, file_size

def main():
    print("✅ CUDA context OK")
    print("=== PostgreSQL → Rust → GPU シンプル版 ===")
    print(f"チャンク数: {TOTAL_CHUNKS}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    
    # クリーンアップ
    cleanup_files()
    
    total_start = time.time()
    total_rust_time = 0
    total_gpu_time = 0
    total_size = 0
    
    try:
        # 各チャンクを順次処理
        for chunk_id in range(TOTAL_CHUNKS):
            rust_time, gpu_time, file_size = process_single_chunk(chunk_id)
            total_rust_time += rust_time
            total_gpu_time += gpu_time
            total_size += file_size
        
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
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        raise
    finally:
        cleanup_files()

if __name__ == "__main__":
    main()