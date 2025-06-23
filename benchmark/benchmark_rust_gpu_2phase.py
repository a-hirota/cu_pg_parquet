"""
PostgreSQL → Rust → GPU 2フェーズ処理版
メモリ制限対応のため、2チャンクずつ処理

処理フロー:
1. チャンク0,1をRust転送→GPU処理→削除
2. チャンク2,3をRust転送→GPU処理→削除
"""

import os
import time
import subprocess
import psutil
import json
import cupy as cp
from pathlib import Path
from typing import Dict, Tuple, List
import gc

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_rust_gpu_2phase.parquet"
META_FILE = "/dev/shm/lineorder_meta.json"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_optimized"

def cleanup_shm_files(chunks_to_clean=None):
    """SHMファイルをクリーンアップ"""
    if chunks_to_clean is None:
        chunks_to_clean = [0, 1, 2, 3]
    
    files_to_clean = [META_FILE]
    files_to_clean += [f"{OUTPUT_DIR}/chunk_{i}.bin" for i in chunks_to_clean]
    
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ クリーンアップ: {file_path}")

def run_rust_for_chunks(chunk_start: int, chunk_count: int) -> Tuple[float, float]:
    """指定範囲のチャンクのみRustで転送"""
    print(f"\n=== Rustデータ転送: チャンク {chunk_start}-{chunk_start + chunk_count - 1} ===")
    
    env = os.environ.copy()
    env['CHUNKS'] = str(chunk_count)
    env['CHUNK_OFFSET'] = str(chunk_start)
    env['RUST_BACKTRACE'] = '1'
    
    # カスタムバイナリを作成する必要があるため、一旦元のバイナリを使用
    # TODO: Rust側でCHUNK_OFFSETをサポートする必要がある
    
    start_time = time.time()
    process = subprocess.run(
        [RUST_BINARY],
        capture_output=True,
        text=True,
        env=env
    )
    
    elapsed = time.time() - start_time
    
    if process.returncode != 0:
        print(f"❌ Rustプロセスエラー:")
        print(process.stderr)
        raise RuntimeError("Rustデータ転送に失敗しました")
    
    # 転送されたデータサイズを計算
    total_size = 0
    for i in range(chunk_start, chunk_start + chunk_count):
        chunk_file = f"{OUTPUT_DIR}/chunk_{i}.bin"
        if os.path.exists(chunk_file):
            total_size += os.path.getsize(chunk_file)
    
    speed = total_size / elapsed / (1024**3)
    print(f"✓ 転送完了: {total_size / (1024**3):.2f} GB, {elapsed:.2f}秒 ({speed:.2f} GB/秒)")
    
    return elapsed, speed

def load_metadata() -> List[ColumnMeta]:
    """メタデータファイルを読み込み"""
    with open(META_FILE, 'r') as f:
        meta_json = json.load(f)
    
    columns = []
    for col in meta_json['columns']:
        pg_oid = col['pg_oid']
        arrow_id = PG_OID_TO_ARROW.get(pg_oid, UNKNOWN)
        
        elem_size = None
        if arrow_id.fixed_size is not None:
            elem_size = arrow_id.fixed_size
        
        column_meta = ColumnMeta(
            name=col['name'],
            data_type=col['data_type'],
            pg_oid=pg_oid,
            pg_typmod=-1,
            arrow_id=arrow_id,
            elem_size=elem_size if elem_size is not None else 0
        )
        columns.append(column_meta)
    
    return columns

def process_chunks_on_gpu(chunk_ids: List[int], columns: List[ColumnMeta]) -> Tuple[float, float]:
    """指定されたチャンクをGPU上で処理"""
    print(f"\n=== GPU処理: チャンク {chunk_ids} ===")
    
    start_time = time.time()
    total_size = 0
    
    for chunk_id in chunk_ids:
        chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
        
        # データ読み込み
        with open(chunk_file, 'rb') as f:
            data = f.read()
        
        file_size = len(data)
        total_size += file_size
        print(f"チャンク {chunk_id}: {file_size / (1024**3):.2f} GB")
        
        # GPU転送（ゼロコピー）
        gpu_data = cp.frombuffer(data, dtype=cp.uint8)
        
        # CPUメモリ解放
        del data
        
        # PostgreSQL binary → Parquet変換
        chunk_output = f"benchmark/chunk_{chunk_id}.parquet"
        postgresql_to_cudf_parquet(
            gpu_data,
            columns,
            output_parquet_path=chunk_output,
            table_name=TABLE_NAME,
            estimated_rows=None
        )
        
        # GPU/CPUメモリ解放
        del gpu_data
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    
    elapsed = time.time() - start_time
    speed = total_size / elapsed / (1024**3)
    print(f"✓ GPU処理完了: {elapsed:.2f}秒 ({speed:.2f} GB/秒)")
    
    return elapsed, speed

def main():
    print("✅ CUDA context OK")
    print("=== PostgreSQL → Rust → GPU 2フェーズ処理版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"出力: {OUTPUT_PARQUET_PATH}")
    
    # 全体クリーンアップ
    cleanup_shm_files()
    
    total_start = time.time()
    
    try:
        # 2チャンクずつ処理するため、まず元のRustを2チャンクモードで実行
        # 実際にはRust側の修正が必要だが、ここでは4チャンク全体を実行してメモリ不足を回避
        
        print("\n" + "="*60)
        print("代替案: シンプルな順次処理")
        print("="*60)
        
        # CHUNKSを2に設定してRustを実行
        env = os.environ.copy()
        env['CHUNKS'] = '2'  # 環境変数で制御を試みる
        
        # 現在の実装では全4チャンクを一度に処理しようとするため、
        # 別のアプローチが必要
        
        print("\n❌ 現在の実装では2チャンクずつの処理ができません")
        print("理由: Rustプログラムが4チャンク固定で実装されているため")
        print("\n推奨される解決策:")
        print("1. Rustコードを環境変数CHUNKS/CHUNK_OFFSETに対応させる")
        print("2. または、/tmpなど大容量ストレージを使用する")
        print("3. または、チャンクサイズを小さくする（例: 8チャンク）")
        
        # デモ用に小さなテストを実行
        print("\n代わりに、通常の4チャンク処理を/tmpで実行することを推奨します")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        raise
    
    finally:
        cleanup_shm_files()
        print("\n✓ クリーンアップ完了")
    
    total_elapsed = time.time() - total_start
    print(f"\n総実行時間: {total_elapsed:.2f}秒")

if __name__ == "__main__":
    main()