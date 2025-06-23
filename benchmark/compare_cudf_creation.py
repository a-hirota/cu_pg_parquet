"""
cuDF DataFrame作成の処理時間比較
CPUシングルスレッド問題の解決効果を測定
"""

import os
import time
import subprocess
import json
import numpy as np
from numba import cuda
import cupy as cp
import rmm

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size

# テスト用に小さいチャンク（1GB）を使用
TEST_SIZE_GB = 1.0


def get_test_chunk():
    """テスト用データを取得"""
    env = os.environ.copy()
    env['CHUNK_ID'] = '0'
    env['TOTAL_CHUNKS'] = '50'  # 約1GBのチャンクを取得
    
    process = subprocess.run(
        ["/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"],
        capture_output=True,
        text=True,
        env=env
    )
    
    if process.returncode != 0:
        raise RuntimeError(f"データ取得失敗: {process.stderr}")
    
    output = process.stdout
    json_start = output.find("===CHUNK_RESULT_JSON===")
    json_end = output.find("===END_CHUNK_RESULT_JSON===")
    
    if json_start != -1 and json_end != -1:
        json_str = output[json_start + len("===CHUNK_RESULT_JSON==="):json_end].strip()
        result = json.loads(json_str)
        return result
    
    raise RuntimeError("JSONデータが見つかりません")


def test_method1_cp_frombuffer(chunk_file, columns):
    """方法1: cp.frombuffer（従来の方法）"""
    print("\n=== 方法1: cp.frombuffer (従来) ===")
    
    start = time.time()
    
    # ファイル読み込み
    read_start = time.time()
    with open(chunk_file, 'rb') as f:
        data = f.read()
    read_time = time.time() - read_start
    
    # GPU転送
    transfer_start = time.time()
    gpu_data = cp.frombuffer(data, dtype=cp.uint8)
    del data
    transfer_time = time.time() - transfer_start
    
    # ヘッダー検出
    header_start = time.time()
    header_size = detect_pg_header_size(gpu_data[:100].get())
    header_time = time.time() - header_start
    
    # 処理（cuDF作成は除く）
    from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2
    parse_start = time.time()
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
        gpu_data, columns, header_size=header_size
    )
    parse_time = time.time() - parse_start
    
    total_time = time.time() - start
    
    print(f"  読み込み: {read_time:.3f}秒")
    print(f"  GPU転送: {transfer_time:.3f}秒")
    print(f"  ヘッダー検出: {header_time:.3f}秒")
    print(f"  GPUパース: {parse_time:.3f}秒")
    print(f"  合計: {total_time:.3f}秒")
    
    # メモリ解放
    del gpu_data
    cp.get_default_memory_pool().free_all_blocks()
    
    return total_time


def test_method2_cuda_to_device(chunk_file, columns):
    """方法2: cuda.to_device（改善版）"""
    print("\n=== 方法2: cuda.to_device (改善版) ===")
    
    start = time.time()
    
    # ファイル読み込み
    read_start = time.time()
    with open(chunk_file, 'rb') as f:
        data = f.read()
    raw_host = np.frombuffer(data, dtype=np.uint8)
    del data
    read_time = time.time() - read_start
    
    # GPU転送
    transfer_start = time.time()
    raw_dev = cuda.to_device(raw_host)
    transfer_time = time.time() - transfer_start
    
    # ヘッダー検出
    header_start = time.time()
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    header_time = time.time() - header_start
    
    # cuDF一括処理でテスト
    process_start = time.time()
    try:
        cudf_df, timing = postgresql_to_cudf_parquet(
            raw_dev=raw_dev,
            columns=columns,
            ncols=len(columns),
            header_size=header_size,
            output_path="test_method2.parquet",
            compression='snappy',
            use_rmm=True,
            optimize_gpu=True
        )
        process_time = time.time() - process_start
        
        print(f"  読み込み: {read_time:.3f}秒")
        print(f"  GPU転送: {transfer_time:.3f}秒")
        print(f"  ヘッダー検出: {header_time:.3f}秒")
        print(f"  統一処理（パース+cuDF+Parquet）: {process_time:.3f}秒")
        print(f"    - GPUパース: {timing.get('gpu_parsing', 0):.3f}秒")
        print(f"    - デコード+cuDF作成: {timing.get('decode_and_export', 0):.3f}秒")
        print(f"    - Parquet書き込み: {timing.get('parquet_export', 0):.3f}秒")
        
        # cuDF作成部分のみの時間
        cudf_creation_time = timing.get('cudf_creation', 0)
        print(f"  ★ cuDF作成のみ: {cudf_creation_time:.3f}秒")
        
        del cudf_df
        
    except Exception as e:
        print(f"  エラー: {e}")
        process_time = 0
    
    total_time = time.time() - start
    print(f"  合計: {total_time:.3f}秒")
    
    # メモリ解放
    del raw_dev
    del raw_host
    
    return total_time


def main():
    print("=== cuDF DataFrame作成の処理時間比較 ===")
    print("テストデータ: 約1GB")
    
    # RMM初期化
    if not rmm.is_initialized():
        gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        pool_size = int(gpu_memory * 0.9)
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=pool_size,
            maximum_pool_size=pool_size
        )
    
    try:
        # テストデータ取得
        print("\nテストデータ取得中...")
        chunk_info = get_test_chunk()
        chunk_file = chunk_info['chunk_file']
        file_size = chunk_info['total_bytes']
        
        # カラム情報
        columns = []
        for col in chunk_info['columns']:
            pg_oid = col['pg_oid']
            arrow_info = PG_OID_TO_ARROW.get(pg_oid, (UNKNOWN, None))
            arrow_id, elem_size = arrow_info
            
            columns.append(ColumnMeta(
                name=col['name'],
                pg_oid=pg_oid,
                pg_typmod=-1,
                arrow_id=arrow_id,
                elem_size=elem_size if elem_size is not None else 0
            ))
        
        print(f"✓ データサイズ: {file_size / 1024**3:.2f} GB")
        print(f"✓ カラム数: {len(columns)}")
        
        # 方法1のテスト
        time1 = test_method1_cp_frombuffer(chunk_file, columns)
        
        # メモリクリア
        cp.get_default_memory_pool().free_all_blocks()
        time.sleep(1)
        
        # 方法2のテスト
        time2 = test_method2_cuda_to_device(chunk_file, columns)
        
        # 結果比較
        print(f"\n=== 結果まとめ ===")
        print(f"方法1 (cp.frombuffer): {time1:.3f}秒")
        print(f"方法2 (cuda.to_device): {time2:.3f}秒")
        if time1 > 0 and time2 > 0:
            improvement = (time1 - time2) / time1 * 100
            print(f"改善率: {improvement:.1f}%")
        
    finally:
        # クリーンアップ
        if 'chunk_file' in locals() and os.path.exists(chunk_file):
            os.remove(chunk_file)
        if os.path.exists("test_method2.parquet"):
            os.remove("test_method2.parquet")


if __name__ == "__main__":
    main()