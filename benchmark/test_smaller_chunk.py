"""
小さいデータでのテスト実行
メモリ使用量とパフォーマンスを確認
"""

import os
import time
import subprocess
import json
import cupy as cp
import numpy as np
from pathlib import Path

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

# 環境変数でテストサイズを制御
TEST_SIZE_GB = float(os.environ.get('TEST_SIZE_GB', '1.0'))  # デフォルト1GB
TABLE_NAME = "lineorder"


def create_test_data():
    """テスト用の小さいデータセットを作成"""
    print(f"テストデータ作成中 ({TEST_SIZE_GB} GB)...")
    
    # Rustでsmall chunk取得
    env = os.environ.copy()
    env['CHUNK_ID'] = '0'
    env['TOTAL_CHUNKS'] = '40'  # より小さいチャンクに分割
    
    rust_start = time.time()
    process = subprocess.run(
        ["/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"],
        capture_output=True,
        text=True,
        env=env
    )
    
    if process.returncode != 0:
        print("❌ Rustエラー:")
        print(process.stderr)
        raise RuntimeError("データ取得失敗")
    
    # JSON結果を抽出
    output = process.stdout
    json_start = output.find("===CHUNK_RESULT_JSON===")
    json_end = output.find("===END_CHUNK_RESULT_JSON===")
    
    if json_start != -1 and json_end != -1:
        json_str = output[json_start + len("===CHUNK_RESULT_JSON==="):json_end].strip()
        result = json.loads(json_str)
        return result
    
    raise RuntimeError("JSONデータが見つかりません")


def test_gpu_processing():
    """GPUメモリ使用量とパフォーマンスをテスト"""
    print("=== GPU処理テスト ===")
    print(f"テストサイズ: {TEST_SIZE_GB} GB")
    
    # テストデータ作成
    chunk_info = create_test_data()
    chunk_file = chunk_info['chunk_file']
    file_size = chunk_info['total_bytes']
    columns_data = chunk_info['columns']
    
    print(f"✓ データ取得完了: {file_size / 1024**3:.2f} GB")
    
    # カラム情報の準備
    columns = []
    for col in columns_data:
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
    
    # GPUメモリ状況（処理前）
    mempool = cp.get_default_memory_pool()
    print(f"\n=== GPUメモリ使用状況（処理前）===")
    print(f"使用中: {mempool.used_bytes() / 1024**3:.2f} GB")
    print(f"総計: {mempool.total_bytes() / 1024**3:.2f} GB")
    
    # GPU転送
    print("\n=== GPU転送 ===")
    transfer_start = time.time()
    
    with open(chunk_file, 'rb') as f:
        data = f.read()
    gpu_data = cp.frombuffer(data, dtype=cp.uint8)
    del data
    
    transfer_time = time.time() - transfer_start
    print(f"転送時間: {transfer_time:.2f}秒 ({file_size / transfer_time / 1024**3:.2f} GB/秒)")
    
    # メモリ使用量（転送後）
    print(f"\n=== GPUメモリ使用状況（転送後）===")
    print(f"使用中: {mempool.used_bytes() / 1024**3:.2f} GB")
    print(f"総計: {mempool.total_bytes() / 1024**3:.2f} GB")
    
    # パース処理
    print("\n=== GPUパース処理 ===")
    from src.cuda_kernels.postgres_binary_parser import (
        detect_pg_header_size,
        parse_binary_chunk_gpu_ultra_fast_v2_lite
    )
    
    header_size = detect_pg_header_size(gpu_data[:100].get())
    
    try:
        parse_start = time.time()
        field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2_lite(
            gpu_data, columns, header_size=header_size, debug=True
        )
        parse_time = time.time() - parse_start
        
        rows = field_offsets_dev.shape[0]
        print(f"✓ パース完了: {rows:,} 行, {parse_time:.2f}秒")
        
        # メモリ使用量（パース後）
        print(f"\n=== GPUメモリ使用状況（パース後）===")
        print(f"使用中: {mempool.used_bytes() / 1024**3:.2f} GB")
        print(f"総計: {mempool.total_bytes() / 1024**3:.2f} GB")
        
        # メモリ使用量の詳細
        print(f"\n=== メモリ使用量の内訳 ===")
        print(f"入力データ: {file_size / 1024**3:.2f} GB")
        print(f"field_offsets: {rows * len(columns) * 4 / 1024**3:.2f} GB")
        print(f"field_lengths: {rows * len(columns) * 4 / 1024**3:.2f} GB")
        print(f"推定合計: {(file_size + rows * len(columns) * 4 * 2) / 1024**3:.2f} GB")
        
        # 完全な処理を試す
        if TEST_SIZE_GB <= 2.0:
            print("\n=== 完全な変換処理 ===")
            convert_start = time.time()
            df, metrics = postgresql_to_cudf_parquet(
                raw_dev=gpu_data,
                columns=columns,
                ncols=len(columns),
                header_size=header_size,
                output_path=f"test_{TEST_SIZE_GB}gb.parquet"
            )
            convert_time = time.time() - convert_start
            print(f"✓ 変換完了: {convert_time:.2f}秒")
            print(f"  出力行数: {len(df):,}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # クリーンアップ
        del gpu_data
        if 'field_offsets_dev' in locals():
            del field_offsets_dev
        if 'field_lengths_dev' in locals():
            del field_lengths_dev
        mempool.free_all_blocks()
        
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
        
        print(f"\n=== GPUメモリ使用状況（クリーンアップ後）===")
        print(f"使用中: {mempool.used_bytes() / 1024**3:.2f} GB")
        print(f"総計: {mempool.total_bytes() / 1024**3:.2f} GB")


if __name__ == "__main__":
    test_gpu_processing()