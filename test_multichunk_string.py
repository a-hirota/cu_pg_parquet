"""マルチチャンク文字列処理テスト

実際の本番条件（複数チャンク並列処理）での文字列破損を調査
"""

import os
import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.main_postgres_to_parquet import ZeroCopyProcessor
from src.direct_column_extractor import DirectColumnExtractor
import psycopg
import time
from concurrent.futures import ThreadPoolExecutor

print("=== マルチチャンク文字列処理テスト ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

def process_chunk(chunk_id, chunk_file):
    """単一チャンクを処理"""
    print(f"\nチャンク{chunk_id}処理開始: {chunk_file}")
    
    try:
        # チャンク読み込み
        with open(chunk_file, 'rb') as f:
            data = f.read()
        
        raw_host = np.frombuffer(data, dtype=np.uint8)
        raw_dev = cuda.to_device(raw_host)
        
        # ヘッダーサイズ検出
        header_sample = raw_dev[:128].copy_to_host()
        header_size = detect_pg_header_size(header_sample)
        
        # GPUパース
        field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=header_size
        )
        
        rows = field_offsets_dev.shape[0]
        print(f"  チャンク{chunk_id}: {rows} 行検出")
        
        # 文字列バッファ作成
        processor = ZeroCopyProcessor()
        string_buffers = processor.create_string_buffers(
            columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
        )
        
        # cuDF DataFrame作成
        extractor = DirectColumnExtractor()
        cudf_df = extractor.extract_columns_direct(
            raw_dev, field_offsets_dev, field_lengths_dev,
            columns, string_buffers
        )
        
        # 文字列データの検証
        errors = {}
        for col in ['lo_orderpriority', 'lo_shippriority']:
            if col in cudf_df.columns:
                error_count = 0
                # 大規模データの場合、サンプリングして確認
                sample_size = min(10000, len(cudf_df))
                sample_indices = np.random.choice(len(cudf_df), sample_size, replace=False)
                
                for idx in sample_indices:
                    try:
                        value = cudf_df[col].iloc[idx]
                        if col == 'lo_orderpriority':
                            expected = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                            if not any(value.startswith(p) for p in expected):
                                error_count += 1
                                if error_count <= 5 and idx % 2 == 1:  # 最初の5個の奇数行エラーを記録
                                    print(f"    ⚠️ チャンク{chunk_id} 行{idx}(奇数): {repr(value)}")
                    except:
                        error_count += 1
                
                errors[col] = error_count
        
        return chunk_id, rows, errors
        
    except Exception as e:
        print(f"  チャンク{chunk_id}処理エラー: {e}")
        return chunk_id, 0, {}

# 利用可能なチャンクを探す
chunk_files = []
for i in range(8):
    chunk_file = f"/dev/shm/chunk_{i}.bin"
    if os.path.exists(chunk_file):
        chunk_files.append((i, chunk_file))

if not chunk_files:
    print("チャンクファイルが見つかりません")
    exit(1)

print(f"発見されたチャンク: {len(chunk_files)}個")

# シーケンシャル処理（比較用）
print("\n=== シーケンシャル処理 ===")
sequential_results = []
for chunk_id, chunk_file in chunk_files[:2]:  # 最初の2チャンクのみ
    result = process_chunk(chunk_id, chunk_file)
    sequential_results.append(result)

# 並列処理
if len(chunk_files) >= 2:
    print("\n=== 並列処理（2チャンク同時） ===")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for chunk_id, chunk_file in chunk_files[:2]:
            futures.append(executor.submit(process_chunk, chunk_id, chunk_file))
        
        parallel_results = []
        for future in futures:
            parallel_results.append(future.result())

    # 結果比較
    print("\n=== 結果比較 ===")
    print("シーケンシャル処理:")
    for chunk_id, rows, errors in sequential_results:
        print(f"  チャンク{chunk_id}: {rows}行, エラー={errors}")
    
    print("\n並列処理:")
    for chunk_id, rows, errors in parallel_results:
        print(f"  チャンク{chunk_id}: {rows}行, エラー={errors}")

print("\n=== 結論 ===")
print("並列処理時にのみ文字列破損が発生する場合、")
print("GPU並列処理の同期問題やメモリ競合が原因の可能性があります。")