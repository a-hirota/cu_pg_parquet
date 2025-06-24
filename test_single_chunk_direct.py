"""単一チャンクでの直接抽出テスト"""

import os
import time
import numpy as np
from numba import cuda
import rmm
from src.types import ColumnMeta
from src.main_postgres_to_parquet_direct import postgresql_to_cudf_parquet_direct
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.metadata import fetch_column_meta
import psycopg
import cudf

print("=== 単一チャンク直接抽出テスト ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# チャンクファイル
chunk_file = "/dev/shm/chunk_5.bin"
if not os.path.exists(chunk_file):
    print(f"チャンクファイルが見つかりません: {chunk_file}")
    exit(1)

print(f"\nチャンク読み込み: {chunk_file}")
with open(chunk_file, 'rb') as f:
    data = f.read()

print(f"チャンクサイズ: {len(data) / (1024**3):.2f} GB")

raw_host = np.frombuffer(data, dtype=np.uint8)
raw_dev = cuda.to_device(raw_host)

# ヘッダーサイズ検出
header_sample = raw_dev[:128].copy_to_host()
header_size = detect_pg_header_size(header_sample)

# 直接抽出処理
output_path = "test_single_chunk.parquet"
print("\n直接抽出処理開始...")
start_time = time.time()

try:
    cudf_df, timing = postgresql_to_cudf_parquet_direct(
        raw_dev=raw_dev,
        columns=columns,
        ncols=len(columns),
        header_size=header_size,
        output_path=output_path,
        compression='snappy',
        use_rmm=True,
        optimize_gpu=True
    )
    
    elapsed = time.time() - start_time
    print(f"\n処理完了: {len(cudf_df):,}行 ({elapsed:.2f}秒)")
    
    # 文字列破損チェック
    print("\n=== 文字列破損チェック ===")
    even_errors = 0
    odd_errors = 0
    error_samples = []
    
    if 'lo_orderpriority' in cudf_df.columns:
        check_rows = min(10000, len(cudf_df))
        
        for i in range(check_rows):
            try:
                value = cudf_df['lo_orderpriority'].iloc[i]
                expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                is_valid = any(value.startswith(p) for p in expected_patterns)
                
                if not is_valid:
                    if i % 2 == 0:
                        even_errors += 1
                    else:
                        odd_errors += 1
                    
                    if len(error_samples) < 10:
                        error_samples.append((i, value))
                
            except Exception as e:
                pass
    
    print(f"チェック行数: {check_rows}")
    print(f"偶数行エラー: {even_errors}")
    print(f"奇数行エラー: {odd_errors}")
    
    if error_samples:
        print(f"\nエラーサンプル:")
        for row, value in error_samples:
            print(f"  行{row}({'偶数' if row % 2 == 0 else '奇数'}): {repr(value)}")
    else:
        print("\n✅ 文字列破損なし！")
    
    # Parquetファイル検証
    print(f"\nParquetファイル検証: {output_path}")
    verify_df = cudf.read_parquet(output_path)
    print(f"読み込み成功: {len(verify_df)}行")
    
    # クリーンアップ
    os.remove(output_path)
    
except Exception as e:
    print(f"\nエラー: {e}")
    import traceback
    traceback.print_exc()