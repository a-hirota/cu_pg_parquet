"""文字列破損修正テスト

基本ベンチマークと同じ強制ホスト転送アプローチを適用した修正をテスト
"""

import os
import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.direct_column_extractor import DirectColumnExtractor
from src.main_postgres_to_parquet import ZeroCopyProcessor
import psycopg

print("=== 文字列破損修正テスト ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# 中規模データセット（5MBから100MB）
with open("/dev/shm/chunk_0.bin", "rb") as f:
    data = f.read(100*1024*1024)  # 100MB
raw_host = np.frombuffer(data, dtype=np.uint8)
raw_dev = cuda.to_device(raw_host)

print(f"データサイズ: {len(raw_host)} バイト")

# ヘッダーサイズ検出
header_sample = raw_dev[:128].copy_to_host()
header_size = detect_pg_header_size(header_sample)

# GPUパース
field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
    raw_dev, columns, header_size=header_size
)

total_rows = field_offsets_dev.shape[0]
print(f"総検出行数: {total_rows}")

# 大きめのテストサイズ（100万行）
test_rows = min(1000000, total_rows)
field_offsets_dev = field_offsets_dev[:test_rows]
field_lengths_dev = field_lengths_dev[:test_rows]
print(f"テスト行数: {test_rows}")

# 文字列バッファ作成
print("\n=== 修正版文字列バッファ作成 ===")
processor = ZeroCopyProcessor()
string_buffers = processor.create_string_buffers(
    columns, test_rows, raw_dev, field_offsets_dev, field_lengths_dev
)

print("\n=== 修正版cuDF DataFrame作成 ===")
extractor = DirectColumnExtractor()
cudf_df = extractor.extract_columns_direct(
    raw_dev, field_offsets_dev, field_lengths_dev,
    columns, string_buffers
)

print(f"DataFrame作成成功: {len(cudf_df)} 行 × {len(cudf_df.columns)} 列")

# 奇数行破損チェック（重点的にテスト）
print("\n=== 奇数行破損チェック（修正版） ===")

# より多くの奇数行をチェック
test_indices = [1, 3, 5, 101, 201, 501, 1001, 2001, 5001, 10001, 50001, 100001]
test_indices = [i for i in test_indices if i < len(cudf_df)]

for col in ['lo_orderpriority', 'lo_shippriority']:
    print(f"\n{col}列のチェック:")
    
    # 偶数行のサンプル（正常なはず）
    even_samples = []
    for i in [0, 2, 4, 100, 200, 500, 1000, 2000, 5000, 10000]:
        if i < len(cudf_df):
            even_samples.append(cudf_df[col].iloc[i])
    
    # 奇数行のサンプル（これまで破損していた）
    odd_samples = []
    corruption_detected = False
    
    for idx in test_indices:
        try:
            value = cudf_df[col].iloc[idx]
            odd_samples.append(value)
            print(f"  奇数行{idx}: {repr(value)}")
            
            # 破損パターン検出（明らかに異常な値）
            if isinstance(value, str):
                if len(value) == 0 or '\x00' in value or any(ord(c) < 32 or ord(c) > 126 for c in value if c != '\n' and c != '\t'):
                    print(f"    ⚠️ 破損の可能性: {repr(value)}")
                    corruption_detected = True
        except Exception as e:
            print(f"  奇数行{idx}: エラー - {e}")
            corruption_detected = True
    
    if not corruption_detected:
        print(f"  ✅ {col}列: 奇数行破損は検出されませんでした")
    else:
        print(f"  ❌ {col}列: 奇数行破損が依然として存在します")

print("\n=== 修正テスト完了 ===")