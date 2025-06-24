"""シンプルな累積和デバッグテスト"""

import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.main_postgres_to_parquet import ZeroCopyProcessor
import psycopg
import os

print("=== 累積和デバッグテスト ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# 既存チャンクファイル読み込み（最初の100MBまで拡張）
with open("/dev/shm/chunk_0.bin", "rb") as f:
    data = f.read(100*1024*1024)  # 100MB
raw_host = np.frombuffer(data, dtype=np.uint8)
raw_dev = cuda.to_device(raw_host)

print(f"データサイズ: {len(raw_host)} バイト")

# ヘッダーサイズ検出
header_sample = raw_dev[:128].copy_to_host()
header_size = detect_pg_header_size(header_sample)
print(f"ヘッダーサイズ: {header_size} バイト")

# GPUパース
field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
    raw_dev, columns, header_size=header_size
)

rows = field_offsets_dev.shape[0]
print(f"検出行数: {rows}")

if rows > 100000:
    rows = 100000  # 10万行に拡大
    field_offsets_dev = field_offsets_dev[:rows]
    field_lengths_dev = field_lengths_dev[:rows]
    print(f"処理行数を {rows} に制限")

# 文字列バッファ作成（デバッグ情報付き）
print("\n=== 文字列バッファ作成開始 ===")
processor = ZeroCopyProcessor()
string_buffers = processor.create_string_buffers(
    columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
)

print("\n=== cuDF DataFrame作成テスト ===")
from src.direct_column_extractor import DirectColumnExtractor
extractor = DirectColumnExtractor()

# 直接抽出でcuDF DataFrame作成
cudf_df = extractor.extract_columns_direct(
    raw_dev, field_offsets_dev, field_lengths_dev,
    columns, string_buffers
)

print(f"DataFrame作成成功: {len(cudf_df)} 行 × {len(cudf_df.columns)} 列")

# 最初の10行と奇数行をチェック
print("\n=== 文字列データ確認 ===")
for col in ['lo_orderpriority', 'lo_shippriority']:
    print(f"\n{col}列:")
    try:
        # 最初の10行
        for i in range(min(10, len(cudf_df))):
            value = cudf_df[col].iloc[i]
            print(f"  行{i}: {repr(value)}")
        
        # 破損の可能性がある奇数行をさらにチェック
        print(f"  --- 奇数行チェック ---")
        for i in [101, 201, 301, 501, 1001]:
            if i < len(cudf_df):
                try:
                    value = cudf_df[col].iloc[i]
                    print(f"  行{i}: {repr(value)}")
                except Exception as e:
                    print(f"  行{i}: エラー - {e}")
                    
    except Exception as e:
        print(f"  エラー: {e}")

print("\n=== テスト完了 ===")