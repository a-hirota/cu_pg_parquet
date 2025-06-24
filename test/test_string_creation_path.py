"""文字列作成パスのデバッグ

どのコードパスで文字列が作成されているか確認
"""

import os
import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.direct_column_extractor import DirectColumnExtractor
from src.main_postgres_to_parquet import ZeroCopyProcessor
import psycopg
import warnings

# warningsを全て表示
warnings.filterwarnings('always')

print("=== 文字列作成パステスト ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# チャンクファイル
chunk_file = "/dev/shm/chunk_2.bin"
if not os.path.exists(chunk_file):
    print("チャンクファイルが見つかりません")
    exit(1)

# チャンク読み込み（小さいサンプルのみ）
with open(chunk_file, 'rb') as f:
    # 最初の10MBのみ読み込み
    data = f.read(10 * 1024 * 1024)

raw_host = np.frombuffer(data, dtype=np.uint8)
raw_dev = cuda.to_device(raw_host)

# ヘッダーサイズ検出
header_sample = raw_dev[:128].copy_to_host()
header_size = detect_pg_header_size(header_sample)

print("\n=== GPUパース ===")
field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
    raw_dev, columns, header_size=header_size
)

rows = field_offsets_dev.shape[0]
print(f"検出行数: {rows}")

# 文字列バッファ作成（デバッグモード）
print("\n=== 文字列バッファ作成 ===")
processor = ZeroCopyProcessor()
string_buffers = processor.create_string_buffers(
    columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
)

print(f"\n文字列バッファ: {list(string_buffers.keys())}")
for col_name, buf_info in string_buffers.items():
    print(f"\n{col_name}:")
    print(f"  data type: {type(buf_info['data'])}")
    print(f"  offsets type: {type(buf_info['offsets'])}")
    print(f"  actual_size: {buf_info['actual_size']}")

# DirectColumnExtractorでSeries作成
print("\n=== DirectColumnExtractor実行 ===")
import sys

# 詳細なエラー出力を有効化
class VerboseDirectColumnExtractor(DirectColumnExtractor):
    def _create_string_series_from_buffer(self, col, rows, buffer_info):
        print(f"\n_create_string_series_from_buffer: {col.name}")
        print(f"  data buffer type: {type(buffer_info['data'])}")
        print(f"  offsets buffer type: {type(buffer_info['offsets'])}")
        
        # 元のメソッドを呼び出し
        try:
            # RMM DeviceBufferチェック
            import rmm
            if isinstance(buffer_info['data'], rmm.DeviceBuffer):
                print("  ✓ RMM DeviceBuffer検出（真のゼロコピーパス）")
            else:
                print("  × Numba配列検出（フォールバックパス）")
            
            # 親クラスのメソッドを呼び出し
            result = super()._create_string_series_from_buffer(col, rows, buffer_info)
            print(f"  → 成功: Series型={type(result)}")
            return result
            
        except Exception as e:
            print(f"  → エラー: {e}")
            import traceback
            traceback.print_exc()
            raise

extractor = VerboseDirectColumnExtractor()

try:
    cudf_df = extractor.extract_columns_direct(
        raw_dev, field_offsets_dev, field_lengths_dev,
        columns, string_buffers
    )
    
    print(f"\n=== 結果 ===")
    print(f"DataFrame作成成功: {len(cudf_df)}行")
    
    # 最初の数行をチェック
    if 'lo_orderpriority' in cudf_df.columns:
        print("\nlo_orderpriority最初の10行:")
        for i in range(min(10, len(cudf_df))):
            value = cudf_df['lo_orderpriority'].iloc[i]
            print(f"  行{i}({'偶数' if i % 2 == 0 else '奇数'}): {repr(value)}")
    
except Exception as e:
    print(f"\nDataFrame作成失敗: {e}")
    import traceback
    traceback.print_exc()