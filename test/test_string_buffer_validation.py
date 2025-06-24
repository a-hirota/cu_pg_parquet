"""文字列バッファ検証テスト

文字列バッファが正しく作成されているか、
それとも cuDF DataFrame 作成や Parquet 書き込みで問題が発生しているかを検証
"""

import os
import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.main_postgres_to_parquet import ZeroCopyProcessor
from src.direct_column_extractor import DirectColumnExtractor
import psycopg
import cupy as cp

print("=== 文字列バッファ検証テスト ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# 既存のチャンクファイルを使用（小規模）
chunk_files = ['/dev/shm/chunk_0.bin', '/dev/shm/chunk_2.bin']
chunk_file = None

for f in chunk_files:
    if os.path.exists(f):
        chunk_file = f
        break

if not chunk_file:
    print("チャンクファイルが見つかりません")
    exit(1)

print(f"使用するチャンク: {chunk_file}")

# チャンク読み込み（最初の100MBのみ）
with open(chunk_file, 'rb') as f:
    data = f.read(100*1024*1024)

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
print(f"検出行数: {rows}")

# 行数を制限
test_rows = min(10000, rows)
field_offsets_dev = field_offsets_dev[:test_rows]
field_lengths_dev = field_lengths_dev[:test_rows]

print(f"\n=== ステップ1: 文字列バッファ作成 ===")
processor = ZeroCopyProcessor()
string_buffers = processor.create_string_buffers(
    columns, test_rows, raw_dev, field_offsets_dev, field_lengths_dev
)

# 文字列バッファの内容を直接検証
print("\n=== 文字列バッファの直接検証 ===")
for col_name, buffer_info in string_buffers.items():
    if buffer_info['data'] is not None:
        print(f"\n{col_name}:")
        print(f"  データサイズ: {buffer_info['actual_size']} bytes")
        
        # オフセット配列を確認
        offsets_buffer = buffer_info['offsets']
        offsets_ptr = offsets_buffer.__cuda_array_interface__['data'][0]
        offsets_cupy = cp.asarray(cp.ndarray(
            shape=(test_rows + 1,),
            dtype=cp.int32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(offsets_ptr, (test_rows + 1) * 4, offsets_buffer),
                0
            )
        ))
        offsets_host = offsets_cupy.get()
        
        # データバッファを確認
        data_buffer = buffer_info['data']
        data_ptr = data_buffer.__cuda_array_interface__['data'][0]
        data_cupy = cp.asarray(cp.ndarray(
            shape=(buffer_info['actual_size'],),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(data_ptr, buffer_info['actual_size'], data_buffer),
                0
            )
        ))
        data_host = data_cupy.get()
        
        # 最初の10行の文字列を検証
        print(f"  最初の10行:")
        for i in range(min(10, test_rows)):
            start = offsets_host[i]
            end = offsets_host[i + 1]
            string_data = data_host[start:end].tobytes()
            print(f"    行{i} ({'偶数' if i % 2 == 0 else '奇数'}): {repr(string_data)}")
        
        # 奇数行の破損パターンをチェック
        error_count = 0
        for i in range(1, min(1000, test_rows), 2):  # 奇数行のみ
            start = offsets_host[i]
            end = offsets_host[i + 1]
            string_data = data_host[start:end].tobytes()
            
            # 期待されるパターンかチェック
            if col_name == 'lo_orderpriority':
                expected_patterns = [b'1-URGENT', b'2-HIGH', b'3-MEDIUM', b'4-NOT SPECI', b'5-LOW']
                if not any(string_data.startswith(p) for p in expected_patterns):
                    error_count += 1
                    if error_count < 5:  # 最初の5個のエラーを表示
                        print(f"    ⚠️ 奇数行{i}で破損: {repr(string_data)}")
        
        if error_count > 0:
            print(f"  ⚠️ 奇数行破損数: {error_count}/500")
        else:
            print(f"  ✅ 文字列バッファは正常")

print(f"\n=== ステップ2: cuDF DataFrame作成 ===")
try:
    extractor = DirectColumnExtractor()
    cudf_df = extractor.extract_columns_direct(
        raw_dev, field_offsets_dev, field_lengths_dev,
        columns, string_buffers
    )
    print(f"✅ DataFrame作成成功: {len(cudf_df)} 行 × {len(cudf_df.columns)} 列")
    
    # DataFrameの文字列列を確認
    print("\n=== DataFrame内の文字列データ確認 ===")
    for col in ['lo_orderpriority', 'lo_shippriority']:
        if col in cudf_df.columns:
            print(f"\n{col}列:")
            # 最初の10行
            for i in range(min(10, len(cudf_df))):
                try:
                    value = cudf_df[col].iloc[i]
                    print(f"  行{i} ({'偶数' if i % 2 == 0 else '奇数'}): {repr(value)}")
                except Exception as e:
                    print(f"  行{i}: エラー - {e}")
            
            # 奇数行の破損をチェック
            error_count = 0
            for i in range(1, min(1000, len(cudf_df)), 2):
                try:
                    value = cudf_df[col].iloc[i]
                    if col == 'lo_orderpriority':
                        expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                        if not any(value.startswith(p) for p in expected_patterns):
                            error_count += 1
                except:
                    error_count += 1
            
            if error_count > 0:
                print(f"  ⚠️ DataFrame内奇数行破損数: {error_count}/500")
            else:
                print(f"  ✅ DataFrame内データは正常")
    
except Exception as e:
    print(f"❌ DataFrame作成エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 結論 ===")
print("上記の結果から、問題が以下のどこで発生しているかが分かります：")
print("1. 文字列バッファ作成時")
print("2. cuDF DataFrame作成時")
print("3. Parquet書き込み時（このテストでは検証していない）")