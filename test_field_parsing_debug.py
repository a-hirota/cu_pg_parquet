"""フィールド解析デバッグテスト"""

import os
import subprocess
import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
import psycopg
from debug_field_parsing import analyze_field_parsing

# 小さなチャンクを生成
print("小さなテストチャンクを生成中...")
env = os.environ.copy()
env['CHUNK_ID'] = '0'
env['TOTAL_CHUNKS'] = '1'
env['TEST_LIMIT'] = '10000'  # 10,000行のみ

# より大きなデータセットでテスト（100万行）
test_sql = """
WITH limited_data AS (
    SELECT * FROM lineorder 
    ORDER BY lo_orderkey 
    LIMIT 1000000
)
SELECT * FROM limited_data
"""

# テストデータ作成用の簡易Pythonスクリプト
create_test_chunk = """
import psycopg
import os

dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)

print("大規模テストチャンク（100万行）を作成中...")
with conn.cursor() as cur:
    # バイナリ形式でCOPY
    with open('/dev/shm/test_chunk.bin', 'wb') as f:
        with cur.copy("COPY (SELECT * FROM lineorder LIMIT 1000000) TO STDOUT (FORMAT BINARY)") as copy:
            for data in copy:
                f.write(data)
                
print("テストチャンク作成完了")
"""

# テストチャンク作成
with open('/tmp/create_test_chunk.py', 'w') as f:
    f.write(create_test_chunk)

subprocess.run(['python', '/tmp/create_test_chunk.py'], env=os.environ.copy())

# チャンク読み込み
with open('/dev/shm/test_chunk.bin', 'rb') as f:
    data = f.read()

print(f"チャンクサイズ: {len(data)} バイト")

raw_host = np.frombuffer(data, dtype=np.uint8)
raw_dev = cuda.to_device(raw_host)

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# ヘッダーサイズ検出
header_sample = raw_dev[:128].copy_to_host()
header_size = detect_pg_header_size(header_sample)
print(f"ヘッダーサイズ: {header_size} バイト")

# GPUパース
field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
    raw_dev, columns, header_size=header_size, debug=True
)

rows = field_offsets_dev.shape[0]
print(f"検出行数: {rows}")

# フィールドオフセットとフィールド長が既に取得されている
field_offsets_actual = field_offsets_dev
field_lengths_actual = field_lengths_dev

# 最初の行のフィールド情報を確認
first_row_offsets = field_offsets_actual[0].copy_to_host()
first_row_lengths = field_lengths_actual[0].copy_to_host()

print(f"\n最初の行のフィールド情報:")
print(f"フィールドオフセット: {first_row_offsets[:5]}...")  # 最初の5フィールド
print(f"フィールド長: {first_row_lengths[:5]}...")

# 列名とインデックスを確認
print("\n=== 列情報 ===")
for i, col in enumerate(columns):
    print(f"{i}: {col.name} (type: {col.pg_oid})")

# lo_orderpriority列のインデックスを探す
lo_orderpriority_idx = None
for i, col in enumerate(columns):
    if col.name == 'lo_orderpriority':
        lo_orderpriority_idx = i
        break

col_idx = lo_orderpriority_idx if lo_orderpriority_idx is not None else 12
print(f"\nlo_orderpriority列（インデックス{col_idx}）の最初の10行:")
for i in range(min(10, rows)):
    offset = field_offsets_actual[i, col_idx]
    length = field_lengths_actual[i, col_idx]
    print(f"行{i}: offset={offset}, length={length}")

# 実際のデータを確認（小規模で）
print("\n=== 実際の文字列データ確認 ===")
field_offsets_host = field_offsets_actual.copy_to_host()
field_lengths_host = field_lengths_actual.copy_to_host()
raw_host = raw_dev.copy_to_host()

# 奇数行と偶数行の比較
print("\n=== 奇数行と偶数行の比較 ===")
for i in range(min(20, rows)):
    length = field_lengths_host[i, col_idx]
    offset = field_offsets_host[i, col_idx]
    
    if length > 0 and offset > 0:
        # 実際のデータを読み取り
        if offset + length <= len(raw_host):
            string_data = raw_host[offset:offset+length].tobytes()
            print(f"行{i} ({'偶数' if i % 2 == 0 else '奇数'}): 長さ={length}, データ={repr(string_data)}")
            
            # 異常チェック
            if length != 15:  # lo_orderpriority は常に15バイト
                print(f"  ⚠️ 異常な長さ！（期待値: 15）")
            if b'\x00' in string_data:
                print(f"  ⚠️ NULL文字が含まれています！")
        else:
            print(f"行{i}: オフセット範囲外 (offset={offset}, length={length})")
    else:
        print(f"行{i}: NULL or 無効 (length={length}, offset={offset})")

# 大規模データでの問題を再現するため、より多くの行をチェック
print("\n=== 大規模行インデックスでのチェック ===")
large_indices = [100, 101, 500, 501, 1000, 1001, 5000, 5001, 7000, 7001, 
                 10000, 10001, 50000, 50001, 100000, 100001, 500000, 500001]
for i in large_indices:
    if i < rows:
        length = field_lengths_host[i, col_idx]
        offset = field_offsets_host[i, col_idx]
        
        if length > 0 and offset > 0 and offset + length <= len(raw_host):
            string_data = raw_host[offset:offset+length].tobytes()
            print(f"行{i} ({'偶数' if i % 2 == 0 else '奇数'}): {repr(string_data)}")
            
            # 奇数行で破損パターンをチェック
            if i % 2 == 1:
                # 期待されるパターン
                expected_patterns = [b'1-URGENT', b'2-HIGH', b'3-MEDIUM', b'4-NOT SPECI', b'5-LOW']
                is_valid = any(string_data.startswith(pattern) for pattern in expected_patterns)
                if not is_valid:
                    print(f"  ⚠️ 奇数行で破損検出！")
        else:
            print(f"行{i}: 無効またはNULL")

# 破損パターンの統計
print("\n=== 破損パターン分析 ===")
even_errors = 0
odd_errors = 0
sample_size = min(100000, rows)  # 最大10万行まで

for i in range(sample_size):
    length = field_lengths_host[i, col_idx]
    offset = field_offsets_host[i, col_idx]
    
    if length > 0 and offset > 0 and offset + length <= len(raw_host):
        string_data = raw_host[offset:offset+length].tobytes()
        expected_patterns = [b'1-URGENT', b'2-HIGH', b'3-MEDIUM', b'4-NOT SPECI', b'5-LOW']
        is_valid = any(string_data.startswith(pattern) for pattern in expected_patterns)
        
        if not is_valid:
            if i % 2 == 0:
                even_errors += 1
            else:
                odd_errors += 1

print(f"サンプル数: {sample_size}")
print(f"偶数行エラー: {even_errors}")
print(f"奇数行エラー: {odd_errors}")
if odd_errors > even_errors * 2:
    print("⚠️ 奇数行に顕著な破損パターンが検出されました！")