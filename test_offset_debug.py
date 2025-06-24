"""オフセット計算のデバッグ

奇数行でオフセットがずれているか確認
"""

import os
import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.main_postgres_to_parquet import ZeroCopyProcessor
import psycopg
import cupy as cp

print("=== オフセット計算デバッグ ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# lo_orderpriority列のインデックス
lo_orderpriority_idx = None
for i, col in enumerate(columns):
    if col.name == 'lo_orderpriority':
        lo_orderpriority_idx = i
        break

# チャンクファイル（小さいサンプル）
chunk_file = "/dev/shm/chunk_5.bin"
with open(chunk_file, 'rb') as f:
    data = f.read(10 * 1024 * 1024)  # 10MB

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

# フィールド情報をホストにコピー
field_offsets_host = field_offsets_dev.copy_to_host()
field_lengths_host = field_lengths_dev.copy_to_host()

# 文字列バッファ作成
processor = ZeroCopyProcessor()
string_buffers = processor.create_string_buffers(
    columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
)

# lo_orderpriority列のバッファ情報を取得
lo_buffer_info = string_buffers['lo_orderpriority']
offsets_buffer = lo_buffer_info['offsets']
data_buffer = lo_buffer_info['data']

# オフセット配列を取得
import rmm
if isinstance(offsets_buffer, rmm.DeviceBuffer):
    offsets_cupy = cp.ndarray(
        shape=(rows + 1,),
        dtype=cp.int32,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(offsets_buffer.ptr, offsets_buffer.size, offsets_buffer),
            0
        )
    )
else:
    offsets_ptr = offsets_buffer.__cuda_array_interface__['data'][0]
    offsets_cupy = cp.asarray(cp.ndarray(
        shape=(rows + 1,),
        dtype=cp.int32,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(offsets_ptr, (rows + 1) * 4, offsets_buffer),
            0
        )
    ))

offsets_host = offsets_cupy.get()

# データバッファを取得
if isinstance(data_buffer, rmm.DeviceBuffer):
    data_cupy = cp.ndarray(
        shape=(lo_buffer_info['actual_size'],),
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(data_buffer.ptr, data_buffer.size, data_buffer),
            0
        )
    )
else:
    data_ptr = data_buffer.__cuda_array_interface__['data'][0]
    data_cupy = cp.asarray(cp.ndarray(
        shape=(lo_buffer_info['actual_size'],),
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(data_ptr, lo_buffer_info['actual_size'], data_buffer),
            0
        )
    ))

# 最初の20行のオフセットとデータを詳細に調査
print("\n=== オフセット詳細分析 ===")
for i in range(min(20, rows)):
    # 文字列バッファ内のオフセット
    string_start = offsets_host[i]
    string_end = offsets_host[i + 1]
    string_length = string_end - string_start
    
    # 元データのオフセット
    field_offset = field_offsets_host[i, lo_orderpriority_idx]
    field_length = field_lengths_host[i, lo_orderpriority_idx]
    
    print(f"\n行{i}({'偶数' if i % 2 == 0 else '奇数'}):")
    print(f"  元データ: offset={field_offset}, length={field_length}")
    print(f"  文字列バッファ: offset={string_start}, length={string_length}")
    
    if field_length > 0 and field_length <= 100:
        # 元データの内容
        raw_data = raw_dev[field_offset:field_offset+field_length].copy_to_host()
        print(f"  元データ内容: {raw_data.tobytes()}")
        
        # 文字列バッファの内容
        if string_length > 0 and string_length <= 100:
            string_data = data_cupy[string_start:string_end].get()
            print(f"  バッファ内容: {string_data.tobytes()}")
            
            # 内容が一致するか確認
            if not np.array_equal(raw_data, string_data):
                print(f"  ⚠️ 内容不一致！")

# オフセット増分のパターンを分析
print("\n=== オフセット増分パターン ===")
print("行番号 | 増分 | 累積オフセット")
for i in range(min(10, rows)):
    if i == 0:
        increment = offsets_host[i]
        print(f"{i:6d} | {increment:4d} | {offsets_host[i]:10d}")
    else:
        increment = offsets_host[i] - offsets_host[i-1]
        print(f"{i:6d} | {increment:4d} | {offsets_host[i]:10d}")

# 奇数行と偶数行の平均長さ
even_lengths = []
odd_lengths = []
for i in range(min(1000, rows-1)):
    length = offsets_host[i+1] - offsets_host[i]
    if i % 2 == 0:
        even_lengths.append(length)
    else:
        odd_lengths.append(length)

print(f"\n偶数行の平均文字列長: {np.mean(even_lengths):.2f}")
print(f"奇数行の平均文字列長: {np.mean(odd_lengths):.2f}")