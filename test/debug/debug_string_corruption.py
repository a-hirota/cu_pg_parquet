import numpy as np
import cupy as cp
from numba import cuda
import rmm

# cudf_dev環境をアクティベート
import os
import sys

# テストデータ準備
rows = 10
cols = 3

# フィールド長データ（列2が文字列列と仮定）
field_lengths_host = np.array([
    [4, 8, 5],   # row 0: "hello"
    [4, 8, 4],   # row 1: "test"
    [4, 8, 6],   # row 2: "world\!"
    [4, 8, 3],   # row 3: "abc"
    [4, 8, 7],   # row 4: "example"
    [4, 8, 4],   # row 5: "data"
    [4, 8, 5],   # row 6: "debug"
    [4, 8, 6],   # row 7: "string"
    [4, 8, 4],   # row 8: "rows"
    [4, 8, 5],   # row 9: "tests"
], dtype=np.int32)

field_lengths_dev = cuda.to_device(field_lengths_host)

# 長さ抽出のテスト
lengths_buffer = rmm.DeviceBuffer(size=rows * 4)
lengths_cupy = cp.ndarray(
    shape=(rows,),
    dtype=cp.int32,
    memptr=cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(lengths_buffer.ptr, lengths_buffer.size, lengths_buffer),
        0
    )
)

# Numbaカーネル用に変換
d_lengths_numba = cuda.as_cuda_array(lengths_cupy)

@cuda.jit
def extract_lengths_debug(field_lengths, col_idx, lengths_out, num_rows):
    """デバッグ版長さ抽出"""
    row = cuda.grid(1)
    if row < num_rows:
        lengths_out[row] = field_lengths[row, col_idx]

# カーネル実行
threads = 256
blocks = (rows + threads - 1) // threads
print(f"Grid: {blocks} blocks x {threads} threads")

extract_lengths_debug[blocks, threads](
    field_lengths_dev, 2, d_lengths_numba, rows  # 列2（文字列列）
)
cuda.synchronize()

# 結果確認
print("\n抽出された長さ:")
lengths_host = lengths_cupy.get()  # CuPyではget()を使用
for i in range(rows):
    print(f"Row {i}: length = {lengths_host[i]} (期待値: {field_lengths_host[i, 2]})")

# オフセット計算
offsets_cumsum = cp.cumsum(lengths_cupy, dtype=cp.int32)
offsets_buffer = rmm.DeviceBuffer(size=(rows + 1) * 4)
offsets_cupy = cp.ndarray(
    shape=(rows + 1,),
    dtype=cp.int32,
    memptr=cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(offsets_buffer.ptr, offsets_buffer.size, offsets_buffer),
        0
    )
)
offsets_cupy[0] = 0
offsets_cupy[1:] = offsets_cumsum

print("\nオフセット配列:")
offsets_host = offsets_cupy.get()  # CuPyではget()を使用
for i in range(rows + 1):
    print(f"offsets[{i}] = {offsets_host[i]}")

# 奇数行のオフセットと長さ
print("\n奇数行の情報:")
for i in range(1, rows, 2):
    print(f"Row {i}: offset={offsets_host[i]}, length={lengths_host[i]}")

# さらにコピー処理をシミュレート
print("\n実際のコピー処理シミュレーション:")

# テスト用の生データ
raw_data = b"hello" + b"test" + b"world!" + b"abc" + b"example" + b"data" + b"debug" + b"string" + b"rows" + b"tests"
raw_data_np = np.frombuffer(raw_data, dtype=np.uint8)
raw_data_dev = cuda.to_device(raw_data_np)

# フィールドオフセット（簡略化）
field_offsets_host = np.zeros((rows, cols), dtype=np.int32)
cumulative_offset = 0
for i in range(rows):
    field_offsets_host[i, 2] = cumulative_offset  # 列2のオフセット
    cumulative_offset += field_lengths_host[i, 2]

field_offsets_dev = cuda.to_device(field_offsets_host)

# データコピーのテスト
total_size = int(offsets_host[-1])
data_buffer = rmm.DeviceBuffer(size=total_size)
data_cupy = cp.ndarray(
    shape=(total_size,),
    dtype=cp.uint8,
    memptr=cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(data_buffer.ptr, data_buffer.size, data_buffer),
        0
    )
)

d_data_numba = cuda.as_cuda_array(data_cupy)
d_offsets_numba = cuda.as_cuda_array(offsets_cupy)

@cuda.jit
def copy_string_data_debug(
    raw_data, field_offsets, field_lengths,
    col_idx, data_out, offsets, num_rows
):
    """デバッグ版文字列コピー"""
    row = cuda.grid(1)
    if row >= num_rows:
        return
    
    field_offset = field_offsets[row, col_idx]
    field_length = field_lengths[row, col_idx]
    output_offset = offsets[row]
    
    # NULL チェック
    if field_length <= 0:
        return
    
    # 直接コピー
    for i in range(field_length):
        src_idx = field_offset + i
        dst_idx = output_offset + i
        
        # 境界チェック
        if (src_idx < raw_data.size and 
            dst_idx < data_out.size):
            data_out[dst_idx] = raw_data[src_idx]

copy_string_data_debug[blocks, threads](
    raw_data_dev, field_offsets_dev, field_lengths_dev,
    2, d_data_numba, d_offsets_numba, rows
)
cuda.synchronize()

# コピー結果の確認
data_host = data_cupy.get()  # CuPyではget()を使用

print("\nコピー結果:")
for i in range(rows):
    start = offsets_host[i]
    end = offsets_host[i + 1]
    data_slice = data_host[start:end]
    string_value = bytes(data_slice).decode('utf-8', errors='replace')
    print(f"Row {i}: offset={start}, length={end-start}, data='{string_value}'")
    
    # 奇数行の検証
    if i % 2 == 1:
        expected_start = field_offsets_host[i, 2]
        expected_length = field_lengths_host[i, 2]
        expected_data = raw_data_np[expected_start:expected_start+expected_length]
        expected_string = bytes(expected_data).decode('utf-8', errors='replace')
        if string_value != expected_string:
            print(f"  ⚠️ 奇数行 {i} が不一致! 期待値: '{expected_string}'")
