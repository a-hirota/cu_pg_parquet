import numpy as np
import cupy as cp
from numba import cuda
import rmm

# 実際のcreate_string_buffersの動作を再現
print("文字列バッファ作成の詳細分析")
print("=" * 50)

# テストデータ（実際のPostgreSQLバイナリデータを模擬）
# 10行のデータ、各行に異なる長さの文字列
test_strings = [
    b"hello",     # row 0: 5 bytes
    b"test",      # row 1: 4 bytes  
    b"world!",    # row 2: 6 bytes
    b"abc",       # row 3: 3 bytes
    b"example",   # row 4: 7 bytes
    b"data",      # row 5: 4 bytes
    b"debug",     # row 6: 5 bytes
    b"string",    # row 7: 6 bytes
    b"rows",      # row 8: 4 bytes
    b"tests",     # row 9: 5 bytes
]

# フィールド長データ（3列: INT, BIGINT, VARCHAR）
rows = 10
cols = 3
field_lengths_host = np.zeros((rows, cols), dtype=np.int32)
field_lengths_host[:, 0] = 4  # INT: 4 bytes
field_lengths_host[:, 1] = 8  # BIGINT: 8 bytes
# VARCHARの長さ
for i, s in enumerate(test_strings):
    field_lengths_host[i, 2] = len(s)

# フィールドオフセット（実際のバイナリ位置）
field_offsets_host = np.zeros((rows, cols), dtype=np.int32)
current_pos = 0
for i in range(rows):
    # 各行のヘッダー（2バイト）+ 各フィールドの長さヘッダー（4バイト×3）
    row_header_size = 2 + 4 * cols
    field_offsets_host[i, 0] = current_pos + row_header_size
    field_offsets_host[i, 1] = field_offsets_host[i, 0] + field_lengths_host[i, 0]
    field_offsets_host[i, 2] = field_offsets_host[i, 1] + field_lengths_host[i, 1]
    # 次の行の開始位置
    current_pos += row_header_size + field_lengths_host[i, :].sum()

# 生データの作成
total_size = current_pos
raw_data = bytearray(total_size)
for i in range(rows):
    # 文字列データをコピー
    string_offset = field_offsets_host[i, 2]
    string_data = test_strings[i]
    raw_data[string_offset:string_offset+len(string_data)] = string_data

# GPUにデータを転送
raw_dev = cuda.to_device(np.frombuffer(raw_data, dtype=np.uint8))
field_offsets_dev = cuda.to_device(field_offsets_host)
field_lengths_dev = cuda.to_device(field_lengths_host)

print("フィールド情報:")
for i in range(rows):
    print(f"Row {i}: offset={field_offsets_host[i, 2]}, length={field_lengths_host[i, 2]}, string='{test_strings[i].decode()}'")

# 実際のcreate_string_buffersの処理を再現
print("\n\ncreate_string_buffers処理の再現:")

# 1. 長さ抽出
lengths_buffer = rmm.DeviceBuffer(size=rows * 4)
lengths_cupy = cp.ndarray(
    shape=(rows,),
    dtype=cp.int32,
    memptr=cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(lengths_buffer.ptr, lengths_buffer.size, lengths_buffer),
        0
    )
)

# デバッグ用カーネル：長さ抽出の詳細を記録
debug_info = cuda.device_array((rows, 4), dtype=np.int32)  # row, col_idx, field_length, thread_id

@cuda.jit
def extract_lengths_debug(field_lengths, col_idx, lengths_out, debug_out, num_rows):
    """デバッグ版長さ抽出"""
    row = cuda.grid(1)
    if row < num_rows:
        lengths_out[row] = field_lengths[row, col_idx]
        # デバッグ情報を記録
        debug_out[row, 0] = row
        debug_out[row, 1] = col_idx
        debug_out[row, 2] = field_lengths[row, col_idx]
        debug_out[row, 3] = cuda.threadIdx.x

# カーネル実行
threads = 256
blocks = (rows + threads - 1) // threads
d_lengths_numba = cuda.as_cuda_array(lengths_cupy)

extract_lengths_debug[blocks, threads](
    field_lengths_dev, 2, d_lengths_numba, debug_info, rows
)
cuda.synchronize()

# デバッグ情報の確認
debug_host = debug_info.copy_to_host()
print("\n長さ抽出のデバッグ情報:")
for i in range(rows):
    print(f"Row {i}: col_idx={debug_host[i, 1]}, length={debug_host[i, 2]}, thread={debug_host[i, 3]}")

# 2. オフセット計算
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

print("\nオフセット計算結果:")
offsets_host = offsets_cupy.get()
for i in range(rows + 1):
    print(f"offsets[{i}] = {offsets_host[i]}")

# 3. データコピー
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

# デバッグ用コピーカーネル
copy_debug = cuda.device_array((rows, 5), dtype=np.int32)  # row, field_offset, field_length, output_offset, bytes_copied

@cuda.jit
def copy_string_data_debug(
    raw_data, field_offsets, field_lengths,
    col_idx, data_out, offsets, debug_out, num_rows
):
    """デバッグ版文字列コピー"""
    row = cuda.grid(1)
    if row >= num_rows:
        return
    
    field_offset = field_offsets[row, col_idx]
    field_length = field_lengths[row, col_idx]
    output_offset = offsets[row]
    
    # デバッグ情報を記録
    debug_out[row, 0] = row
    debug_out[row, 1] = field_offset
    debug_out[row, 2] = field_length
    debug_out[row, 3] = output_offset
    
    # NULL チェック
    if field_length <= 0:
        debug_out[row, 4] = -1
        return
    
    # 実際にコピーしたバイト数
    bytes_copied = 0
    
    # 直接コピー
    for i in range(field_length):
        src_idx = field_offset + i
        dst_idx = output_offset + i
        
        # 境界チェック
        if (src_idx < raw_data.size and 
            dst_idx < data_out.size):
            data_out[dst_idx] = raw_data[src_idx]
            bytes_copied += 1
    
    debug_out[row, 4] = bytes_copied

d_data_numba = cuda.as_cuda_array(data_cupy)
d_offsets_numba = cuda.as_cuda_array(offsets_cupy)

copy_string_data_debug[blocks, threads](
    raw_dev, field_offsets_dev, field_lengths_dev,
    2, d_data_numba, d_offsets_numba, copy_debug, rows
)
cuda.synchronize()

# コピーデバッグ情報の確認
copy_debug_host = copy_debug.copy_to_host()
print("\n\nコピー処理のデバッグ情報:")
for i in range(rows):
    print(f"Row {i}: field_offset={copy_debug_host[i, 1]}, field_length={copy_debug_host[i, 2]}, " +
          f"output_offset={copy_debug_host[i, 3]}, bytes_copied={copy_debug_host[i, 4]}")

# 結果の検証
data_host = data_cupy.get()
print("\n\n最終結果の検証:")
for i in range(rows):
    start = offsets_host[i]
    end = offsets_host[i + 1]
    data_slice = data_host[start:end]
    string_value = bytes(data_slice).decode('utf-8', errors='replace')
    expected = test_strings[i].decode()
    
    status = "✓" if string_value == expected else "✗"
    print(f"Row {i}: '{string_value}' (期待値: '{expected}') {status}")
    
    if i % 2 == 1 and string_value != expected:
        print(f"  ⚠️ 奇数行 {i} で不一致!")
        print(f"  実際のバイト: {list(data_slice)}")
        print(f"  期待のバイト: {list(test_strings[i])}")
