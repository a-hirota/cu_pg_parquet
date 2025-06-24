"""ワープ最適化テスト（小規模データ）"""

import numpy as np
import cupy as cp
from numba import cuda
import rmm

print("=== ワープ最適化文字列コピーテスト ===")

# テストデータ作成
rows = 10000  # 10,000行
col_idx = 0
field_offsets = np.zeros((rows, 1), dtype=np.int32)
field_lengths = np.zeros((rows, 1), dtype=np.int32)

# テスト文字列データ（各行15バイト）
test_strings = [
    b"2-HIGH         ",  # 偶数行
    b"5-LOW          ",  # 奇数行
]

# ソースデータ作成
total_size = rows * 15
raw_data = np.zeros(total_size, dtype=np.uint8)

# フィールドオフセットと長さを設定
for i in range(rows):
    field_offsets[i, 0] = i * 15
    field_lengths[i, 0] = 15
    # 偶数/奇数で異なるデータ
    string_data = test_strings[i % 2]
    raw_data[i*15:(i+1)*15] = np.frombuffer(string_data, dtype=np.uint8)

# GPU転送
raw_dev = cuda.to_device(raw_data)
field_offsets_dev = cuda.to_device(field_offsets)
field_lengths_dev = cuda.to_device(field_lengths)

# オフセット計算（累積和）
offsets = np.zeros(rows + 1, dtype=np.int32)
offsets[1:] = np.cumsum(field_lengths[:, 0])
offsets_dev = cuda.to_device(offsets)

# 出力バッファ（RMM DeviceBuffer使用）
output_size = offsets[-1]
data_buffer = rmm.DeviceBuffer(size=output_size)

# CuPy配列としてビュー作成
data_cupy = cp.ndarray(
    shape=(output_size,),
    dtype=cp.uint8,
    memptr=cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(data_buffer.ptr, data_buffer.size, data_buffer),
        0
    )
)

# Numbaカーネル用に変換
d_data_numba = cuda.as_cuda_array(data_cupy)
d_offsets_numba = cuda.as_cuda_array(offsets_dev)

@cuda.jit
def copy_string_data_warp_optimized(
    raw_data, field_offsets, field_lengths,
    col_idx, data_out, offsets, num_rows
):
    """ワープ最適化文字列コピーカーネル"""
    # ワープレベルの協調処理
    warp_id = cuda.blockIdx.x * (cuda.blockDim.x // 32) + (cuda.threadIdx.x // 32)
    lane_id = cuda.threadIdx.x % 32
    warps_per_grid = cuda.gridDim.x * (cuda.blockDim.x // 32)
    
    # 各ワープが複数行を処理（ストライド方式）
    row = warp_id
    while row < num_rows:
        if row >= num_rows:
            break
            
        field_offset = field_offsets[row, col_idx]
        field_length = field_lengths[row, col_idx]
        output_offset = offsets[row]
        
        # NULL チェック
        if field_length > 0 and field_offset >= 0:
            # ワープ内の全スレッドで協調的にコピー
            # 各スレッドが32バイトごとにストライドアクセス
            for base_idx in range(0, field_length, 32):
                i = base_idx + lane_id
                if i < field_length:
                    src_idx = field_offset + i
                    dst_idx = output_offset + i
                    
                    # 境界チェック
                    if (src_idx < raw_data.size and 
                        dst_idx < data_out.size):
                        # コアレスアクセスでコピー
                        data_out[dst_idx] = raw_data[src_idx]
        
        # 次の行へ（ワープストライド）
        row += warps_per_grid

# カーネル実行
warp_threads = 256  # 8ワープ
warp_blocks = max(1, (rows + 255) // 256)

print(f"グリッド設定: {warp_blocks} blocks × {warp_threads} threads")
print(f"ワープ数: {warp_blocks * warp_threads // 32}")

copy_string_data_warp_optimized[warp_blocks, warp_threads](
    raw_dev, field_offsets_dev, field_lengths_dev,
    col_idx, d_data_numba, d_offsets_numba, rows
)
cuda.synchronize()

# 結果確認
result = data_cupy.get()

print("\n=== 奇数/偶数行の確認 ===")
errors = 0
for i in range(min(100, rows)):
    start = offsets[i]
    end = offsets[i + 1]
    string_data = result[start:end].tobytes()
    expected = test_strings[i % 2]
    
    if string_data != expected:
        if i < 10:  # 最初の10個のエラーを表示
            print(f"行{i}: {repr(string_data)} (期待値: {repr(expected)})")
        errors += 1

if errors == 0:
    print("✅ 全ての行が正しくコピーされました！")
else:
    print(f"❌ {errors}/{rows} 行でエラーが発生しました")

# パフォーマンステスト
print("\n=== パフォーマンス確認 ===")
import time

# ウォームアップ
for _ in range(10):
    copy_string_data_warp_optimized[warp_blocks, warp_threads](
        raw_dev, field_offsets_dev, field_lengths_dev,
        col_idx, d_data_numba, d_offsets_numba, rows
    )
cuda.synchronize()

# 計測
start = time.time()
for _ in range(100):
    copy_string_data_warp_optimized[warp_blocks, warp_threads](
        raw_dev, field_offsets_dev, field_lengths_dev,
        col_idx, d_data_numba, d_offsets_numba, rows
    )
cuda.synchronize()
elapsed = time.time() - start

throughput = (output_size * 100) / elapsed / (1024**3)  # GB/s
print(f"スループット: {throughput:.2f} GB/s")
print(f"平均実行時間: {elapsed/100*1000:.3f} ms")