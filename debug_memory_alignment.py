import numpy as np
import cupy as cp
from numba import cuda
import rmm

print("メモリアライメント問題の調査")
print("=" * 50)

# GPUのワープサイズ（通常32）
WARP_SIZE = 32

# テストケース：偶数行と奇数行でのメモリアクセスパターン
rows = 16  # 十分な行数でテスト
cols = 3

# フィールド長データ（実際のデータをシミュレート）
field_lengths_host = np.zeros((rows, cols), dtype=np.int32)
field_lengths_host[:, 0] = 4  # INT
field_lengths_host[:, 1] = 8  # BIGINT

# 文字列長をパターン化
string_lengths = [5, 4, 6, 3, 7, 4, 5, 6, 4, 5, 6, 7, 3, 4, 5, 6]
for i in range(rows):
    field_lengths_host[i, 2] = string_lengths[i]

# フィールドオフセット（簡略化）
field_offsets_host = np.zeros((rows, cols), dtype=np.int32)
cumulative = 0
for i in range(rows):
    field_offsets_host[i, 2] = cumulative
    cumulative += string_lengths[i]

# GPUに転送
field_lengths_dev = cuda.to_device(field_lengths_host)
field_offsets_dev = cuda.to_device(field_offsets_host)

# メモリアクセスパターンを記録するカーネル
access_pattern = cuda.device_array((rows, 5), dtype=np.int32)

@cuda.jit
def analyze_memory_access(field_lengths, field_offsets, col_idx, access_info, num_rows):
    """メモリアクセスパターンを分析"""
    row = cuda.grid(1)
    if row >= num_rows:
        return
    
    # スレッド情報
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    block_id = cuda.blockIdx.x
    
    # アクセス情報を記録
    access_info[row, 0] = row
    access_info[row, 1] = warp_id
    access_info[row, 2] = lane_id
    access_info[row, 3] = field_lengths[row, col_idx]
    access_info[row, 4] = row % 2  # 偶数/奇数フラグ

# 異なるグリッド設定でテスト
print("テスト1: threads=32 (1ワープ)")
threads = 32
blocks = (rows + threads - 1) // threads
analyze_memory_access[blocks, threads](
    field_lengths_dev, field_offsets_dev, 2, access_pattern, rows
)
cuda.synchronize()

pattern1 = access_pattern.copy_to_host()
print("\nワープ内のメモリアクセスパターン:")
print("Row  < /dev/null |  Warp | Lane | Length | Even/Odd")
print("-" * 40)
for i in range(rows):
    even_odd = "Even" if pattern1[i, 4] == 0 else "Odd"
    print(f"{pattern1[i, 0]:3d} | {pattern1[i, 1]:4d} | {pattern1[i, 2]:4d} | {pattern1[i, 3]:6d} | {even_odd}")

# ワープごとのアクセスパターン分析
print("\n\nワープごとの偶数/奇数行の分布:")
warp_even_odd = {}
for i in range(rows):
    warp = i // WARP_SIZE
    if warp not in warp_even_odd:
        warp_even_odd[warp] = {"even": [], "odd": []}
    
    if i % 2 == 0:
        warp_even_odd[warp]["even"].append(i)
    else:
        warp_even_odd[warp]["odd"].append(i)

for warp, data in warp_even_odd.items():
    print(f"Warp {warp}:")
    print(f"  偶数行: {data['even']}")
    print(f"  奇数行: {data['odd']}")

# メモリコアレッシングの分析
print("\n\nメモリコアレッシング分析:")
print("2D配列field_lengths[row, col]のメモリレイアウト:")
print(f"  Shape: {field_lengths_host.shape}")
print(f"  Strides: {field_lengths_host.strides}")
print(f"  Item size: {field_lengths_host.itemsize} bytes")

# 連続する行へのアクセス
print("\n連続する行へのアクセス時のメモリアドレス差:")
for i in range(min(8, rows-1)):
    addr_diff = field_lengths_host.strides[0]
    print(f"  Row {i} → Row {i+1}: {addr_diff} bytes")

# 問題の可能性
print("\n\n考えられる問題:")
print("1. 2D配列のストライドが大きい場合、連続する行へのアクセスがコアレッシングされない")
print("2. 奇数行と偶数行で異なるメモリバンクにアクセスする可能性")
print("3. ワープ内のスレッドが離れた行にアクセスする場合、キャッシュ効率が低下")

# 最適化された1D配列アクセスのシミュレーション
print("\n\n最適化案: 列ごとの1D配列を使用")
lengths_1d = field_lengths_host[:, 2].copy()  # 文字列長のみの1D配列
print(f"1D配列のストライド: {lengths_1d.strides[0]} bytes")
print("→ 連続するスレッドが連続するメモリにアクセス（完全コアレッシング）")

# RMM DeviceBufferでのアライメント確認
print("\n\nRMM DeviceBufferのアライメント:")
test_buffer = rmm.DeviceBuffer(size=rows * 4)
print(f"Buffer ptr: {test_buffer.ptr}")
print(f"Buffer size: {test_buffer.size}")
print(f"Alignment (ptr % 128): {test_buffer.ptr % 128}")
print("→ RMM DeviceBufferは通常128バイト境界にアライメントされる")
