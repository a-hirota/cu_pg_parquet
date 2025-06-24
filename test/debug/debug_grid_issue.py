import numpy as np
import cupy as cp
from numba import cuda
import rmm

# optimize_grid_size の動作確認
def optimize_grid_size_test(data_size, num_rows, device_props):
    """GPU特性に基づいたGrid/Blockサイズを計算（テスト版）"""
    
    # デバイス特性の取得
    max_threads_per_block = device_props.get('MAX_THREADS_PER_BLOCK', 1024)
    multiprocessor_count = device_props.get('MULTIPROCESSOR_COUNT', 16)
    max_blocks_per_grid = device_props.get('MAX_GRID_DIM_X', 65535)
    
    # スレッド数の計算
    threads = 256  # 256 = 8ワープ
    
    # 行数ベースのブロック数計算
    if num_rows > 0:
        blocks_for_rows = (num_rows + threads - 1) // threads
    else:
        blocks_for_rows = 1
    
    # データサイズベースのブロック数計算（行検出用）
    stride_size = 1024
    blocks_for_data = (data_size + stride_size - 1) // stride_size
    
    # 最小でもSM数の倍数のブロックを確保
    min_blocks = multiprocessor_count * 2  # SMあたり2ブロック
    
    # 最終的なブロック数の決定
    final_blocks = max(min_blocks, blocks_for_rows, blocks_for_data // 4)
    final_blocks = min(final_blocks, max_blocks_per_grid)
    
    return final_blocks, threads

# テストケース
device_props = {
    'MAX_THREADS_PER_BLOCK': 1024,
    'MULTIPROCESSOR_COUNT': 16,
    'MAX_GRID_DIM_X': 65535,
    'GLOBAL_MEMORY': 8 * 1024**3,
    'SHARED_MEMORY_PER_BLOCK': 48 * 1024,
    'WARP_SIZE': 32
}

print("Grid/Block計算テスト")
print("=" * 50)

test_cases = [
    (10, "10行"),
    (100, "100行"),
    (1000, "1000行"),
    (10000, "10000行"),
    (100000, "100000行"),
]

for rows, desc in test_cases:
    blocks, threads = optimize_grid_size_test(0, rows, device_props)
    total_threads = blocks * threads
    print(f"\n{desc}:")
    print(f"  Blocks: {blocks}, Threads: {threads}")
    print(f"  Total threads: {total_threads}")
    print(f"  スレッド/行: {total_threads / rows:.2f}")
    
    # 各スレッドの処理範囲を計算
    if total_threads > rows:
        print(f"  ⚠️ 警告: スレッド数({total_threads})が行数({rows})より多い")
        print(f"  → 一部のスレッドは処理する行がない")
    
    # スレッドからグローバルインデックスへのマッピングを確認
    print(f"  スレッド割り当て例:")
    for thread_idx in range(min(5, total_threads)):
        row_idx = thread_idx
        if row_idx < rows:
            print(f"    Thread {thread_idx}: Row {row_idx}")
        else:
            print(f"    Thread {thread_idx}: (処理なし)")

# 実際のカーネル動作をシミュレート
print("\n\n実際のカーネル動作シミュレーション")
print("=" * 50)

@cuda.jit
def test_kernel(data_array, output_array, num_rows):
    """テスト用カーネル"""
    row = cuda.grid(1)
    if row < num_rows:
        # 各スレッドが処理する行番号を記録
        output_array[row] = row * 10 + cuda.threadIdx.x

# 10行でテスト
test_rows = 10
blocks, threads = optimize_grid_size_test(0, test_rows, device_props)

# データ準備
data_dev = cuda.device_array(test_rows, dtype=np.int32)
output_dev = cuda.device_array(test_rows, dtype=np.int32)

# カーネル実行
print(f"\n{test_rows}行のテスト実行:")
print(f"Grid: {blocks} blocks × {threads} threads")

test_kernel[blocks, threads](data_dev, output_dev, test_rows)
cuda.synchronize()

# 結果確認
output_host = output_dev.copy_to_host()
print("\n各行の処理結果:")
for i in range(test_rows):
    print(f"  Row {i}: value = {output_host[i]}")
    
# 奇数行のチェック
print("\n奇数行の値:")
for i in range(1, test_rows, 2):
    print(f"  Row {i}: value = {output_host[i]}")

# グリッド計算の問題点分析
print("\n\n問題点分析:")
print("1. cuda.grid(1) は正しくグローバルスレッドインデックスを計算")
print("2. 各スレッドは正しい行インデックスにアクセス")
print("3. 問題は他の場所にある可能性:")
print("   - メモリアクセスパターン")
print("   - カーネル内のバグ")
print("   - データの初期化")
