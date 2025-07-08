#!/usr/bin/env python3
"""
CUDAグリッド構成を分析して、スレッドIDギャップの原因を特定
"""

def analyze_grid_config(data_size, header_size=19):
    """実際のグリッド構成を計算"""
    
    # 定数（postgres_binary_parser.pyから）
    threads_per_block = 256
    max_blocks_x = 2147483647
    max_blocks_y = 65535
    estimated_row_size = 192
    
    # データサイズ（バイト）
    print(f"データサイズ: {data_size:,} bytes")
    print(f"推定行サイズ: {estimated_row_size} bytes")
    
    # 推定行数
    estimated_rows = (data_size - header_size) // estimated_row_size
    print(f"推定行数: {estimated_rows:,}")
    
    # ターゲットスレッド数（行数の1.2倍）
    target_threads = int(estimated_rows * 1.2)
    print(f"ターゲットスレッド数: {target_threads:,}")
    
    # グリッドサイズ計算
    blocks_x = min((target_threads + threads_per_block - 1) // threads_per_block, max_blocks_x)
    blocks_y = (target_threads + blocks_x * threads_per_block - 1) // (blocks_x * threads_per_block)
    
    print(f"\nグリッド構成:")
    print(f"  blocks_x: {blocks_x:,}")
    print(f"  blocks_y: {blocks_y:,}")
    print(f"  threads_per_block: {threads_per_block}")
    
    # 実際のスレッド数
    actual_threads = blocks_x * blocks_y * threads_per_block
    print(f"  実際のスレッド数: {actual_threads:,}")
    
    # thread_stride計算
    thread_stride = (data_size + actual_threads - 1) // actual_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size
    print(f"  thread_stride: {thread_stride} bytes")
    
    # スレッドIDの計算方法を確認
    print(f"\nスレッドIDの計算:")
    print(f"  tid = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x")
    print(f"  tid = blockIdx.x * {blocks_y} * {threads_per_block} + blockIdx.y * {threads_per_block} + threadIdx.x")
    
    # 各Y層の開始スレッドID
    print(f"\n各Y層の開始スレッドID:")
    for y in range(min(blocks_y, 10)):  # 最初の10層のみ
        start_tid = 0 * blocks_y * threads_per_block + y * threads_per_block + 0
        end_tid = (blocks_x - 1) * blocks_y * threads_per_block + y * threads_per_block + (threads_per_block - 1)
        print(f"  Y={y}: {start_tid:,} - {end_tid:,}")
    
    # 問題のあるスレッドIDを分析
    print(f"\n問題のあるスレッドIDの分析:")
    problem_tids = [1048575, 3201426, 1747626, 3606663, 2097152, 3809255]
    
    for tid in problem_tids:
        # 逆算してブロック位置を特定
        total_blocks = tid // threads_per_block
        thread_in_block = tid % threads_per_block
        
        block_y = total_blocks // blocks_x
        block_x = total_blocks % blocks_x
        
        # 正しい計算
        calculated_tid = block_x * blocks_y * threads_per_block + block_y * threads_per_block + thread_in_block
        
        print(f"\n  Thread ID: {tid:,}")
        print(f"    Block位置: X={block_x}, Y={block_y}")
        print(f"    Block内スレッド: {thread_in_block}")
        print(f"    計算されたTID: {calculated_tid:,}")
        if calculated_tid != tid:
            print(f"    → 不一致！実際のTIDと計算が合わない")

def main():
    # 実際のデータサイズ（約91GB）
    data_size = 97710505856  # 例として使用
    
    print("="*80)
    print("CUDAグリッド構成の分析")
    print("="*80)
    
    analyze_grid_config(data_size)
    
    # 1048575の次が3201426になる理由を詳しく分析
    print("\n\n" + "="*80)
    print("スレッドIDギャップの詳細分析")
    print("="*80)
    
    # 1048575 = 0xFFFFF (20ビット全て1)
    # 3201426 = 0x30D592
    
    print(f"\n1048575 (0x{1048575:X}) の分析:")
    print(f"  = 4095 * 256 + 255")
    print(f"  = ブロック4095の最後のスレッド")
    
    print(f"\n3201426 (0x{3201426:X}) の分析:")
    # 3201426 / 256 = 12505.726...
    blocks_total = 3201426 // 256
    thread_offset = 3201426 % 256
    print(f"  = {blocks_total} * 256 + {thread_offset}")
    print(f"  = ブロック{blocks_total}のスレッド{thread_offset}")

if __name__ == "__main__":
    main()