#!/usr/bin/env python3
"""
スレッドギャップの根本原因を分析
"""

def analyze_thread_id_pattern():
    """スレッドIDのパターンを分析"""
    
    # 問題のあるスレッドIDペア
    thread_pairs = [
        (1048575, 3201426),  # 476029欠落
        (1747626, 3606663),  # 1227856欠落
        (2097152, 3809255),  # 1979731欠落
    ]
    
    print("スレッドIDパターンの分析:")
    print("="*80)
    
    for before, after in thread_pairs:
        print(f"\nThread {before:,} → {after:,}")
        print(f"  16進数: 0x{before:X} → 0x{after:X}")
        print(f"  2進数:")
        print(f"    前: {before:024b}")
        print(f"    後: {after:024b}")
        
        # ビットパターンを分析
        if before == 1048575:
            print("  → 0xFFFFF (20ビット全て1)")
        elif before == 2097151:
            print("  → 0x1FFFFF (21ビット全て1)")
        elif before == 2097152:
            print("  → 0x200000 (21ビット目のみ1)")
        
        # 差を分析
        diff = after - before
        print(f"  差: {diff:,} (0x{diff:X})")
        
        # ブロック番号を計算（256スレッド/ブロック）
        block_before = before // 256
        block_after = after // 256
        print(f"  ブロック: {block_before:,} → {block_after:,} (差: {block_after - block_before:,})")

def simulate_thread_assignment():
    """CUDAカーネルのスレッド割り当てをシミュレート"""
    
    print("\n\nスレッド割り当てシミュレーション:")
    print("="*80)
    
    # 実際のパラメータ（推定）
    data_size = 97710505856
    header_size = 19
    thread_stride = 192
    threads_per_block = 256
    
    # 問題のある位置を計算
    problematic_positions = [
        0x0C000013,  # Thread 1048575の終了位置
        0x24A32D93,  # Thread 3201426の開始位置
    ]
    
    for pos in problematic_positions:
        # どのスレッドがこの位置を担当すべきか
        relative_pos = pos - header_size
        thread_id = relative_pos // thread_stride
        thread_start = header_size + thread_id * thread_stride
        thread_end = header_size + (thread_id + 1) * thread_stride
        
        print(f"\n位置 0x{pos:08X}:")
        print(f"  担当すべきスレッド: {thread_id:,}")
        print(f"  スレッド範囲: 0x{thread_start:08X} - 0x{thread_end:08X}")
        print(f"  ブロック: {thread_id // 256:,}")

def check_grid_calculation():
    """グリッド計算の問題をチェック"""
    
    print("\n\nグリッド計算の検証:")
    print("="*80)
    
    # CUDAカーネルの定数
    threads_per_block = 256
    max_blocks_x = 2147483647
    max_blocks_y = 65535
    
    # 実際のデータサイズ
    data_size = 97710505856
    header_size = 19
    estimated_row_size = 192
    
    # 推定行数とターゲットスレッド数
    estimated_rows = (data_size - header_size) // estimated_row_size
    target_threads = int(estimated_rows * 1.2)
    
    print(f"推定行数: {estimated_rows:,}")
    print(f"ターゲットスレッド数: {target_threads:,}")
    
    # 実際の計算
    blocks_x = min((target_threads + threads_per_block - 1) // threads_per_block, max_blocks_x)
    blocks_y = (target_threads + blocks_x * threads_per_block - 1) // (blocks_x * threads_per_block)
    
    print(f"\nグリッド構成:")
    print(f"  blocks_x: {blocks_x:,}")
    print(f"  blocks_y: {blocks_y:,}")
    
    actual_threads = blocks_x * blocks_y * threads_per_block
    print(f"  実際のスレッド数: {actual_threads:,}")
    
    # スレッドIDの最大値
    max_thread_id = actual_threads - 1
    print(f"  最大スレッドID: {max_thread_id:,} (0x{max_thread_id:X})")
    
    # 問題: 実際のデータでは4,417,212までしかスレッドIDがない
    print(f"\n実際のデータの最大スレッドID: 4,417,212")
    print(f"計算された最大スレッドID: {max_thread_id:,}")
    print(f"差: {max_thread_id - 4417212:,}")

if __name__ == "__main__":
    analyze_thread_id_pattern()
    simulate_thread_assignment()
    check_grid_calculation()