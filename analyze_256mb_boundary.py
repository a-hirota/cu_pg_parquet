#!/usr/bin/env python3
"""
256MB境界でのスレッドスキップ問題を詳細分析
"""

def analyze_thread_positions():
    # 定数
    HEADER_SIZE = 19
    THREAD_STRIDE = 192
    MB_256_BOUNDARY = 256 * 1024 * 1024  # 0x10000000
    
    # 問題のスレッド
    thread_before = 1398101
    thread_after = 3404032
    thread_gap = thread_after - thread_before
    
    print("=== 256MB境界でのスレッド分析 ===")
    print(f"256MB境界: 0x{MB_256_BOUNDARY:08X} ({MB_256_BOUNDARY:,} bytes)")
    print()
    
    # 境界付近のスレッドを計算
    print("境界付近のスレッド:")
    for offset in range(-5, 6):
        thread_id = thread_before + offset
        start_pos = HEADER_SIZE + thread_id * THREAD_STRIDE
        end_pos = start_pos + THREAD_STRIDE
        
        marker = ""
        if thread_id == thread_before:
            marker = " ← 最後の正常スレッド"
        elif thread_id == thread_after:
            marker = " ← 次の正常スレッド"
        elif start_pos <= MB_256_BOUNDARY < end_pos:
            marker = " ← 256MB境界を含む"
            
        print(f"  Thread {thread_id:7d}: 0x{start_pos:08X} - 0x{end_pos:08X}{marker}")
    
    print()
    print(f"スレッドギャップ: {thread_gap:,} スレッド")
    print(f"スキップされた範囲: Thread {thread_before + 1} - Thread {thread_after - 1}")
    
    # ギャップの特徴を分析
    print()
    print("ギャップの特徴:")
    print(f"  開始位置: 0x{HEADER_SIZE + (thread_before + 1) * THREAD_STRIDE:08X}")
    print(f"  終了位置: 0x{HEADER_SIZE + (thread_after - 1) * THREAD_STRIDE + THREAD_STRIDE:08X}")
    print(f"  ギャップサイズ: {thread_gap * THREAD_STRIDE:,} bytes")
    
    # 256MBを基準にした計算
    print()
    print("256MB境界を基準にした分析:")
    mb_256_thread = (MB_256_BOUNDARY - HEADER_SIZE) // THREAD_STRIDE
    print(f"  256MB境界に最も近いスレッド: {mb_256_thread}")
    print(f"  実際の境界位置: 0x{HEADER_SIZE + mb_256_thread * THREAD_STRIDE:08X}")
    print(f"  thread_beforeとの差: {mb_256_thread - thread_before}")
    print(f"  thread_afterとの差: {thread_after - mb_256_thread}")
    
    # 可能な原因の推測
    print()
    print("可能な原因:")
    print("1. 32ビット整数オーバーフロー（256MB = 0x10000000）")
    print("2. メモリアライメント制約")
    print("3. GPUメモリ割り当ての制限")
    
    # 他の境界も確認
    print()
    print("他の重要な境界:")
    boundaries = [128, 256, 384, 512, 640, 768, 896, 1024]
    for mb in boundaries:
        boundary = mb * 1024 * 1024
        boundary_thread = (boundary - HEADER_SIZE) // THREAD_STRIDE
        print(f"  {mb:4d}MB境界: Thread {boundary_thread:7d} (0x{boundary:08X})")

if __name__ == "__main__":
    analyze_thread_positions()