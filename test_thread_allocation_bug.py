#!/usr/bin/env python3
"""GPUカーネルのスレッド割り当て問題を詳細に分析"""

import numpy as np
import math

def analyze_thread_allocation():
    """実際のカーネル呼び出しパラメータを再現"""
    print("=== スレッド割り当て分析 ===\n")
    
    # 実際のパラメータ（16チャンク並列処理）
    total_data_size = 130 * 1024 * 1024 * 1024  # 130GB
    num_chunks = 16
    chunk_size = total_data_size // num_chunks
    
    print(f"総データサイズ: {total_data_size:,} bytes ({total_data_size/1024/1024/1024:.1f}GB)")
    print(f"チャンク数: {num_chunks}")
    print(f"チャンクサイズ: {chunk_size:,} bytes ({chunk_size/1024/1024:.1f}MB)")
    
    # カーネル設定パラメータ
    threads_per_block = 256
    max_blocks_x = 65535  # CUDAのX次元制限
    max_blocks_y = 65535  # CUDAのY次元制限
    estimated_row_size = 1150  # 推定行サイズ
    header_size = 11
    
    # 各チャンクの処理を分析
    print("\n=== チャンクごとの分析 ===")
    
    for chunk_id in range(16):
        # チャンクのデータ範囲
        data_start = chunk_id * chunk_size
        data_end = data_start + chunk_size
        data_size = chunk_size
        
        # 推定行数とターゲットスレッド数
        estimated_rows = data_size // estimated_row_size
        target_threads = estimated_rows
        
        # グリッドサイズ計算（実際のコードと同じ）
        blocks_x = min((target_threads + threads_per_block - 1) // threads_per_block, max_blocks_x)
        blocks_y = (target_threads + blocks_x * threads_per_block - 1) // (blocks_x * threads_per_block)
        
        if blocks_y > max_blocks_y:
            blocks_y = max_blocks_y
        
        actual_threads = blocks_x * blocks_y * threads_per_block
        thread_stride = (data_size + actual_threads - 1) // actual_threads
        
        if thread_stride < estimated_row_size:
            thread_stride = estimated_row_size
        
        print(f"\nチャンク {chunk_id}:")
        print(f"  データ範囲: 0x{data_start:010X} - 0x{data_end:010X}")
        print(f"  グリッド: ({blocks_x}, {blocks_y})")
        print(f"  総スレッド数: {actual_threads:,}")
        print(f"  スレッドストライド: {thread_stride:,} bytes")
        
        # 256MB境界をチェック
        chunk_start_mb = data_start // (1024 * 1024)
        chunk_end_mb = data_end // (1024 * 1024)
        boundaries_in_chunk = []
        
        for mb in range(256, int(chunk_end_mb), 256):
            if chunk_start_mb <= mb < chunk_end_mb:
                boundaries_in_chunk.append(mb)
        
        if boundaries_in_chunk:
            print(f"  256MB境界: {boundaries_in_chunk} MB")

def simulate_thread_processing():
    """スレッド1398101と3404032の処理をシミュレート"""
    print("\n\n=== 問題のスレッド処理シミュレーション ===\n")
    
    # チャンク0のパラメータ（実際の値）
    chunk_size = 8589934592  # 8GB
    header_size = 11
    thread_stride = 192  # 実際に観測された値
    
    # 問題のスレッド番号
    threads = [1398101, 1398102, 3404031, 3404032]
    
    print("チャンク0の処理:")
    print(f"  チャンクサイズ: {chunk_size:,} bytes")
    print(f"  ヘッダサイズ: {header_size} bytes")
    print(f"  スレッドストライド: {thread_stride} bytes")
    
    print("\nスレッドごとの担当範囲:")
    for tid in threads:
        start_pos = header_size + tid * thread_stride
        end_pos = header_size + (tid + 1) * thread_stride
        
        # チャンク内での相対位置
        relative_start = start_pos
        relative_end = end_pos
        
        print(f"\nThread {tid}:")
        print(f"  開始位置: 0x{start_pos:08X} ({start_pos:,} bytes, {start_pos/1024/1024:.2f}MB)")
        print(f"  終了位置: 0x{end_pos:08X} ({end_pos:,} bytes, {end_pos/1024/1024:.2f}MB)")
        print(f"  ビット28: {(start_pos >> 28) & 1}")
        
        # チャンク境界チェック
        if start_pos >= chunk_size:
            print(f"  ⚠️ このスレッドはチャンクサイズを超えています！")

def analyze_missing_threads():
    """欠落スレッドの詳細分析"""
    print("\n\n=== 欠落スレッドの詳細分析 ===\n")
    
    # 観測されたギャップ
    last_thread = 1398101
    next_thread = 3404032
    thread_stride = 192
    
    gap_threads = next_thread - last_thread - 1
    print(f"欠落スレッド範囲: Thread {last_thread + 1} から Thread {next_thread - 1}")
    print(f"欠落スレッド数: {gap_threads:,}")
    
    # 欠落範囲のアドレス
    gap_start_addr = 11 + (last_thread + 1) * thread_stride
    gap_end_addr = 11 + (next_thread - 1) * thread_stride
    
    print(f"\n欠落範囲のアドレス:")
    print(f"  開始: 0x{gap_start_addr:08X} ({gap_start_addr/1024/1024:.2f}MB)")
    print(f"  終了: 0x{gap_end_addr:08X} ({gap_end_addr/1024/1024:.2f}MB)")
    print(f"  サイズ: {gap_end_addr - gap_start_addr:,} bytes ({(gap_end_addr - gap_start_addr)/1024/1024:.2f}MB)")
    
    # 256MB境界との関係
    print(f"\n256MB境界との関係:")
    mb_256_boundary = 256 * 1024 * 1024
    for i in range(1, 4):
        boundary = i * mb_256_boundary
        if gap_start_addr <= boundary <= gap_end_addr:
            thread_at_boundary = (boundary - 11) // thread_stride
            print(f"  {i*256}MB境界 (0x{boundary:08X}): Thread {thread_at_boundary}付近")

def calculate_grid_for_threads():
    """特定のスレッド数に必要なグリッド設定を計算"""
    print("\n\n=== グリッド設定の計算 ===\n")
    
    # 3404032スレッドを実行するために必要なグリッド
    target_thread = 3404032
    threads_per_block = 256
    
    min_blocks_needed = (target_thread + 1 + threads_per_block - 1) // threads_per_block
    print(f"Thread {target_thread}を実行するために必要な最小ブロック数: {min_blocks_needed:,}")
    
    # 2Dグリッドでの設定
    max_blocks_x = 65535
    
    if min_blocks_needed <= max_blocks_x:
        print(f"  1Dグリッドで可能: blocks_x = {min_blocks_needed}")
    else:
        blocks_x = max_blocks_x
        blocks_y = (min_blocks_needed + blocks_x - 1) // blocks_x
        print(f"  2Dグリッドが必要: blocks_x = {blocks_x}, blocks_y = {blocks_y}")
        print(f"  実際のブロック数: {blocks_x * blocks_y:,}")
        print(f"  実際のスレッド数: {blocks_x * blocks_y * threads_per_block:,}")

if __name__ == "__main__":
    analyze_thread_allocation()
    simulate_thread_processing()
    analyze_missing_threads()
    calculate_grid_for_threads()