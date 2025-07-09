#!/usr/bin/env python3
"""256MB境界問題の簡単な確認"""

import numpy as np

def analyze_thread_gaps():
    """Thread 1398101と3404032の間のギャップを分析"""
    print("=== スレッドギャップ分析 ===\n")
    
    # 観測されたデータ
    last_thread = 1398101
    next_thread = 3404032
    thread_stride = 192  # 観測されたストライド
    
    # アドレス計算
    header_size = 11
    last_addr = header_size + last_thread * thread_stride
    next_addr = header_size + next_thread * thread_stride
    
    print(f"Thread {last_thread}:")
    print(f"  アドレス: 0x{last_addr:08X} ({last_addr:,} bytes)")
    print(f"  = {last_addr/1024/1024:.3f}MB")
    print(f"  ビット28: {(last_addr >> 28) & 1}")
    print(f"  ビット27-24: 0x{(last_addr >> 24) & 0xF:X}")
    
    print(f"\nThread {next_thread}:")
    print(f"  アドレス: 0x{next_addr:08X} ({next_addr:,} bytes)")
    print(f"  = {next_addr/1024/1024:.3f}MB")
    print(f"  ビット28: {(next_addr >> 28) & 1}")
    print(f"  ビット27-24: 0x{(next_addr >> 24) & 0xF:X}")
    
    # ギャップ
    gap_threads = next_thread - last_thread - 1
    gap_bytes = (next_thread - last_thread) * thread_stride
    
    print(f"\nギャップ:")
    print(f"  スキップされたスレッド: {gap_threads:,}")
    print(f"  スキップされたバイト: {gap_bytes:,} ({gap_bytes/1024/1024:.3f}MB)")
    
    # 256MB境界との関係
    mb_256 = 256 * 1024 * 1024
    print(f"\n256MB境界 (0x{mb_256:08X}):")
    print(f"  Thread {last_thread}からの距離: {mb_256 - last_addr:,} bytes")
    print(f"  Thread {next_thread}からの距離: {next_addr - mb_256:,} bytes")
    
    # パターン分析
    print("\n=== パターン分析 ===")
    
    # スキップ開始とスキップ終了のアドレスパターン
    skip_start = last_addr
    skip_end = next_addr
    
    print(f"\nスキップ開始: 0x{skip_start:08X}")
    print(f"バイナリ: {bin(skip_start)[2:].zfill(32)}")
    print(f"         3         2         1         0")
    print(f"         10987654321098765432109876543210")
    
    print(f"\nスキップ終了: 0x{skip_end:08X}")
    print(f"バイナリ: {bin(skip_end)[2:].zfill(32)}")
    print(f"         3         2         1         0")
    print(f"         10987654321098765432109876543210")
    
    # ビットパターンの違い
    print(f"\n異なるビット:")
    diff = skip_start ^ skip_end
    for bit in range(31, -1, -1):
        if (diff >> bit) & 1:
            print(f"  ビット{bit}: {(skip_start >> bit) & 1} → {(skip_end >> bit) & 1}")

def check_possible_causes():
    """可能な原因をチェック"""
    print("\n\n=== 可能な原因の検証 ===")
    
    # 1. 32ビット整数オーバーフロー
    print("\n1. 32ビット整数オーバーフロー:")
    max_int32 = 2**31 - 1
    max_uint32 = 2**32 - 1
    
    thread_1398101 = 1398101
    thread_3404032 = 3404032
    stride = 192
    
    calc_1398101 = thread_1398101 * stride
    calc_3404032 = thread_3404032 * stride
    
    print(f"  Thread {thread_1398101} * {stride} = {calc_1398101:,}")
    print(f"    32ビット符号付き範囲内: {calc_1398101 <= max_int32}")
    print(f"    32ビット符号なし範囲内: {calc_1398101 <= max_uint32}")
    
    print(f"  Thread {thread_3404032} * {stride} = {calc_3404032:,}")
    print(f"    32ビット符号付き範囲内: {calc_3404032 <= max_int32}")
    print(f"    32ビット符号なし範囲内: {calc_3404032 <= max_uint32}")
    
    # 2. GPU warps/blocksとの関係
    print("\n2. GPU実行単位との関係:")
    warp_size = 32
    block_size = 256
    
    print(f"  Thread {thread_1398101}:")
    print(f"    ワープ内位置: {thread_1398101 % warp_size}")
    print(f"    ブロック内位置: {thread_1398101 % block_size}")
    print(f"    ブロック番号: {thread_1398101 // block_size}")
    
    print(f"  Thread {thread_3404032}:")
    print(f"    ワープ内位置: {thread_3404032 % warp_size}")
    print(f"    ブロック内位置: {thread_3404032 % block_size}")
    print(f"    ブロック番号: {thread_3404032 // block_size}")
    
    # 3. メモリバンクとの関係
    print("\n3. GPUメモリバンクとの関係:")
    bank_size = 4  # 4バイトバンク
    num_banks = 32
    
    addr_1398101 = 11 + thread_1398101 * stride
    addr_3404032 = 11 + thread_3404032 * stride
    
    print(f"  Thread {thread_1398101} (0x{addr_1398101:08X}):")
    print(f"    バンク番号: {(addr_1398101 // bank_size) % num_banks}")
    
    print(f"  Thread {thread_3404032} (0x{addr_3404032:08X}):")
    print(f"    バンク番号: {(addr_3404032 // bank_size) % num_banks}")

def analyze_missing_key():
    """欠落したキー3483451の位置を分析"""
    print("\n\n=== 欠落キー3483451の分析 ===")
    
    # ユーザーから提供された情報
    print("欠落キー情報（ユーザー提供）:")
    print("  キー3483451が欠落")
    print("  385MBのギャップ内に存在")
    
    # 15行が欠落している場合の推定位置
    missing_rows = 15
    avg_row_size = 1150
    
    print(f"\n15行の推定データサイズ: {missing_rows * avg_row_size:,} bytes")
    
    # Thread 1398101の処理範囲
    thread_1398101 = 1398101
    stride = 192
    header = 11
    
    start_1398101 = header + thread_1398101 * stride
    end_1398101 = start_1398101 + stride
    
    print(f"\nThread {thread_1398101}の処理範囲:")
    print(f"  開始: 0x{start_1398101:08X}")
    print(f"  終了: 0x{end_1398101:08X}")
    print(f"  = {start_1398101/1024/1024:.3f}MB - {end_1398101/1024/1024:.3f}MB")

if __name__ == "__main__":
    analyze_thread_gaps()
    check_possible_causes()
    analyze_missing_key()