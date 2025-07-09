#!/usr/bin/env python3
"""
スレッドスキップのパターンを分析
"""

def analyze_skip_pattern():
    # 問題のスレッド
    thread_before = 1398101
    thread_after = 3404032
    skip_count = thread_after - thread_before - 1
    
    print("=== スキップパターン分析 ===")
    print(f"最後の正常スレッド: {thread_before}")
    print(f"次の正常スレッド: {thread_after}")
    print(f"スキップされたスレッド数: {skip_count:,}")
    print()
    
    # スキップ数の特徴を分析
    print("スキップ数の分析:")
    print(f"  2進数: 0b{skip_count:032b}")
    print(f"  16進数: 0x{skip_count:08X}")
    print(f"  2のべき乗に近い値:")
    
    for i in range(20, 25):
        power_of_2 = 2**i
        diff = skip_count - power_of_2
        print(f"    2^{i} = {power_of_2:,}: 差 = {diff:,}")
    
    # 特定のパターンを確認
    print()
    print("特定のパターン:")
    
    # 2^21 - 2^17 のパターン
    pattern1 = 2**21 - 2**17
    print(f"  2^21 - 2^17 = {pattern1:,} (差: {skip_count - pattern1:,})")
    
    # 2^21 - 2^16 のパターン  
    pattern2 = 2**21 - 2**16
    print(f"  2^21 - 2^16 = {pattern2:,} (差: {skip_count - pattern2:,})")
    
    # より正確なパターンを探す
    print()
    print("ビットパターン分析:")
    # skip_count = 2,005,930 = 0x1E9FAA
    print(f"  skip_count + 1 = {skip_count + 1} = 0x{skip_count + 1:06X}")
    print(f"  これは 0x1E9FAB = 0b111101001111110101011")
    
    # Thread IDとメモリアドレスの関係
    print()
    print("Thread IDとメモリアドレスの関係:")
    HEADER_SIZE = 19
    THREAD_STRIDE = 192
    
    for tid in [thread_before, thread_before + 1, thread_after - 1, thread_after]:
        addr = HEADER_SIZE + tid * THREAD_STRIDE
        print(f"  Thread {tid:7d}: 0x{addr:08X} ({addr:,})")
        if addr & 0x10000000:
            print(f"    → ビット28がセット！")

if __name__ == "__main__":
    analyze_skip_pattern()