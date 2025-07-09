#!/usr/bin/env python3
"""
256MB境界でのスレッドスキップを単純に確認
"""

def analyze_256mb_skip():
    """256MB境界でのスレッドスキップを分析"""
    
    HEADER_SIZE = 19
    THREAD_STRIDE = 192
    MB_256 = 256 * 1024 * 1024
    
    print("=== 256MB境界スレッドスキップ分析 ===\n")
    
    # customerテーブルでの実際の欠落
    print("実際の欠落パターン:")
    print("  欠落キー: 3483451")
    print("  前のスレッド: Thread 1398101")
    print("  次のスレッド: Thread 3404032")
    print("  スキップ: 2,005,931 スレッド")
    print()
    
    # スレッド位置の計算
    print("スレッド位置の計算:")
    for tid in [1398099, 1398100, 1398101, 1398102, 1398103, 3404030, 3404031, 3404032]:
        start_pos = HEADER_SIZE + tid * THREAD_STRIDE
        end_pos = start_pos + THREAD_STRIDE
        
        # ビット解析
        bit28 = (start_pos >> 28) & 1
        bit29 = (start_pos >> 29) & 1
        bit30 = (start_pos >> 30) & 1
        
        marker = ""
        if tid == 1398101:
            marker = " ← 最後の検出"
        elif tid == 3404032:
            marker = " ← 次の検出"
        elif tid == 1398102:
            marker = " ← スキップ開始"
        elif tid == 3404031:
            marker = " ← スキップ終了"
            
        print(f"  Thread {tid:7d}: 0x{start_pos:08X} - 0x{end_pos:08X}")
        print(f"              ビット30-28: {bit30}{bit29}{bit28}b{marker}")
    
    print()
    
    # パターン分析
    print("パターン分析:")
    print("  256MB = 0x10000000 (ビット28がセット)")
    print("  Thread 1398101: 0x0FFFFFD3 (ビット28=0)")
    print("  Thread 1398102: 0x10000093 (ビット28=1)")
    print()
    
    # 推測される条件
    print("推測されるGPUカーネルの問題:")
    print("  1. ビット28の変化（0→1）を検出")
    print("  2. 何らかの条件分岐でスレッドをスキップ")
    print("  3. 次にビット28が0になるまでスキップ継続")
    print()
    
    # スキップ範囲の計算
    skip_start = HEADER_SIZE + 1398102 * THREAD_STRIDE
    skip_end = HEADER_SIZE + 3404031 * THREAD_STRIDE
    print(f"スキップ範囲:")
    print(f"  開始: 0x{skip_start:08X} ({skip_start:,} bytes)")
    print(f"  終了: 0x{skip_end:08X} ({skip_end:,} bytes)")
    print(f"  サイズ: {skip_end - skip_start:,} bytes ({(skip_end - skip_start)/1024/1024:.1f} MB)")
    
    # 15行の欠落がどこにあるか
    print()
    print("15行の欠落の位置:")
    print("  スキップ開始付近（Thread 1398101の最後の方）")
    print("  または、スキップ終了付近（Thread 3404032の最初の方）")
    print("  実際のキー3483451は、Thread 1398101が処理すべきだった")

if __name__ == "__main__":
    analyze_256mb_skip()