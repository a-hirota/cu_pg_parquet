#!/usr/bin/env python3
"""
GPUでのアドレス計算をシミュレート
"""

def simulate_gpu_address_calc():
    HEADER_SIZE = 19
    THREAD_STRIDE = 192
    
    print("=== GPUアドレス計算シミュレーション ===")
    print()
    
    # 問題のスレッド付近
    test_threads = [1398099, 1398100, 1398101, 1398102, 1398103]
    
    print("32ビット計算の場合:")
    for tid in test_threads:
        # 32ビット計算をシミュレート
        pos_32bit = (HEADER_SIZE + tid * THREAD_STRIDE) & 0xFFFFFFFF
        pos_64bit = HEADER_SIZE + tid * THREAD_STRIDE
        
        marker = ""
        if tid == 1398101:
            marker = " ← 最後の正常スレッド"
        elif tid == 1398102:
            marker = " ← スキップ開始"
            
        print(f"  Thread {tid}: ")
        print(f"    64bit: 0x{pos_64bit:08X} ({pos_64bit:,})")
        print(f"    32bit: 0x{pos_32bit:08X} ({pos_32bit:,}){marker}")
        
        if pos_32bit != pos_64bit:
            print(f"    → オーバーフロー検出！")
    
    print()
    print("中間計算の分析:")
    for tid in test_threads:
        # tid * strideの計算
        mul_result = tid * THREAD_STRIDE
        mul_32bit = mul_result & 0xFFFFFFFF
        
        print(f"  Thread {tid}: tid * {THREAD_STRIDE} = {mul_result:,} (0x{mul_result:08X})")
        if mul_32bit != mul_result:
            print(f"    → 32bitでオーバーフロー: 0x{mul_32bit:08X}")
    
    # より詳細な分析
    print()
    print("256MB境界付近の詳細:")
    MB_256 = 256 * 1024 * 1024
    critical_thread = (MB_256 - HEADER_SIZE) // THREAD_STRIDE
    
    print(f"  256MB = 0x{MB_256:08X}")
    print(f"  臨界スレッド: {critical_thread}")
    print(f"  臨界スレッドの位置: 0x{HEADER_SIZE + critical_thread * THREAD_STRIDE:08X}")
    
    # ビット演算の問題を確認
    print()
    print("ビット演算の影響:")
    for tid in [1398101, 1398102]:
        pos = HEADER_SIZE + tid * THREAD_STRIDE
        print(f"  Thread {tid}: 0x{pos:08X}")
        print(f"    ビット28: {(pos >> 28) & 1}")
        print(f"    ビット27-24: 0x{(pos >> 24) & 0xF:X}")

if __name__ == "__main__":
    simulate_gpu_address_calc()