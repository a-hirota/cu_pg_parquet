#!/usr/bin/env python3
"""
ビット28のパターンと欠落の関係を分析
"""

def analyze_bit28():
    print("=== ビット28のパターン分析 ===")
    print()
    
    # ビット28（0x10000000）の意味
    print("ビット28の意味:")
    print("  0x10000000 = 256MB")
    print("  0x20000000 = 512MB") 
    print("  0x30000000 = 768MB")
    print("  0x40000000 = 1GB")
    print()
    
    # ビットパターンを確認
    print("各境界のビットパターン:")
    for mb in range(128, 1537, 128):
        addr = mb * 1024 * 1024
        bit31_28 = (addr >> 28) & 0xF
        bit27_24 = (addr >> 24) & 0xF
        
        print(f"  {mb:4d}MB: 0x{addr:08X} - ビット31-28: 0b{bit31_28:04b}, ビット27-24: 0x{bit27_24:X}")
        
        # ビット28とビット29の組み合わせ
        if mb == 256:
            print("         → ビット28が初めて1になる")
        elif mb == 768:
            print("         → ビット28,29が両方1になる")
    
    print()
    print("パターンの解釈:")
    print("  256MB (0x10000000): ビット28=1, ビット29=0")
    print("  512MB (0x20000000): ビット28=0, ビット29=1")
    print("  768MB (0x30000000): ビット28=1, ビット29=1")
    print()
    
    # 欠落15個の分布を推測
    print("15個の欠落の可能性:")
    print("  1. 256MB境界で大きなスキップ（200万スレッド）")
    print("  2. その中に15行分のデータが含まれていた")
    print("  3. または、スキップ範囲の両端で複数行が欠落")
    
    # スレッド1398101付近の詳細
    print()
    print("スレッド1398101付近の行数推定:")
    THREAD_STRIDE = 192
    AVG_ROW_SIZE = 141
    
    # 各スレッドが処理する行数を推定
    rows_per_thread = THREAD_STRIDE / AVG_ROW_SIZE
    print(f"  平均行サイズ: {AVG_ROW_SIZE} bytes")
    print(f"  スレッドストライド: {THREAD_STRIDE} bytes")
    print(f"  スレッドあたり行数: {rows_per_thread:.2f}")
    
    # スキップされた200万スレッドに含まれる行数
    skipped_threads = 2005930
    estimated_skipped_rows = skipped_threads * rows_per_thread
    print(f"  スキップされたスレッド: {skipped_threads:,}")
    print(f"  推定スキップ行数: {estimated_skipped_rows:,.0f}")

if __name__ == "__main__":
    analyze_bit28()