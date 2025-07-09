#!/usr/bin/env python3
"""
全ての重要な境界でスレッドスキップがあるか確認
"""

def check_boundaries():
    HEADER_SIZE = 19
    THREAD_STRIDE = 192
    
    print("=== 各境界でのスレッドスキップ確認 ===")
    print()
    
    # 欠落が15個あるので、各境界で1つずつスキップがあるか確認
    boundaries_mb = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]
    
    skip_points = []
    
    for mb in boundaries_mb:
        boundary = mb * 1024 * 1024
        boundary_thread = (boundary - HEADER_SIZE) // THREAD_STRIDE
        
        # 境界前後のスレッド
        before_thread = boundary_thread
        after_pos = HEADER_SIZE + (before_thread + 1) * THREAD_STRIDE
        
        print(f"{mb:4d}MB境界 (0x{boundary:08X}):")
        print(f"  境界直前スレッド: {before_thread}")
        print(f"  次スレッドの位置: 0x{after_pos:08X}")
        
        # ビット28の変化を確認
        before_bit28 = (HEADER_SIZE + before_thread * THREAD_STRIDE) & 0x10000000
        after_bit28 = after_pos & 0x10000000
        
        if before_bit28 == 0 and after_bit28 != 0:
            print(f"  → ビット28が変化！スキップの可能性")
            skip_points.append(mb)
        
        # より詳細な境界確認
        if mb % 256 == 0:  # 256MBの倍数
            nth_256mb = mb // 256
            print(f"  → {nth_256mb}番目の256MB境界")
    
    print()
    print(f"スキップの可能性がある境界: {len(skip_points)}箇所")
    for mb in skip_points:
        print(f"  - {mb}MB")
    
    # 15個の欠落との関係
    print()
    print("15個の欠落との関係:")
    print(f"  発見されたスキップ点: {len(skip_points)}箇所")
    print(f"  欠落行数: 15")
    
    if len(skip_points) > 0:
        # 各スキップ点でのスキップ幅を推定
        print()
        print("各境界でのスキップ幅（推定）:")
        for i, mb in enumerate(skip_points):
            if i < len(skip_points) - 1:
                next_mb = skip_points[i + 1]
                skip_size = (next_mb - mb) * 1024 * 1024 // THREAD_STRIDE
                print(f"  {mb}MB境界: 約{skip_size:,}スレッド")

if __name__ == "__main__":
    check_boundaries()