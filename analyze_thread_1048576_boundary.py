#!/usr/bin/env python3
"""Thread 1048576が跨ぐ境界を詳細分析"""

def analyze_boundary():
    """Thread 1048576の境界分析"""
    print("=== Thread 1048576の境界分析 ===\n")
    
    # 64MB境界
    mb_64 = 64 * 1024 * 1024  # 67,108,864バイト = 0x04000000
    mb_192 = 192 * 1024 * 1024  # 201,326,592バイト = 0x0C000000
    mb_256 = 256 * 1024 * 1024  # 268,435,456バイト = 0x10000000
    
    # Thread位置情報
    thread_1048575_start = 0x0BFFFF53
    thread_1048575_end = 0x0C000013
    
    thread_1048576_start = 0x0C000013  # 推定（前threadの終了位置）
    thread_1048576_end = 0x0C0000D3    # 推定（後threadの開始位置）
    
    thread_1048577_start = 0x0C0000D3
    thread_1048577_end = 0x0C000193
    
    print("Thread 1048575:")
    print(f"  開始: 0x{thread_1048575_start:08X} ({thread_1048575_start:,}バイト, {thread_1048575_start/(1024*1024):.6f}MB)")
    print(f"  終了: 0x{thread_1048575_end:08X} ({thread_1048575_end:,}バイト, {thread_1048575_end/(1024*1024):.6f}MB)")
    
    print(f"\n192MB境界: 0x{mb_192:08X} ({mb_192:,}バイト)")
    
    print(f"\nThread 1048576（欠落）:")
    print(f"  推定開始: 0x{thread_1048576_start:08X} ({thread_1048576_start:,}バイト, {thread_1048576_start/(1024*1024):.6f}MB)")
    print(f"  推定終了: 0x{thread_1048576_end:08X} ({thread_1048576_end:,}バイト, {thread_1048576_end/(1024*1024):.6f}MB)")
    
    print(f"\nThread 1048577:")
    print(f"  開始: 0x{thread_1048577_start:08X} ({thread_1048577_start:,}バイト, {thread_1048577_start/(1024*1024):.6f}MB)")
    print(f"  終了: 0x{thread_1048577_end:08X} ({thread_1048577_end:,}バイト, {thread_1048577_end/(1024*1024):.6f}MB)")
    
    # 境界との関係を詳細に分析
    print("\n\n=== 境界との関係 ===")
    
    # Thread 1048575と192MB境界
    print("\nThread 1048575:")
    if thread_1048575_start < mb_192 < thread_1048575_end:
        print("  ⚠️ 192MB境界を跨いでいる！")
        print(f"    境界前: {mb_192 - thread_1048575_start:,}バイト")
        print(f"    境界後: {thread_1048575_end - mb_192:,}バイト")
    else:
        print(f"  192MB境界との距離:")
        print(f"    開始位置から境界まで: {mb_192 - thread_1048575_start:,}バイト")
        print(f"    終了位置から境界まで: {thread_1048575_end - mb_192:,}バイト")
    
    # Thread 1048576（欠落）と192MB境界
    print("\nThread 1048576（欠落）:")
    if thread_1048576_start < mb_192:
        print(f"  開始位置は192MB境界の前: 境界まで{mb_192 - thread_1048576_start:,}バイト")
    elif thread_1048576_start == mb_192:
        print("  ⚠️ 開始位置が192MB境界と一致！")
    else:
        print(f"  開始位置は192MB境界の後: 境界から{thread_1048576_start - mb_192:,}バイト")
    
    if thread_1048576_end < mb_192:
        print(f"  終了位置は192MB境界の前: 境界まで{mb_192 - thread_1048576_end:,}バイト")
    elif thread_1048576_end == mb_192:
        print("  ⚠️ 終了位置が192MB境界と一致！")
    else:
        print(f"  終了位置は192MB境界の後: 境界から{thread_1048576_end - mb_192:,}バイト")
    
    # より詳細な境界分析
    print("\n\n=== 詳細な境界分析 ===")
    
    # 各種境界を確認
    boundaries = [
        ("1MB", 1 * 1024 * 1024),
        ("4MB", 4 * 1024 * 1024),
        ("16MB", 16 * 1024 * 1024),
        ("64MB", 64 * 1024 * 1024),
        ("128MB", 128 * 1024 * 1024),
        ("192MB", 192 * 1024 * 1024),
        ("256MB", 256 * 1024 * 1024),
    ]
    
    for name, boundary in boundaries:
        # Thread 1048575の終了位置がどの境界を超えたか
        if thread_1048575_end > boundary > thread_1048575_start:
            print(f"\nThread 1048575が{name}境界を跨ぐ:")
            print(f"  境界位置: 0x{boundary:08X}")
            print(f"  開始→境界: {boundary - thread_1048575_start}バイト")
            print(f"  境界→終了: {thread_1048575_end - boundary}バイト")
    
    # 0x0C000000の特殊性を確認
    print("\n\n=== 0x0C000000（192MB）の特殊性 ===")
    print(f"192MB = {mb_192:,}バイト = 0x{mb_192:08X}")
    print(f"  = 3 * 64MB")
    print(f"  = 48 * 4MB")
    print(f"  = 192 * 1MB")
    
    # Thread 1048575の終了位置の詳細
    print(f"\nThread 1048575終了位置: 0x0C000013")
    print(f"  = 192MB + 0x13（19バイト）")
    print(f"  = 192MB境界を19バイト超えた位置")
    
    # Thread 1048577の開始位置の詳細  
    print(f"\nThread 1048577開始位置: 0x0C0000D3")
    print(f"  = 192MB + 0xD3（211バイト）")
    print(f"  = 192MB境界を211バイト超えた位置")
    
    # ギャップの詳細
    gap_size = thread_1048577_start - thread_1048575_end
    print(f"\nギャップ（Thread 1048576が処理すべき範囲）:")
    print(f"  サイズ: {gap_size}バイト（0x{gap_size:02X}）")
    print(f"  位置: 192MB + 19バイト 〜 192MB + 211バイト")

if __name__ == "__main__":
    analyze_boundary()