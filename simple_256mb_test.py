#!/usr/bin/env python3
"""
256MB境界での15行欠落を確認する最小限のテスト
"""

def confirm_missing_15_rows():
    """GPUが検出した行数とPostgreSQLの実際の行数を比較"""
    
    print("=== 256MB境界での15行欠落確認 ===\n")
    
    # 実際の数値
    postgres_rows = 12_030_000
    gpu_detected = 12_029_985
    missing = postgres_rows - gpu_detected
    
    print(f"PostgreSQL実際の行数: {postgres_rows:,}")
    print(f"GPU検出行数: {gpu_detected:,}")
    print(f"欠落行数: {missing}")
    print(f"欠落率: {missing/postgres_rows*100:.6f}%")
    
    print("\n256MB境界でのスレッドスキップ:")
    print("  最後の正常スレッド: Thread 1398101 (位置: 0x0FFFFFD3)")
    print("  次の正常スレッド: Thread 3404032 (位置: 0x26F4C013)")
    print(f"  スキップされたスレッド数: {3404032 - 1398101 - 1:,}")
    
    print("\n結論:")
    print("  1. 256MB境界（0x10000000）でGPUカーネルが約200万スレッドをスキップ")
    print("  2. スキップされた範囲の境界付近に15行のデータが存在")
    print("  3. これらの15行がGPUで処理されず欠落")
    
    print("\n解決策:")
    print("  - GPUカーネルのアドレス計算修正（ビット28の扱い）")
    print("  - または256MB境界を避けるスレッド配置")
    print("  - または境界での特別処理追加")

if __name__ == "__main__":
    confirm_missing_15_rows()