#!/usr/bin/env python3
"""
Grid境界スレッドの偶数/奇数分布を分析
"""

import json
import sys

# Grid境界デバッグ情報の例（実際のデータから抜粋）
debug_data = """
Row Position: 98567853 (奇数)
Row Position: 391168463 (奇数)
Row Position: 391301843 (奇数)
Row Position: 444955707 (奇数)
Row Position: 46149503 (奇数)
Row Position: 508865678 (偶数)
"""

# より包括的な分析
positions = [
    98567853,   # 奇数
    391168463,  # 奇数
    391301843,  # 奇数
    444955707,  # 奇数
    46149503,   # 奇数
    508865678,  # 偶数
]

print("=== Grid境界スレッドの行位置分析 ===")
even_count = 0
odd_count = 0

for pos in positions:
    parity = "偶数" if pos % 2 == 0 else "奇数"
    print(f"位置 {pos}: {parity}")
    if pos % 2 == 0:
        even_count += 1
    else:
        odd_count += 1

print(f"\n統計:")
print(f"偶数位置: {even_count}個")
print(f"奇数位置: {odd_count}個")

print("\n考察:")
print("1. GPUカーネルは偶数位置の行ヘッダも検出できている")
print("2. しかし、Grid境界スレッドのサンプルでは奇数位置が圧倒的に多い")
print("3. これは、データ内の行ヘッダの分布が奇数位置に偏っていることを示唆")

# チャンクごとの分析が必要
print("\n次の調査項目:")
print("- 各チャンクの開始位置と行ヘッダ分布の関係")
print("- チャンク分割時のオフセット計算")
print("- 偶数チャンクが実際に偶数位置から開始しているか")