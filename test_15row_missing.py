#!/usr/bin/env python3
"""15行欠落問題を最小限のコードで再現"""

import subprocess
import sys
import os

def run_single_chunk_test():
    """単一チャンクで15行欠落を確認"""
    print("=== 15行欠落問題の再現テスト ===\n")
    
    # 環境設定
    env = os.environ.copy()
    env['RUST_LOG'] = 'info'
    env['RUST_PARALLEL_CONNECTIONS'] = '16'
    
    # テーブル情報
    table_name = 'customer'
    expected_rows = 12030000
    
    print(f"テーブル: {table_name}")
    print(f"期待行数: {expected_rows:,}")
    
    # Python側のGPUパーサーを実行
    cmd = [
        'python', '-c',
        f'''
import sys
sys.path.append('/home/ubuntu/gpupgparser')
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2_lite
import cupy as cp
import numpy as np

# サンプルデータで行検出数を確認
# 実際のPostgreSQLバイナリフォーマットのヘッダ
header = bytes([
    0x50, 0x47, 0x43, 0x4F, 0x50, 0x59,  # PGCOPY
    0x0A, 0xFF, 0x0D, 0x0A, 0x00,        # \\n\\377\\r\\n\\0
    0x00, 0x00, 0x00, 0x00,              # flags
    0x00, 0x00, 0x00, 0x00               # header extension
])

# 簡単なテスト行（17列）
test_row = bytearray()
test_row.extend(b'\\x00\\x11')  # 17列

# 各フィールド（簡略版）
for i in range(17):
    test_row.extend(b'\\x00\\x00\\x00\\x04')  # 長さ4
    test_row.extend(b'TEST')  # データ

# 12,030,000行分のデータを作成するのは大きすぎるので、
# 256MB境界を跨ぐ小さなテストデータを作成
mb_256 = 256 * 1024 * 1024
test_size = mb_256 + 1024 * 1024  # 257MB

# データを作成
data = bytearray(header)
row_size = len(test_row)
num_rows = (test_size - len(header)) // row_size

print(f"テストデータサイズ: {{test_size:,}} bytes ({{test_size/1024/1024:.1f}}MB)")
print(f"行サイズ: {{row_size}} bytes")
print(f"生成行数: {{num_rows:,}}")

# 256MB境界付近に特別な行を配置
boundary_row = (mb_256 - len(header)) // row_size
print(f"\\n256MB境界付近の行: {{boundary_row}}")

# データ生成
for i in range(num_rows):
    data.extend(test_row)

# GPUでパース
try:
    # データをGPUメモリに転送
    data_gpu = cp.asarray(data, dtype=cp.uint8)
    
    # パース実行
    columns = [
        ('c_custkey', 'int32'),
        ('c_name', 'string'),
        ('c_address', 'string'),
        ('c_nationkey', 'int32'),
        ('c_phone', 'string'),
        ('c_acctbal', 'decimal'),
        ('c_mktsegment', 'string'),
        ('c_comment', 'string'),
        ('c_extra1', 'string'),
        ('c_extra2', 'string'),
        ('c_extra3', 'string'),
        ('c_extra4', 'string'),
        ('c_extra5', 'string'),
        ('c_extra6', 'string'),
        ('c_extra7', 'string'),
        ('c_extra8', 'string'),
        ('c_extra9', 'string')
    ]
    
    result = parse_binary_chunk_gpu_ultra_fast_v2_lite(
        data_gpu,
        columns=columns,
        debug=True,
        test_mode=True
    )
    
    detected_rows = result['nrow']
    print(f"\\n検出された行数: {{detected_rows:,}}")
    print(f"期待との差: {{num_rows - detected_rows}}")
    
    # 256MB境界付近の行を確認
    if 'row_positions' in result:
        positions = result['row_positions']
        print(f"\\n最初の5行の位置:")
        for i in range(min(5, len(positions))):
            print(f"  行{{i}}: 0x{{positions[i]:08X}} ({{positions[i]/1024/1024:.2f}}MB)")
        
        # 256MB境界付近を探す
        for i, pos in enumerate(positions):
            if pos > mb_256 - 1000 and pos < mb_256 + 1000:
                print(f"\\n256MB境界付近の行:")
                for j in range(max(0, i-2), min(len(positions), i+3)):
                    marker = " ← 256MB境界" if abs(positions[j] - mb_256) < 100 else ""
                    print(f"  行{{j}}: 0x{{positions[j]:08X}} ({{positions[j]/1024/1024:.2f}}MB){{marker}}")
                break
    
except Exception as e:
    print(f"エラー: {{e}}")
    import traceback
    traceback.print_exc()
'''
    ]
    
    # 実行
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.stdout:
        print("\n--- GPU Parser Output ---")
        print(result.stdout)
    
    if result.stderr:
        print("\n--- GPU Parser Errors ---")
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_single_chunk_test()
    sys.exit(0 if success else 1)