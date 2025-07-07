#!/usr/bin/env python3
"""
bpchar修正の動作確認
"""

import os
import sys
sys.path.append('/home/ubuntu/gpupgparser')

from src.readPostgres.metadata import fetch_column_meta
from src.cuda_kernels.postgres_binary_parser import estimate_row_size_from_columns
import psycopg

# PostgreSQL接続
dsn = os.environ.get("GPUPASER_PG_DSN", "")
conn = psycopg.connect(dsn)

# customerテーブルのメタデータ取得
print("=== 修正後のメタデータ取得 ===\n")
columns = fetch_column_meta(conn, "SELECT * FROM customer")

print(f"{'列名':<15} {'Arrow型':<10} {'elem_size':<10} {'arrow_param':<15}")
print("-" * 60)

for col in columns:
    arrow_name = {
        0: "INT16", 1: "INT32", 2: "INT64", 
        3: "FLOAT32", 4: "FLOAT64", 5: "DECIMAL128",
        6: "UTF8", 7: "BINARY", 8: "DATE32", 
        9: "TS64_US", 10: "BOOL", 255: "UNKNOWN"
    }.get(col.arrow_id, f"ID:{col.arrow_id}")
    
    print(f"{col.name:<15} {arrow_name:<10} {col.elem_size:<10} {str(col.arrow_param):<15}")

# 行サイズ推定（詳細デバッグ）
print("\n=== 修正後の行サイズ推定（詳細） ===")
size = 2  # フィールド数
print(f"フィールド数: 2バイト")

for col in columns:
    col_size = 0  # 4はあとで加算
    
    if col.elem_size > 0:
        col_size = col.elem_size
        print(f"{col.name}: 4 + {col.elem_size} = {4 + col_size}バイト (固定長)")
    else:
        if col.arrow_id == 6:  # UTF8
            if col.arrow_param is not None and isinstance(col.arrow_param, int):
                col_size = col.arrow_param
                print(f"{col.name}: 4 + {col.arrow_param} = {4 + col_size}バイト (bpchar)")
            else:
                col_size = 20
                print(f"{col.name}: 4 + 20 = {4 + col_size}バイト (varchar推定)")
        elif col.arrow_id == 5:  # DECIMAL128
            col_size = 16
            print(f"{col.name}: 4 + 16 = {4 + col_size}バイト (DECIMAL128)")
        else:
            col_size = 4
            print(f"{col.name}: 4 + 4 = {4 + col_size}バイト (その他)")
    
    size += 4 + col_size

aligned_size = ((size + 31) // 32) * 32
print(f"\n合計: {size}バイト → 32バイト整列: {aligned_size}バイト")

# デバッグモードを有効にして関数を実行
import os
os.environ['GPUPGPARSER_DEBUG_ESTIMATE'] = '1'
estimated_size = estimate_row_size_from_columns(columns)
print(f"\nestimate_row_size_from_columns結果: {estimated_size}バイト")

# 効果の計算
data_size = 848_104_872
estimated_rows = data_size // estimated_size
buffer_rows_1_05 = int(estimated_rows * 1.05)
buffer_rows_1_2 = int(estimated_rows * 1.2)
buffer_rows_1_4 = int(estimated_rows * 1.4)
actual_rows = 6_015_118

print(f"\nデータサイズ: {data_size:,}バイト")
print(f"推定行数: {estimated_rows:,}")
print(f"バッファサイズ（旧1.05倍）: {buffer_rows_1_05:,}")
print(f"バッファサイズ（旧1.2倍）: {buffer_rows_1_2:,}")
print(f"バッファサイズ（新1.4倍）: {buffer_rows_1_4:,}")
print(f"実際の行数: {actual_rows:,}")

if buffer_rows_1_4 >= actual_rows:
    print(f"\n✅ 成功！ バッファサイズが十分です")
    print(f"   余裕: {buffer_rows_1_4 - actual_rows:,}行")
else:
    print(f"\n❌ 失敗！ まだバッファが不足しています")
    print(f"   不足: {actual_rows - buffer_rows_1_4:,}行")

conn.close()