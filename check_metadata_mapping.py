#!/usr/bin/env python3
"""
PostgreSQLとArrowのメタデータマッピングを検証
"""

import os
import sys
sys.path.append('/home/ubuntu/gpupgparser')

from src.readPostgres.metadata import fetch_column_meta
from src.types import PG_OID_TO_ARROW, UTF8, DECIMAL128
import psycopg

# PostgreSQL接続
dsn = os.environ.get("GPUPASER_PG_DSN", "")
conn = psycopg.connect(dsn)

# customerテーブルのメタデータ取得
columns = fetch_column_meta(conn, "SELECT * FROM customer")

print("=== customerテーブルのメタデータマッピング ===\n")
print(f"{'列名':<15} {'PG型':<10} {'PG OID':<8} {'Arrow ID':<10} {'elem_size':<10} {'arrow_param':<15}")
print("-" * 80)

for col in columns:
    # PG OIDから型名を取得
    cur = conn.cursor()
    cur.execute("SELECT typname FROM pg_type WHERE oid = %s", (col.pg_oid,))
    pg_type = cur.fetchone()[0] if cur.rowcount > 0 else "unknown"
    cur.close()
    
    arrow_name = {
        0: "INT16", 1: "INT32", 2: "INT64", 
        3: "FLOAT32", 4: "FLOAT64", 5: "DECIMAL128",
        6: "UTF8", 7: "BINARY", 8: "DATE32", 
        9: "TS64_US", 10: "BOOL", 255: "UNKNOWN"
    }.get(col.arrow_id, f"ID:{col.arrow_id}")
    
    print(f"{col.name:<15} {pg_type:<10} {col.pg_oid:<8} {arrow_name:<10} {col.elem_size:<10} {str(col.arrow_param):<15}")

print("\n=== 問題の分析 ===")
print("\n1. bpchar（固定長文字列）の扱い:")
for col in columns:
    if col.arrow_id == UTF8 and col.arrow_param is not None:
        cur = conn.cursor()
        cur.execute("SELECT typname FROM pg_type WHERE oid = %s", (col.pg_oid,))
        pg_type = cur.fetchone()[0]
        cur.close()
        print(f"   {col.name}: {pg_type}({col.arrow_param}) → Arrow UTF8 (elem_size={col.elem_size})")
        print(f"      arrow_paramに長さ{col.arrow_param}が格納されているが、elem_sizeは0（可変長扱い）")

print("\n2. estimate_row_size_from_columns関数での計算:")
total_size = 2  # フィールド数
for col in columns:
    total_size += 4  # フィールド長
    if col.elem_size > 0:
        total_size += col.elem_size
        print(f"   {col.name}: 固定長 {col.elem_size}バイト")
    else:
        if col.arrow_id == UTF8:
            total_size += 20
            actual_size = col.arrow_param if col.arrow_param else "?"
            print(f"   {col.name}: UTF8として20バイト推定（実際は{actual_size}バイト）")
        elif col.arrow_id == DECIMAL128:
            total_size += 16
            print(f"   {col.name}: DECIMAL128として16バイト")

aligned_size = ((total_size + 31) // 32) * 32
print(f"\n合計: {total_size}バイト → 32バイト整列: {aligned_size}バイト")

# 実際のbpchar長を使った計算
print("\n3. bpcharの実際の長さを使った計算:")
correct_size = 2  # フィールド数
for col in columns:
    correct_size += 4  # フィールド長
    if col.elem_size > 0:
        correct_size += col.elem_size
    else:
        if col.arrow_id == UTF8:
            if col.arrow_param:  # bpchar
                correct_size += col.arrow_param
                print(f"   {col.name}: bpchar({col.arrow_param}) → {col.arrow_param}バイト")
            else:  # varchar
                correct_size += 15  # より現実的な推定
                print(f"   {col.name}: varchar → 15バイト推定")
        elif col.arrow_id == DECIMAL128:
            correct_size += 16

correct_aligned = ((correct_size + 31) // 32) * 32
print(f"\n修正後合計: {correct_size}バイト → 32バイト整列: {correct_aligned}バイト")

data_size = 848_104_872
print(f"\nデータサイズ: {data_size:,}バイト")
print(f"現在の推定行数: {data_size // aligned_size:,}")
print(f"修正後の推定行数: {data_size // correct_aligned:,}")
print(f"実際の行数: 6,015,118")

conn.close()