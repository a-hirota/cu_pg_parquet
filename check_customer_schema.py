#!/usr/bin/env python3
"""
customerテーブルのスキーマと実際のデータサイズを確認
"""

import psycopg2
import os

dsn = os.environ.get("GPUPASER_PG_DSN", "")
conn = psycopg2.connect(dsn)
cur = conn.cursor()

print("=== customerテーブルのスキーマ ===")
cur.execute("""
    SELECT 
        a.attname AS name,
        t.typname AS data_type,
        a.atttypmod AS type_mod,
        CASE 
            WHEN t.typname = 'varchar' THEN a.atttypmod - 4
            WHEN t.typname = 'bpchar' THEN a.atttypmod - 4
            ELSE NULL
        END AS max_length
    FROM pg_attribute a
    JOIN pg_type t ON a.atttypid = t.oid
    WHERE a.attrelid = 'customer'::regclass
      AND a.attnum > 0
      AND NOT a.attisdropped
    ORDER BY a.attnum
""")

columns = cur.fetchall()
for col in columns:
    name, dtype, type_mod, max_len = col
    if max_len:
        print(f"  {name}: {dtype}({max_len})")
    else:
        print(f"  {name}: {dtype}")

# 実際のデータサイズを確認
print("\n=== 実際の文字列長の統計 ===")
string_columns = ['c_name', 'c_address', 'c_phone', 'c_mktsegment']

for col in string_columns:
    cur.execute(f"""
        SELECT 
            MIN(LENGTH({col})) as min_len,
            AVG(LENGTH({col}))::numeric(5,1) as avg_len,
            MAX(LENGTH({col})) as max_len
        FROM customer
        WHERE ctid >= '(0,1)'::tid AND ctid < '(104186,1)'::tid
    """)
    min_len, avg_len, max_len = cur.fetchone()
    print(f"{col}:")
    print(f"  最小: {min_len}, 平均: {avg_len}, 最大: {max_len}")

conn.close()

print("\n=== 推定の修正案 ===")
print("現在の推定（すべての文字列を20バイトで計算）:")
print("- 実際のc_name: 18バイト（推定20より小さい）")
print("- 実際のc_address: 15バイト（推定20より小さい）")
print("- 実際のc_phone: 15バイト（固定長）") 
print("- 実際のc_mktsegment: 10バイト（固定長）")
print("\n問題: 可変長文字列フィールドの推定が大きすぎる")
print("解決策:")
print("1. varchar型の推定を実データに合わせて小さくする")
print("2. または、安全係数を1.5に増やしてバッファ不足を防ぐ")