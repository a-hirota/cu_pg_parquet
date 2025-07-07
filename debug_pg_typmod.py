#!/usr/bin/env python3
"""
pg_typmodの値を詳しく調査
"""

import os
import psycopg

dsn = os.environ.get("GPUPASER_PG_DSN", "")
conn = psycopg.connect(dsn)

print("=== pg_typmodの調査 ===\n")

# 直接PostgreSQLのシステムカタログを確認
cur = conn.cursor()
cur.execute("""
    SELECT 
        a.attname AS column_name,
        t.typname AS type_name,
        a.atttypmod AS typmod,
        a.attlen AS attlen,
        CASE 
            WHEN t.typname IN ('varchar', 'bpchar') AND a.atttypmod > 0 
            THEN a.atttypmod - 4 
            ELSE NULL 
        END AS char_length
    FROM pg_attribute a
    JOIN pg_type t ON a.atttypid = t.oid
    WHERE a.attrelid = 'customer'::regclass
      AND a.attnum > 0
      AND NOT a.attisdropped
    ORDER BY a.attnum
""")

print(f"{'列名':<15} {'型名':<10} {'typmod':<10} {'attlen':<10} {'文字長':<10}")
print("-" * 60)

for row in cur.fetchall():
    column_name, type_name, typmod, attlen, char_length = row
    print(f"{column_name:<15} {type_name:<10} {typmod:<10} {attlen:<10} {char_length if char_length else '-':<10}")

# fetch_column_metaで何が取得されているか確認
print("\n=== cursor.descriptionの内容 ===")
cur.execute("SELECT * FROM customer LIMIT 0")
for i, desc in enumerate(cur.description):
    print(f"\n列{i}: {desc.name}")
    print(f"  type_code: {desc.type_code}")
    print(f"  display_size: {desc.display_size}")
    print(f"  internal_size: {desc.internal_size}")
    print(f"  precision: {desc.precision}")
    print(f"  scale: {desc.scale}")

conn.close()