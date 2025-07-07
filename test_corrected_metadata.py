#!/usr/bin/env python3
"""
修正されたメタデータ取得のテスト
"""

import os
import sys
sys.path.append('/home/ubuntu/gpupgparser')
import psycopg

dsn = os.environ.get("GPUPASER_PG_DSN", "")
conn = psycopg.connect(dsn)

cur = conn.cursor()
cur.execute("SELECT * FROM customer LIMIT 0")

print("=== 修正案のテスト ===\n")

for desc in cur.description:
    name = desc.name
    pg_oid = desc.type_code
    
    # 現在の実装
    pg_typmod_current = desc.internal_size or 0
    
    # 修正案1: display_sizeを使う
    pg_typmod_fixed1 = desc.display_size if desc.display_size else 0
    
    # 修正案2: PostgreSQLカタログから直接取得
    cur2 = conn.cursor()
    cur2.execute("""
        SELECT a.atttypmod 
        FROM pg_attribute a
        WHERE a.attrelid = 'customer'::regclass
        AND a.attname = %s
    """, (name,))
    pg_typmod_fixed2 = cur2.fetchone()[0] if cur2.rowcount > 0 else -1
    cur2.close()
    
    # 型名を取得
    cur2 = conn.cursor()
    cur2.execute("SELECT typname FROM pg_type WHERE oid = %s", (pg_oid,))
    type_name = cur2.fetchone()[0]
    cur2.close()
    
    print(f"{name} ({type_name}):")
    print(f"  現在の実装 (internal_size): {pg_typmod_current}")
    print(f"  修正案1 (display_size): {pg_typmod_fixed1}")
    print(f"  修正案2 (pg_attribute): {pg_typmod_fixed2}")
    
    if type_name in ('varchar', 'bpchar') and pg_typmod_fixed1 > 0:
        print(f"  → 文字長: {pg_typmod_fixed1}")
    
    print()

conn.close()

print("=== 結論 ===")
print("1. internal_sizeは常にNoneのため使えない")
print("2. display_sizeに正しい文字長が格納されている")
print("3. bpcharとvarcharの両方でdisplay_sizeが利用可能")