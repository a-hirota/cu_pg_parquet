#!/usr/bin/env python3
"""lineorderテーブルのサイズ確認"""
import os
import psycopg2

dsn = os.environ.get('GPUPASER_PG_DSN', "dbname=postgres user=postgres host=localhost port=5432")
conn = psycopg2.connect(dsn)
cur = conn.cursor()

# テーブルサイズ
cur.execute("""
    SELECT 
        pg_size_pretty(pg_relation_size('lineorder')) as table_size,
        pg_relation_size('lineorder') as table_size_bytes,
        pg_size_pretty(pg_total_relation_size('lineorder')) as total_size,
        COUNT(*) as row_count
    FROM lineorder
    GROUP BY pg_relation_size('lineorder'), pg_total_relation_size('lineorder')
""")

result = cur.fetchone()
print(f"テーブルサイズ: {result[0]} ({result[1] / 1024**3:.2f} GB)")
print(f"総サイズ: {result[2]}")
print(f"行数: {result[3]:,}")
print(f"平均行サイズ: {result[1] / result[3]:.1f} bytes/行")

cur.close()
conn.close()