#!/usr/bin/env python3
"""
pg_relation_sizeの値を確認
"""
import psycopg2
import os

def check_pg_relation_size():
    """pg_relation_sizeの値を確認"""
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("エラー: GPUPASER_PG_DSN環境変数が設定されていません")
        return
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    print("=== pg_relation_sizeの確認 ===\n")
    
    # 1. pg_relation_sizeの値
    cur.execute("""
        SELECT pg_relation_size('lineorder'::regclass) as size_bytes,
               pg_relation_size('lineorder'::regclass) / current_setting('block_size')::int as total_pages
    """)
    size_bytes, total_pages = cur.fetchone()
    print(f"pg_relation_size: {size_bytes:,} bytes")
    print(f"総ページ数: {total_pages}")
    
    # 2. 実際の最大ctid
    cur.execute("SELECT MAX(ctid) FROM lineorder")
    max_ctid = cur.fetchone()[0]
    page_num, tuple_num = map(int, max_ctid.strip('()').split(','))
    print(f"\n最大ctid: {max_ctid}")
    print(f"最大ページ番号: {page_num}")
    
    # 3. 差分
    print(f"\n差分: pg_relation_size - 実際の最大ページ = {total_pages - page_num - 1}")
    
    # 4. 最後の数ページを確認
    print("\n最後の10ページの内容:")
    for i in range(10):
        check_page = page_num - i
        cur.execute(f"""
            SELECT COUNT(*) 
            FROM lineorder 
            WHERE ctid >= '({check_page},0)'::tid 
              AND ctid < '({check_page + 1},0)'::tid
        """)
        rows = cur.fetchone()[0]
        print(f"  ページ{check_page}: {rows}行")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    check_pg_relation_size()