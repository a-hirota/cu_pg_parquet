#!/usr/bin/env python3
"""
lineorderテーブルのページ分布を調査
"""
import os
import psycopg2
from psycopg2 import sql

def main():
    # PostgreSQL接続
    dsn = os.environ.get('GPUPASER_PG_DSN', "dbname=postgres user=postgres host=localhost port=5432")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # テーブル名
    table_name = 'lineorder'
    
    # 1. 基本情報取得
    print("=== lineorderテーブル基本情報 ===")
    
    # 総行数
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cur.fetchone()[0]
    print(f"総行数: {total_rows:,}")
    
    # relpagesの値
    cur.execute(f"SELECT relpages FROM pg_class WHERE relname = '{table_name}'")
    relpages = cur.fetchone()[0]
    print(f"pg_class.relpages: {relpages:,}")
    
    # 実際のページ数
    cur.execute(f"SELECT (pg_relation_size('{table_name}'::regclass) / current_setting('block_size')::int)::int")
    actual_pages = cur.fetchone()[0]
    print(f"実際のページ数: {actual_pages:,}")
    print(f"差分: {actual_pages - relpages:,} ページ ({(actual_pages/relpages-1)*100:.1f}%)")
    
    # 2. ページ分布調査
    print("\n=== ページ分布調査 ===")
    
    # 最初と最後のctid
    cur.execute(f"SELECT MIN(ctid), MAX(ctid) FROM {table_name}")
    min_ctid, max_ctid = cur.fetchone()
    print(f"最小ctid: {min_ctid}")
    print(f"最大ctid: {max_ctid}")
    
    # ページごとの行数分布（サンプリング）
    print("\n=== ページごとの行数分布（8チャンク想定） ===")
    
    chunk_size = actual_pages // 8
    for i in range(8):
        start_page = i * chunk_size
        end_page = actual_pages if i == 7 else (i + 1) * chunk_size
        
        # 各チャンクの行数をカウント
        cur.execute(f"""
            SELECT COUNT(*) 
            FROM {table_name} 
            WHERE ctid >= '({start_page},1)'::tid 
              AND ctid < '({end_page},1)'::tid
        """)
        chunk_rows = cur.fetchone()[0]
        
        print(f"チャンク{i}: ページ {start_page:,}-{end_page:,} ({end_page-start_page:,}ページ) = {chunk_rows:,}行")
    
    # 3. ページの実際の最大値確認
    print("\n=== 実際に使用されているページ範囲 ===")
    
    # 最大ページ番号を取得
    cur.execute(f"""
        SELECT (ctid::text::point)[0]::int as page_num
        FROM {table_name}
        ORDER BY ctid DESC
        LIMIT 1
    """)
    max_used_page = cur.fetchone()[0]
    print(f"実際に使用されている最大ページ番号: {max_used_page:,}")
    print(f"未使用ページ: {actual_pages - max_used_page - 1:,} ページ")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()