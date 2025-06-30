#!/usr/bin/env python3
"""
CTIDの問題を素早く確認
"""

import os
import psycopg

def quick_check():
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    cursor = conn.cursor()
    
    try:
        # 総ページ数
        cursor.execute("""
            SELECT (pg_relation_size('lineorder'::regclass) / current_setting('block_size')::int)::int
        """)
        max_page = cursor.fetchone()[0]
        print(f"総ページ数: {max_page:,}")
        
        # 32チャンクで計算
        pages_per_chunk = (max_page + 1) // 32
        print(f"チャンクあたりページ数: {pages_per_chunk:,}")
        
        # いくつかのチャンクをサンプリング
        sample_chunks = [0, 19, 20, 30, 31]
        
        for chunk_id in sample_chunks:
            chunk_start_page = chunk_id * pages_per_chunk
            chunk_end_page = (chunk_id + 1) * pages_per_chunk if chunk_id < 31 else max_page + 1
            
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM lineorder
                WHERE ctid >= '({chunk_start_page},1)'::tid 
                AND ctid < '({chunk_end_page},1)'::tid
            """)
            count = cursor.fetchone()[0]
            
            print(f"\nチャンク {chunk_id}: ページ {chunk_start_page:,} - {chunk_end_page:,}")
            print(f"  ページ数: {chunk_end_page - chunk_start_page:,}")
            print(f"  行数: {count:,}")
            print(f"  平均行/ページ: {count / (chunk_end_page - chunk_start_page):.1f}")
        
        # 最後のページの状況を確認
        print("\n=== 最後のページ付近の詳細 ===")
        cursor.execute(f"""
            SELECT 
                (ctid::text::point)[0]::int as page_num,
                COUNT(*) as row_count
            FROM lineorder
            WHERE ctid >= '({max_page - 10},1)'::tid
            GROUP BY 1
            ORDER BY 1 DESC
            LIMIT 10
        """)
        
        for page_num, row_count in cursor.fetchall():
            print(f"ページ {page_num}: {row_count} 行")
            
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    quick_check()