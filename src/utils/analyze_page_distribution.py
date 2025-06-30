#!/usr/bin/env python3
"""
ページごとの行数分布を分析
"""

import os
import psycopg

def analyze_distribution():
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    cursor = conn.cursor()
    
    try:
        # 総行数と総ページ数
        cursor.execute("SELECT COUNT(*) FROM lineorder")
        total_rows = cursor.fetchone()[0]
        print(f"総行数: {total_rows:,}")
        
        cursor.execute("""
            SELECT (pg_relation_size('lineorder'::regclass) / current_setting('block_size')::int)::int
        """)
        max_page = cursor.fetchone()[0]
        print(f"総ページ数: {max_page:,}")
        print(f"平均行/ページ: {total_rows / (max_page + 1):.2f}")
        
        # ページごとの行数分布をサンプリング
        print("\n=== ページ分布のサンプリング ===")
        
        # 前半、中盤、後半からサンプル
        sample_points = [
            (0, 1000, "最初の1000ページ"),
            (2000000, 2001000, "200万ページ付近"),
            (3000000, 3001000, "300万ページ付近"),
            (4000000, 4001000, "400万ページ付近"),
            (max_page - 1000, max_page + 1, "最後の1000ページ")
        ]
        
        for start, end, desc in sample_points:
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_rows,
                    MIN((ctid::text::point)[1]::int) as min_tuple,
                    MAX((ctid::text::point)[1]::int) as max_tuple,
                    AVG((ctid::text::point)[1]::int) as avg_tuple
                FROM lineorder
                WHERE ctid >= '({start},1)'::tid 
                AND ctid < '({end},1)'::tid
            """)
            result = cursor.fetchone()
            if result[0] > 0:
                print(f"\n{desc}:")
                print(f"  行数: {result[0]:,}")
                print(f"  平均行/ページ: {result[0] / (end - start):.2f}")
                print(f"  タプルID範囲: {result[1]} - {result[2]} (平均: {result[3]:.1f})")
        
        # チャンク20-31の詳細分析
        print("\n=== チャンク20-31の詳細 ===")
        pages_per_chunk = (max_page + 1) // 32
        
        for chunk_id in range(20, 32):
            chunk_start_page = chunk_id * pages_per_chunk
            chunk_end_page = (chunk_id + 1) * pages_per_chunk if chunk_id < 31 else max_page + 1
            
            # ページ分布の統計
            cursor.execute(f"""
                SELECT 
                    COUNT(DISTINCT (ctid::text::point)[0]::int) as distinct_pages,
                    COUNT(*) as total_rows
                FROM lineorder
                WHERE ctid >= '({chunk_start_page},1)'::tid 
                AND ctid < '({chunk_end_page},1)'::tid
            """)
            distinct_pages, total_rows = cursor.fetchone()
            
            print(f"\nチャンク {chunk_id}:")
            print(f"  ページ範囲: {chunk_start_page:,} - {chunk_end_page:,}")
            print(f"  ページ数（計算）: {chunk_end_page - chunk_start_page:,}")
            print(f"  実際のページ数: {distinct_pages:,}")
            print(f"  行数: {total_rows:,}")
            if distinct_pages > 0:
                print(f"  平均行/ページ: {total_rows / distinct_pages:.2f}")
                
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    analyze_distribution()