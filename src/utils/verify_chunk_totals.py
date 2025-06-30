#!/usr/bin/env python3
"""
32チャンクの合計を検証
"""

import os
import psycopg

def verify_totals():
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    cursor = conn.cursor()
    
    try:
        # 総行数
        cursor.execute("SELECT COUNT(*) FROM lineorder")
        actual_total = cursor.fetchone()[0]
        print(f"PostgreSQL実際の総行数: {actual_total:,}")
        
        # ページ数
        cursor.execute("""
            SELECT (pg_relation_size('lineorder'::regclass) / current_setting('block_size')::int)::int
        """)
        max_page = cursor.fetchone()[0]
        pages_per_chunk = (max_page + 1) // 32
        
        print(f"\n総ページ数: {max_page:,} (0から{max_page})")
        print(f"チャンクあたりページ数: {pages_per_chunk:,}")
        
        # 各チャンクの行数を計算
        chunk_total = 0
        chunk_rows = []
        
        print("\n=== 各チャンクの行数 ===")
        for chunk_id in range(32):
            chunk_start_page = chunk_id * pages_per_chunk
            chunk_end_page = (chunk_id + 1) * pages_per_chunk if chunk_id < 31 else max_page + 1
            
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM lineorder
                WHERE ctid >= '({chunk_start_page},1)'::tid 
                AND ctid < '({chunk_end_page},1)'::tid
            """)
            rows = cursor.fetchone()[0]
            chunk_rows.append(rows)
            chunk_total += rows
            
            if chunk_id < 5 or chunk_id >= 27:  # 最初と最後のチャンクを表示
                print(f"チャンク {chunk_id:2d}: {rows:,} 行 (ページ {chunk_start_page:,} - {chunk_end_page - 1:,})")
        
        print("...")
        print(f"\nチャンク合計: {chunk_total:,}")
        print(f"実際の総行数: {actual_total:,}")
        print(f"差分: {chunk_total - actual_total:,} 行 ({(chunk_total - actual_total) / actual_total * 100:.4f}%)")
        
        # 問題の原因を調査
        print("\n=== 問題の調査 ===")
        
        # 最後のページの実際の番号を確認
        cursor.execute("""
            SELECT MAX((ctid::text::point)[0]::int) FROM lineorder
        """)
        actual_max_page = cursor.fetchone()[0]
        print(f"実際の最大ページ番号: {actual_max_page:,}")
        
        # チャンク31の実際の範囲を確認
        chunk_start_page = 31 * pages_per_chunk
        chunk_end_page = max_page + 1
        
        print(f"\nチャンク31の計算:")
        print(f"  開始ページ: {chunk_start_page:,}")
        print(f"  終了ページ（計算）: {chunk_end_page:,}")
        print(f"  実際の最大ページ: {actual_max_page:,}")
        
        # 境界外のデータがあるか確認
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM lineorder
            WHERE (ctid::text::point)[0]::int > {max_page}
        """)
        beyond_max = cursor.fetchone()[0]
        print(f"\nページ {max_page} を超えるデータ: {beyond_max:,} 行")
        
        # VACUUMやDELETEによる穴を確認
        print("\n=== ページ利用率の確認 ===")
        for chunk_id in [0, 19, 20, 31]:
            chunk_start_page = chunk_id * pages_per_chunk
            chunk_end_page = (chunk_id + 1) * pages_per_chunk if chunk_id < 31 else max_page + 1
            
            cursor.execute(f"""
                SELECT 
                    COUNT(DISTINCT (ctid::text::point)[0]::int) as used_pages,
                    {chunk_end_page - chunk_start_page} as total_pages
                FROM lineorder
                WHERE ctid >= '({chunk_start_page},1)'::tid 
                AND ctid < '({chunk_end_page},1)'::tid
            """)
            used_pages, total_pages = cursor.fetchone()
            utilization = used_pages / total_pages * 100 if total_pages > 0 else 0
            print(f"チャンク {chunk_id}: {used_pages:,}/{total_pages:,} ページ使用中 ({utilization:.1f}%)")
            
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    verify_totals()