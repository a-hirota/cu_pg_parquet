#!/usr/bin/env python3
"""
CTID範囲の重複問題を調査するスクリプト
"""

import os
import psycopg
from typing import List, Tuple
import json

def investigate_ctid_ranges(table_name: str = "lineorder", total_chunks: int = 32):
    """各チャンクのCTID範囲と実際の行数を調査"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    conn = psycopg.connect(dsn)
    cursor = conn.cursor()
    
    try:
        # テーブルの総ページ数を取得
        cursor.execute(f"""
            SELECT (pg_relation_size('{table_name}'::regclass) / current_setting('block_size')::int)::int
        """)
        max_page = cursor.fetchone()[0]
        print(f"総ページ数: {max_page:,}")
        
        # 総行数を取得
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        print(f"総行数: {total_rows:,}")
        print()
        
        # チャンクごとの分析
        pages_per_chunk = (max_page + 1) // total_chunks
        print(f"チャンクあたりページ数: {pages_per_chunk:,}")
        print()
        
        chunk_infos = []
        actual_total = 0
        
        for chunk_id in range(total_chunks):
            chunk_start_page = chunk_id * pages_per_chunk
            chunk_end_page = (chunk_id + 1) * pages_per_chunk if chunk_id < total_chunks - 1 else max_page + 1
            
            # 実際の行数を取得
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM {table_name} 
                WHERE ctid >= '({chunk_start_page},1)'::tid 
                AND ctid < '({chunk_end_page},1)'::tid
            """)
            chunk_rows = cursor.fetchone()[0]
            
            # 最初と最後のCTIDを取得
            cursor.execute(f"""
                SELECT MIN(ctid::text), MAX(ctid::text)
                FROM {table_name} 
                WHERE ctid >= '({chunk_start_page},1)'::tid 
                AND ctid < '({chunk_end_page},1)'::tid
            """)
            min_ctid, max_ctid = cursor.fetchone()
            
            # ページ内の最大タプル番号を確認（最後のページのみ）
            if chunk_id == total_chunks - 1:
                cursor.execute(f"""
                    SELECT 
                        (ctid::text::point)[0]::int as page_num,
                        MAX((ctid::text::point)[1]::int) as max_tuple_id
                    FROM {table_name}
                    WHERE ctid >= '({chunk_end_page - 1},1)'::tid
                    GROUP BY 1
                    ORDER BY 1 DESC
                    LIMIT 5
                """)
                last_pages = cursor.fetchall()
            
            chunk_info = {
                'chunk_id': chunk_id,
                'start_page': chunk_start_page,
                'end_page': chunk_end_page,
                'page_count': chunk_end_page - chunk_start_page,
                'row_count': chunk_rows,
                'min_ctid': min_ctid,
                'max_ctid': max_ctid
            }
            chunk_infos.append(chunk_info)
            actual_total += chunk_rows
            
            print(f"チャンク {chunk_id:2d}: ページ {chunk_start_page:,} - {chunk_end_page:,} ({chunk_end_page - chunk_start_page:,} ページ)")
            print(f"  行数: {chunk_rows:,}")
            print(f"  CTID範囲: {min_ctid} - {max_ctid}")
            
            # 最後のチャンクの場合、最後のページの詳細を表示
            if chunk_id == total_chunks - 1 and 'last_pages' in locals():
                print(f"  最後のページの詳細:")
                for page_num, max_tuple_id in last_pages:
                    print(f"    ページ {page_num}: 最大タプルID = {max_tuple_id}")
            print()
        
        print(f"\n合計行数（チャンク合計）: {actual_total:,}")
        print(f"実際の総行数: {total_rows:,}")
        print(f"差分: {actual_total - total_rows:,} ({'+' if actual_total > total_rows else ''}{(actual_total - total_rows) / total_rows * 100:.4f}%)")
        
        # 重複チェック
        print("\n=== 重複チェック ===")
        for i in range(len(chunk_infos) - 1):
            curr = chunk_infos[i]
            next = chunk_infos[i + 1]
            if curr['end_page'] != next['start_page']:
                print(f"⚠️ チャンク {i} と {i+1} の境界に問題あり: {curr['end_page']} != {next['start_page']}")
        
        # 結果をJSONで保存
        with open('/tmp/ctid_investigation.json', 'w') as f:
            json.dump({
                'total_pages': max_page,
                'total_rows': total_rows,
                'pages_per_chunk': pages_per_chunk,
                'chunks': chunk_infos,
                'actual_total': actual_total,
                'difference': actual_total - total_rows
            }, f, indent=2)
        print("\n結果を /tmp/ctid_investigation.json に保存しました")
        
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import sys
    table_name = os.environ.get("TABLE_NAME", "lineorder")
    total_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    investigate_ctid_ranges(table_name, total_chunks)