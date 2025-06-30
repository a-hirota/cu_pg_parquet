#!/usr/bin/env python3
"""
16チャンクでのページカバレッジを検証するスクリプト
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

def main():
    # データベース接続
    dsn = os.environ['GPUPASER_PG_DSN']
    table_name = os.environ.get('TABLE_NAME', 'lineorder')
    total_chunks = 16
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # テーブルの総ページ数を取得
    cur.execute(f"""
        SELECT (pg_relation_size('{table_name}'::regclass) / current_setting('block_size')::int)::int as max_page
    """)
    result = cur.fetchone()
    max_page = result['max_page']
    print(f"テーブル: {table_name}")
    print(f"総ページ数: {max_page + 1} (0-{max_page})")
    print()
    
    # 旧方式（整数除算）での計算
    print("=== 旧方式（整数除算） ===")
    old_pages_per_chunk = (max_page + 1) // total_chunks
    old_total_pages = 0
    
    for chunk_id in range(total_chunks):
        chunk_start = chunk_id * old_pages_per_chunk
        if chunk_id == total_chunks - 1:
            chunk_end = max_page + 1
        else:
            chunk_end = (chunk_id + 1) * old_pages_per_chunk
        
        pages = chunk_end - chunk_start
        old_total_pages += pages
        print(f"チャンク{chunk_id:2d}: ページ {chunk_start:6d} - {chunk_end:6d} ({pages:6d}ページ)")
    
    print(f"\n旧方式カバレッジ: {old_total_pages} / {max_page + 1} = {old_total_pages / (max_page + 1) * 100:.2f}%")
    print(f"欠損ページ数: {max_page + 1 - old_total_pages}")
    
    # 新方式（切り上げ除算）での計算
    print("\n=== 新方式（切り上げ除算） ===")
    new_pages_per_chunk = (max_page + total_chunks) // total_chunks
    new_total_pages = 0
    
    for chunk_id in range(total_chunks):
        chunk_start = chunk_id * new_pages_per_chunk
        if chunk_id == total_chunks - 1:
            chunk_end = max_page + 1
        else:
            chunk_end = (chunk_id + 1) * new_pages_per_chunk
            # 最大ページを超えないようにクリップ
            if chunk_end > max_page + 1:
                chunk_end = max_page + 1
        
        pages = chunk_end - chunk_start
        new_total_pages += pages
        print(f"チャンク{chunk_id:2d}: ページ {chunk_start:6d} - {chunk_end:6d} ({pages:6d}ページ)")
    
    print(f"\n新方式カバレッジ: {new_total_pages} / {max_page + 1} = {new_total_pages / (max_page + 1) * 100:.2f}%")
    
    # ctidの行番号による影響を推定
    print("\n=== ctid行番号の影響 ===")
    # サンプルページの行数を確認
    cur.execute(f"""
        SELECT COUNT(*) as row_count 
        FROM {table_name} 
        WHERE ctid >= '(0,0)'::tid AND ctid < '(1,0)'::tid
    """)
    sample_rows = cur.fetchone()['row_count']
    print(f"1ページあたりの推定行数: {sample_rows}")
    
    # ctid (page,1) vs (page,0) の差を確認
    cur.execute(f"""
        SELECT 
            COUNT(*) FILTER (WHERE ctid >= '(0,0)'::tid AND ctid < '(0,1)'::tid) as row_0,
            COUNT(*) FILTER (WHERE ctid >= '(0,1)'::tid AND ctid < '(1,0)'::tid) as row_1_plus
        FROM {table_name}
        WHERE ctid >= '(0,0)'::tid AND ctid < '(1,0)'::tid
    """)
    ctid_result = cur.fetchone()
    print(f"ページ0の行0: {ctid_result['row_0']}行")
    print(f"ページ0の行1以降: {ctid_result['row_1_plus']}行")
    
    if ctid_result['row_0'] > 0:
        loss_per_page = ctid_result['row_0'] / sample_rows * 100
        print(f"\nctid (page,1) 開始による行損失: {loss_per_page:.2f}%/ページ")
        print(f"推定総行損失: {loss_per_page * (max_page + 1) / 100:.0f}行")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()