#!/usr/bin/env python3
"""
lineorderテーブルのチャンク分割調査スクリプト
"""

import os
import psycopg
import pyarrow.parquet as pq
from pathlib import Path

def main():
    # PostgreSQLに接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("❌ 環境変数 GPUPASER_PG_DSN が設定されていません")
        return
    
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # 1. 実際の行数を確認
            cur.execute("SELECT COUNT(*) FROM lineorder")
            actual_rows = cur.fetchone()[0]
            print(f"PostgreSQLの実際の行数: {actual_rows:,}")
            
            # 2. テーブルサイズとページ数を確認
            cur.execute("""
                SELECT 
                    pg_size_pretty(pg_relation_size('lineorder'::regclass)) as table_size,
                    (pg_relation_size('lineorder'::regclass) / current_setting('block_size')::int)::int as total_pages,
                    pg_relation_size('lineorder'::regclass) as size_bytes
            """)
            result = cur.fetchone()
            print(f"テーブルサイズ: {result[0]}")
            print(f"総ページ数: {result[1]:,}")
            print(f"バイト数: {result[2]:,} ({result[2] / 1024**3:.2f} GB)")
            
            total_pages = result[1]
            
            # 3. チャンクごとのページ範囲を計算（16チャンクの場合）
            total_chunks = 16
            pages_per_chunk = (total_pages + 1) / total_chunks
            
            print(f"\n【16チャンクでの分割情報】")
            print(f"チャンクあたりのページ数: {pages_per_chunk:.0f}")
            
            # 4. 各チャンクの理論的な行数を確認
            print("\n【各チャンクの理論的な行数】")
            for chunk_id in range(total_chunks):
                chunk_start = int(chunk_id * pages_per_chunk)
                chunk_end = int((chunk_id + 1) * pages_per_chunk) if chunk_id < total_chunks - 1 else total_pages + 1
                
                # 実際の行数をカウント（最初と最後のチャンクのみ）
                if chunk_id in [0, 15]:
                    cur.execute(f"""
                        SELECT COUNT(*) 
                        FROM lineorder 
                        WHERE ctid >= '({chunk_start},1)'::tid 
                          AND ctid < '({chunk_end},1)'::tid
                    """)
                    chunk_rows = cur.fetchone()[0]
                    print(f"チャンク {chunk_id:2d}: ページ {chunk_start:,} - {chunk_end:,} → {chunk_rows:,} 行")
                else:
                    print(f"チャンク {chunk_id:2d}: ページ {chunk_start:,} - {chunk_end:,}")
    
    # 5. Parquetファイルの実際の行数を確認
    print("\n【Parquetファイルの実際の行数】")
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    total_parquet_rows = 0
    
    for pf in parquet_files:
        try:
            table = pq.read_table(pf)
            total_parquet_rows += table.num_rows
            chunk_id = int(pf.stem.split('_')[1])
            print(f"チャンク {chunk_id:2d}: {table.num_rows:,} 行")
        except Exception as e:
            print(f"チャンク {pf.stem}: エラー - {e}")
    
    print(f"\nParquetファイルの総行数: {total_parquet_rows:,}")
    print(f"PostgreSQLの実際の行数: {actual_rows:,}")
    print(f"差分: {actual_rows - total_parquet_rows:,} 行 ({(actual_rows - total_parquet_rows) / actual_rows * 100:.2f}%)")

if __name__ == "__main__":
    main()