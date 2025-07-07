#!/usr/bin/env python3
"""
欠落行のパターンを素早く分析
Parquetファイルのキー範囲とPostgreSQLのキー範囲を比較
"""

import pandas as pd
import psycopg2
import os
from pathlib import Path

def analyze_key_ranges():
    """Parquetファイルとデータベースのキー範囲を分析"""
    
    # Parquetファイルのキー範囲を確認
    print("=== Parquetファイルのキー範囲分析 ===")
    output_dir = Path("/home/ubuntu/gpupgparser/output")
    parquet_files = sorted(output_dir.glob("chunk_*_queue.parquet"))
    
    all_ranges = []
    
    for file in parquet_files[:3]:  # 最初の3ファイルだけ確認
        print(f"\n{file.name}:")
        df = pd.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
        
        min_orderkey = df['lo_orderkey'].min()
        max_orderkey = df['lo_orderkey'].max()
        row_count = len(df)
        unique_keys = df.drop_duplicates(['lo_orderkey', 'lo_linenumber']).shape[0]
        
        print(f"  行数: {row_count:,}")
        print(f"  ユニークキー数: {unique_keys:,}")
        print(f"  lo_orderkey範囲: {min_orderkey:,} - {max_orderkey:,}")
        
        # サンプル表示
        print(f"  最初の5行:")
        print(df.head().to_string(index=False))
        
        all_ranges.append({
            'file': file.name,
            'min_orderkey': min_orderkey,
            'max_orderkey': max_orderkey,
            'rows': row_count
        })
    
    # PostgreSQLのキー範囲を確認
    print("\n\n=== PostgreSQLのキー範囲分析 ===")
    conn = psycopg2.connect(
        dbname="dbmgpu",
        user="postgres",
        password="postgres"
    )
    
    with conn.cursor() as cur:
        # 総行数
        cur.execute("SELECT COUNT(*) FROM lineorder")
        total_rows = cur.fetchone()[0]
        print(f"総行数: {total_rows:,}")
        
        # キー範囲
        cur.execute("SELECT MIN(lo_orderkey), MAX(lo_orderkey) FROM lineorder")
        min_key, max_key = cur.fetchone()
        print(f"lo_orderkey範囲: {min_key:,} - {max_key:,}")
        
        # 各Parquetファイルのキー範囲内での行数を確認
        print("\n各Parquetキー範囲内のPostgreSQL行数:")
        for r in all_ranges:
            cur.execute("""
                SELECT COUNT(*) 
                FROM lineorder 
                WHERE lo_orderkey >= %s AND lo_orderkey <= %s
            """, (r['min_orderkey'], r['max_orderkey']))
            pg_rows = cur.fetchone()[0]
            coverage = r['rows'] / pg_rows * 100 if pg_rows > 0 else 0
            print(f"  {r['file']}: {pg_rows:,}行 (Parquet: {r['rows']:,}行, カバー率: {coverage:.1f}%)")
        
        # ページごとの行数分布を確認
        print("\n\n=== ページごとの行数分布 ===")
        cur.execute("""
            WITH page_stats AS (
                SELECT 
                    (ctid::text::point)[0]::int as page_number,
                    COUNT(*) as row_count
                FROM lineorder
                GROUP BY page_number
                ORDER BY page_number
            )
            SELECT 
                MIN(row_count) as min_rows,
                AVG(row_count) as avg_rows,
                MAX(row_count) as max_rows,
                STDDEV(row_count) as stddev_rows,
                COUNT(*) as total_pages
            FROM page_stats
        """)
        
        result = cur.fetchone()
        print(f"最小行数/ページ: {result[0]}")
        print(f"平均行数/ページ: {result[1]:.1f}")
        print(f"最大行数/ページ: {result[2]}")
        print(f"標準偏差: {result[3]:.1f}")
        print(f"総ページ数: {result[4]:,}")
        
        # 行数が少ないページの例
        print("\n行数が少ないページの例（最初の10ページ）:")
        cur.execute("""
            WITH page_stats AS (
                SELECT 
                    (ctid::text::point)[0]::int as page_number,
                    COUNT(*) as row_count
                FROM lineorder
                GROUP BY page_number
                HAVING COUNT(*) < 40
                ORDER BY page_number
                LIMIT 10
            )
            SELECT * FROM page_stats
        """)
        
        for row in cur.fetchall():
            print(f"  ページ {row[0]}: {row[1]}行")
    
    conn.close()

def main():
    analyze_key_ranges()

if __name__ == "__main__":
    main()