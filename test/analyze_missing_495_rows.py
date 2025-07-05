#!/usr/bin/env python3
"""
495行欠落の詳細分析スクリプト

PostgreSQLの実際の行数とParquetファイルの行数の差分を詳しく分析
"""

import os
import sys
import pandas as pd
import pyarrow.parquet as pq
import psycopg2
from collections import defaultdict

def analyze_row_discrepancy():
    """行数の不一致を分析"""
    
    # 1. Parquetファイルから行数を集計
    parquet_dir = "output"
    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
    
    chunk_info = []
    total_parquet_rows = 0
    
    print("=== Parquetファイル分析 ===")
    print(f"{'チャンク':<20} {'行数':>12} {'差分':>12}")
    print("-" * 50)
    
    expected_rows_per_chunk = 7931865  # 標準的なチャンクの行数
    
    for i, pf in enumerate(parquet_files):
        file_path = os.path.join(parquet_dir, pf)
        table = pq.read_table(file_path)
        rows = len(table)
        total_parquet_rows += rows
        
        # チャンク番号を抽出
        chunk_num = int(pf.split('_')[1])
        
        # 標準行数との差分
        diff = rows - expected_rows_per_chunk
        
        chunk_info.append({
            'chunk': chunk_num,
            'file': pf,
            'rows': rows,
            'diff': diff
        })
        
        # 差分が大きいチャンクを強調表示
        if abs(diff) > 100:
            print(f"{pf:<20} {rows:>12,} {diff:>+12,} ***")
        else:
            print(f"{pf:<20} {rows:>12,} {diff:>+12,}")
    
    print("-" * 50)
    print(f"{'合計':<20} {total_parquet_rows:>12,}")
    
    # 2. PostgreSQLから実際の行数を取得
    print("\n=== PostgreSQL行数 ===")
    try:
        # 環境変数から接続情報を取得
        conn = psycopg2.connect(
            host=os.environ.get('PG_HOST', 'localhost'),
            port=os.environ.get('PG_PORT', '5432'),
            database=os.environ.get('PG_DATABASE', 'postgres'),
            user=os.environ.get('PG_USER', 'postgres'),
            password=os.environ.get('PG_PASSWORD', 'postgres')
        )
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM lineorder")
        pg_rows = cur.fetchone()[0]
        
        print(f"PostgreSQL行数: {pg_rows:,}")
        print(f"Parquet行数:    {total_parquet_rows:,}")
        print(f"差分:           {pg_rows - total_parquet_rows:,} 行")
        
        # 3. チャンク毎のページ分布を確認
        print("\n=== チャンク毎のページ分布分析 ===")
        
        # 各チャンクがカバーすべきページ範囲を推定
        cur.execute("SELECT relpages FROM pg_class WHERE relname = 'lineorder'")
        total_pages = cur.fetchone()[0]
        pages_per_chunk = total_pages // 32  # 32チャンク想定
        
        print(f"総ページ数: {total_pages:,}")
        print(f"チャンクあたり平均ページ数: {pages_per_chunk:,}")
        
        # 最後の数チャンクの詳細
        print("\n=== 最後の5チャンクの詳細 ===")
        last_chunks = sorted(chunk_info, key=lambda x: x['chunk'])[-5:]
        
        for chunk in last_chunks:
            print(f"\nチャンク {chunk['chunk']}:")
            print(f"  ファイル: {chunk['file']}")
            print(f"  行数: {chunk['rows']:,}")
            print(f"  標準との差分: {chunk['diff']:+,}")
            print(f"  推定ページ範囲: {chunk['chunk'] * pages_per_chunk:,} - {(chunk['chunk'] + 1) * pages_per_chunk - 1:,}")
        
        # 4. 行サイズの統計
        print("\n=== 行サイズ統計（最初のチャンクから推定） ===")
        first_file = os.path.join(parquet_dir, "chunk_0_queue.parquet")
        first_df = pd.read_parquet(first_file)
        
        # 各列のサイズを推定
        column_sizes = {}
        for col in first_df.columns:
            if first_df[col].dtype == 'object':  # 文字列
                avg_size = first_df[col].astype(str).str.len().mean()
                column_sizes[col] = avg_size + 4  # 長さ情報
            else:  # 数値
                column_sizes[col] = first_df[col].dtype.itemsize
        
        total_row_size = sum(column_sizes.values()) + 24  # PostgreSQLオーバーヘッド
        print(f"推定平均行サイズ: {total_row_size:.1f} バイト")
        
        # 5. 欠落行の推定位置
        print("\n=== 欠落行の推定位置 ===")
        missing_rows = pg_rows - total_parquet_rows
        print(f"欠落行数: {missing_rows}")
        
        # どのチャンクで欠落しているか
        chunks_with_deficit = [c for c in chunk_info if c['diff'] < -100]
        if chunks_with_deficit:
            print("\n行数が少ないチャンク:")
            for chunk in chunks_with_deficit:
                print(f"  チャンク {chunk['chunk']}: {chunk['diff']:,} 行不足")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"PostgreSQL接続エラー: {e}")
        print("環境変数 PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD を設定してください")
    
    # 6. 提案される対策
    print("\n=== 対策提案 ===")
    print("1. max_rows計算のマージンを20%から30%に増加")
    print("2. 最後のチャンクは特別扱いで余裕を持たせる")
    print("3. チャンク境界でのスレッド処理を改善")
    print("4. Grid境界スレッドのデバッグ情報を詳細に記録")
    print("5. チャンク29と31の特別な処理が必要")

if __name__ == "__main__":
    analyze_row_discrepancy()