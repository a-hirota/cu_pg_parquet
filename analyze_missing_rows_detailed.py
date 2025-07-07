#!/usr/bin/env python3
"""
欠落行の詳細分析スクリプト
cuDFで処理したParquetデータとPostgreSQLデータを比較し、
欠落行のパターンを特定する
"""

import pandas as pd
import cudf
import psycopg2
import os
from pathlib import Path
import time

def load_parquet_keys():
    """Parquetファイルからキー情報を読み込み"""
    print("Parquetファイルからキー情報を読み込み中...")
    
    output_dir = Path("/home/ubuntu/gpupgparser/output")
    parquet_files = sorted(output_dir.glob("chunk_*_queue.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"Parquetファイルが見つかりません: {output_dir}")
    
    print(f"見つかったParquetファイル数: {len(parquet_files)}")
    
    all_keys = []
    
    for file in parquet_files:
        print(f"処理中: {file.name}")
        # cuDFで読み込み
        gdf = cudf.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
        
        # CPUメモリに転送（メモリ効率のため）
        pdf = gdf.to_pandas()
        all_keys.append(pdf)
        
        del gdf
    
    # 全データを結合
    print("データ結合中...")
    df_parquet = pd.concat(all_keys, ignore_index=True)
    
    # ユニークキーの集合を作成
    parquet_keys = set(zip(df_parquet['lo_orderkey'], df_parquet['lo_linenumber']))
    
    print(f"Parquetレコード数: {len(df_parquet):,}")
    print(f"ユニークキー数: {len(parquet_keys):,}")
    print(f"重複レコード数: {len(df_parquet) - len(parquet_keys):,}")
    
    return parquet_keys, df_parquet

def load_postgres_keys(limit=None):
    """PostgreSQLからキー情報を読み込み"""
    print("\nPostgreSQLからキー情報を読み込み中...")
    
    conn = psycopg2.connect(
        dbname="dbmgpu",
        user="postgres",
        password="postgres"
    )
    
    # 行数確認
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM lineorder")
        total_rows = cur.fetchone()[0]
        print(f"PostgreSQL総行数: {total_rows:,}")
    
    # キー情報を取得（メモリ効率のためチャンク処理）
    query = "SELECT lo_orderkey, lo_linenumber FROM lineorder"
    if limit:
        query += f" LIMIT {limit}"
    
    postgres_keys = set()
    
    with conn.cursor(name='server_cursor') as cur:
        cur.itersize = 1000000  # 100万行ずつフェッチ
        cur.execute(query)
        
        batch_count = 0
        for row in cur:
            postgres_keys.add((row[0], row[1]))
            batch_count += 1
            
            if batch_count % 10000000 == 0:
                print(f"  処理済み: {batch_count:,} 行")
    
    conn.close()
    
    print(f"PostgreSQLユニークキー数: {len(postgres_keys):,}")
    
    return postgres_keys, total_rows

def analyze_missing_keys(parquet_keys, postgres_keys, sample_size=1000):
    """欠落キーの分析"""
    print("\n欠落キーの分析中...")
    
    # 欠落キーを特定
    missing_keys = postgres_keys - parquet_keys
    print(f"欠落キー数: {len(missing_keys):,}")
    
    # ParquetにあってPostgreSQLにないキー
    extra_keys = parquet_keys - postgres_keys
    print(f"余分なキー数: {len(extra_keys):,}")
    
    # サンプル抽出
    missing_sample = list(missing_keys)[:sample_size]
    
    # 欠落キーのパターン分析
    print(f"\n欠落キーのサンプル（最初の{min(10, len(missing_sample))}件）:")
    for i, (orderkey, linenumber) in enumerate(missing_sample[:10]):
        print(f"  {i+1}: lo_orderkey={orderkey}, lo_linenumber={linenumber}")
    
    # ページ分布確認用のSQL生成
    if missing_sample:
        print("\n欠落行のページ分布確認用SQL:")
        keys_condition = " OR ".join([
            f"(lo_orderkey = {k[0]} AND lo_linenumber = {k[1]})"
            for k in missing_sample[:100]
        ])
        
        sql = f"""
WITH missing_rows AS (
    SELECT ctid, lo_orderkey, lo_linenumber
    FROM lineorder
    WHERE {keys_condition}
)
SELECT 
    (ctid::text::point)[0]::int as page_number,
    COUNT(*) as missing_count,
    array_agg(lo_orderkey || ',' || lo_linenumber) as sample_keys
FROM missing_rows
GROUP BY page_number
ORDER BY page_number
LIMIT 20;
"""
        print(sql)
    
    return missing_keys, extra_keys

def save_analysis_results(missing_keys, extra_keys, df_parquet):
    """分析結果を保存"""
    print("\n分析結果を保存中...")
    
    # 欠落キーをCSVに保存（サンプル）
    if missing_keys:
        sample_size = min(100000, len(missing_keys))
        missing_df = pd.DataFrame(list(missing_keys)[:sample_size], 
                                  columns=['lo_orderkey', 'lo_linenumber'])
        missing_df.to_csv('missing_keys_sample.csv', index=False)
        print(f"欠落キーのサンプル({sample_size:,}件)を missing_keys_sample.csv に保存")
    
    # 重複キーの分析
    duplicates = df_parquet[df_parquet.duplicated(['lo_orderkey', 'lo_linenumber'], keep=False)]
    if len(duplicates) > 0:
        dup_sample = duplicates.head(1000)
        dup_sample.to_csv('duplicate_keys_sample.csv', index=False)
        print(f"重複キーのサンプル({len(dup_sample):,}件)を duplicate_keys_sample.csv に保存")
        
        # 重複パターンの統計
        dup_stats = duplicates.groupby(['lo_orderkey', 'lo_linenumber']).size().reset_index(name='count')
        print(f"\n重複の統計:")
        print(f"  最大重複数: {dup_stats['count'].max()}")
        print(f"  平均重複数: {dup_stats['count'].mean():.2f}")

def main():
    start_time = time.time()
    
    # Parquetデータ読み込み
    parquet_keys, df_parquet = load_parquet_keys()
    
    # PostgreSQLデータ読み込み（メモリ制限のため最初の1億行）
    # 完全な分析が必要な場合は limit=None にする
    postgres_keys, total_rows = load_postgres_keys(limit=100000000)
    
    # 欠落キー分析
    missing_keys, extra_keys = analyze_missing_keys(parquet_keys, postgres_keys)
    
    # 結果保存
    save_analysis_results(missing_keys, extra_keys, df_parquet)
    
    # サマリー表示
    print("\n=== 分析サマリー ===")
    print(f"PostgreSQL総行数: {total_rows:,}")
    print(f"Parquet行数: {len(df_parquet):,}")
    print(f"カバー率: {len(df_parquet) / total_rows * 100:.1f}%")
    print(f"欠落キー数: {len(missing_keys):,}")
    print(f"余分なキー数: {len(extra_keys):,}")
    print(f"重複レコード数: {len(df_parquet) - len(parquet_keys):,}")
    
    elapsed = time.time() - start_time
    print(f"\n処理時間: {elapsed:.1f}秒")

if __name__ == "__main__":
    main()