#!/usr/bin/env python3
"""
欠落行分析スクリプト
==================

ParquetファイルとPostgreSQLのlineorderテーブルを比較し、
どの行が出力されていないかを特定します。
"""

import pandas as pd
import pyarrow.parquet as pq
import psycopg2
import numpy as np
from pathlib import Path
import sys
import os
from typing import Set, Tuple
import time

def extract_parquet_keys(output_dir: str = "output") -> Set[Tuple[int, int]]:
    """ParquetファイルからOrderKey, LineNumberを抽出"""
    print("=== Parquetファイルからキーを抽出 ===")
    
    parquet_files = sorted(Path(output_dir).glob("*.parquet"))
    if not parquet_files:
        print(f"エラー: {output_dir}にParquetファイルが見つかりません")
        return set()
    
    print(f"検出されたParquetファイル: {len(parquet_files)}個")
    
    all_keys = set()
    total_rows = 0
    
    for i, file in enumerate(parquet_files):
        print(f"  [{i+1}/{len(parquet_files)}] {file.name}を読み込み中...", end="", flush=True)
        
        # lo_orderkeyとlo_linenumberのみを読み込み（メモリ効率化）
        df = pd.read_parquet(file, columns=['lo_orderkey', 'lo_linenumber'])
        
        # データ型を確認して必要に応じて変換
        if df['lo_orderkey'].dtype == 'object':
            # Decimal型の場合、文字列として読み込まれることがある
            df['lo_orderkey'] = df['lo_orderkey'].astype(str).str.replace('.0', '', regex=False).astype(np.int64)
        if df['lo_linenumber'].dtype == 'object':
            df['lo_linenumber'] = df['lo_linenumber'].astype(str).str.replace('.0', '', regex=False).astype(np.int32)
        
        # キーをセットに追加
        file_keys = set(zip(df['lo_orderkey'].values, df['lo_linenumber'].values))
        all_keys.update(file_keys)
        
        rows = len(df)
        total_rows += rows
        print(f" {rows:,}行")
    
    print(f"\n合計: {total_rows:,}行, ユニークキー: {len(all_keys):,}個")
    
    # サンプル表示
    print("\nParquetキーのサンプル（最初の5個）:")
    for i, key in enumerate(sorted(list(all_keys)[:5])):
        print(f"  {key}")
    
    return all_keys

def extract_postgres_keys(limit: int = None) -> Set[Tuple[int, int]]:
    """PostgreSQLからOrderKey, LineNumberを抽出"""
    print("\n=== PostgreSQLからキーを抽出 ===")
    
    # DSN環境変数から接続
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("エラー: GPUPASER_PG_DSN環境変数が設定されていません")
        return set()
    
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        
        # まず総行数を確認
        cur.execute("SELECT COUNT(*) FROM lineorder")
        total_count = cur.fetchone()[0]
        print(f"lineorderテーブルの総行数: {total_count:,}")
        
        # キーを取得
        if limit:
            print(f"最初の{limit:,}行のみを取得します...")
            query = f"SELECT lo_orderkey, lo_linenumber FROM lineorder ORDER BY ctid LIMIT {limit}"
        else:
            print("全行を取得します（時間がかかる場合があります）...")
            query = "SELECT lo_orderkey, lo_linenumber FROM lineorder ORDER BY ctid"
        
        start_time = time.time()
        cur.execute(query)
        
        # 結果をセットに格納
        postgres_keys = set()
        batch_size = 100000
        rows_fetched = 0
        
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            
            postgres_keys.update(rows)
            rows_fetched += len(rows)
            
            if rows_fetched % 1000000 == 0:
                print(f"  {rows_fetched:,}行取得済み...")
        
        elapsed = time.time() - start_time
        print(f"取得完了: {len(postgres_keys):,}個のキー ({elapsed:.1f}秒)")
        
        # サンプル表示
        print("\nPostgreSQLキーのサンプル（最初の5個）:")
        for i, key in enumerate(sorted(list(postgres_keys)[:5])):
            print(f"  {key}")
        
        cur.close()
        conn.close()
        
        return postgres_keys
        
    except Exception as e:
        print(f"PostgreSQL接続エラー: {e}")
        return set()

def analyze_missing_keys(postgres_keys: Set[Tuple[int, int]], 
                        parquet_keys: Set[Tuple[int, int]]) -> None:
    """欠落キーの分析"""
    print("\n=== 欠落キーの分析 ===")
    
    # 基本統計
    print(f"PostgreSQLキー数: {len(postgres_keys):,}")
    print(f"Parquetキー数: {len(parquet_keys):,}")
    print(f"カバー率: {len(parquet_keys) / len(postgres_keys) * 100:.1f}%")
    
    # 差分計算
    missing_keys = postgres_keys - parquet_keys
    extra_keys = parquet_keys - postgres_keys
    
    print(f"\n欠落キー数: {len(missing_keys):,}")
    print(f"余分なキー数: {len(extra_keys):,}")
    
    if missing_keys:
        # 欠落キーのサンプル表示
        print("\n欠落キーのサンプル（最初の10個）:")
        for i, key in enumerate(sorted(list(missing_keys)[:10])):
            print(f"  {key}")
        
        # 欠落キーをCSVに保存
        missing_df = pd.DataFrame(list(missing_keys), columns=['lo_orderkey', 'lo_linenumber'])
        missing_df = missing_df.sort_values(['lo_orderkey', 'lo_linenumber'])
        missing_df.to_csv('missing_keys.csv', index=False)
        print(f"\n欠落キーをmissing_keys.csvに保存しました（{len(missing_keys):,}行）")
        
        # ページ分布分析のためのSQLを生成
        print("\n以下のSQLでページ分布を分析できます:")
        print("----")
        print("""
WITH missing_keys AS (
    -- missing_keys.csvの内容をここに読み込む
    SELECT lo_orderkey::bigint, lo_linenumber::integer
    FROM '/tmp/missing_keys.csv' -- psqlの\\copyコマンドで読み込み
)
SELECT 
    (l.ctid::text::point)[0]::int as page_num,
    COUNT(*) as missing_rows,
    MIN(l.lo_orderkey) as min_orderkey,
    MAX(l.lo_orderkey) as max_orderkey
FROM lineorder l
JOIN missing_keys m 
    ON l.lo_orderkey = m.lo_orderkey 
    AND l.lo_linenumber = m.lo_linenumber
GROUP BY page_num
ORDER BY missing_rows DESC
LIMIT 20;
        """)
        print("----")
    
    if extra_keys:
        print(f"\n警告: Parquetに{len(extra_keys):,}個の余分なキーが含まれています")
        print("余分なキーのサンプル（最初の5個）:")
        for i, key in enumerate(sorted(list(extra_keys)[:5])):
            print(f"  {key}")

def analyze_page_distribution():
    """欠落行のページ分布を分析"""
    print("\n=== ページ分布分析用のPythonスクリプト ===")
    
    print("""
# 欠落行のページ分析（PostgreSQL内で実行）
import psycopg2
import pandas as pd
import os

# missing_keys.csvを読み込み
missing_df = pd.read_csv('missing_keys.csv')

# PostgreSQLに一時テーブルを作成
dsn = os.environ.get('GPUPASER_PG_DSN')
conn = psycopg2.connect(dsn)
cur = conn.cursor()

# 一時テーブル作成
cur.execute('''
CREATE TEMP TABLE missing_keys (
    lo_orderkey BIGINT,
    lo_linenumber INTEGER
)
''')

# データを挿入（バッチ処理）
from psycopg2.extras import execute_values
data = [(int(row['lo_orderkey']), int(row['lo_linenumber'])) 
        for _, row in missing_df.iterrows()]
execute_values(cur, 
    "INSERT INTO missing_keys (lo_orderkey, lo_linenumber) VALUES %s",
    data
)

# ページ分布を分析
cur.execute('''
SELECT 
    (l.ctid::text::point)[0]::int as page_num,
    COUNT(*) as missing_rows,
    MIN(l.lo_orderkey) as min_orderkey,
    MAX(l.lo_orderkey) as max_orderkey
FROM lineorder l
JOIN missing_keys m 
    ON l.lo_orderkey = m.lo_orderkey 
    AND l.lo_linenumber = m.lo_linenumber
GROUP BY page_num
ORDER BY missing_rows DESC
LIMIT 20
''')

print("\\n欠落行が多いページTOP20:")
for row in cur.fetchall():
    print(f"  ページ{row[0]}: {row[1]:,}行欠落 (OrderKey範囲: {row[2]}-{row[3]})")

# チャンク境界での分析
cur.execute('''
SELECT 
    page_num / 288430 as chunk_id,  -- 288430 = pages_per_chunk
    COUNT(*) as missing_rows
FROM (
    SELECT (l.ctid::text::point)[0]::int as page_num
    FROM lineorder l
    JOIN missing_keys m 
        ON l.lo_orderkey = m.lo_orderkey 
        AND l.lo_linenumber = m.lo_linenumber
) t
GROUP BY chunk_id
ORDER BY chunk_id
''')

print("\\nチャンクごとの欠落行数:")
for row in cur.fetchall():
    print(f"  チャンク{int(row[0])}: {row[1]:,}行欠落")

conn.close()
    """)

def main():
    """メイン処理"""
    print("GPUPGParser 欠落行分析ツール")
    print("=" * 60)
    
    # Parquetキーを抽出
    parquet_keys = extract_parquet_keys()
    if not parquet_keys:
        print("エラー: Parquetキーの抽出に失敗しました")
        return 1
    
    # PostgreSQLキーを抽出（最初は制限付きでテスト）
    limit = None
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        limit = 1000000  # 100万行でサンプリング
        print(f"\n注意: サンプルモード（最初の{limit:,}行のみ）")
    
    postgres_keys = extract_postgres_keys(limit)
    if not postgres_keys:
        print("エラー: PostgreSQLキーの抽出に失敗しました")
        return 1
    
    # 欠落キーを分析
    analyze_missing_keys(postgres_keys, parquet_keys)
    
    # ページ分布分析の手順を表示
    analyze_page_distribution()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())