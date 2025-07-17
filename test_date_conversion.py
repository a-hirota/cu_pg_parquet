#!/usr/bin/env python3
"""
date型の変換が正しく動作するかテストするスクリプト
"""

import psycopg2
import subprocess
import cudf
import pandas as pd
from datetime import datetime, date

# PostgreSQL接続
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="postgres",
    host="localhost"
)

# テストテーブルの作成
cur = conn.cursor()

# 既存のテーブルを削除（存在する場合）
cur.execute("DROP TABLE IF EXISTS test_date_conversion")

# date型を含むテーブルを作成
cur.execute("""
CREATE TABLE test_date_conversion (
    id INTEGER,
    test_date DATE,
    description TEXT
)
""")

# テストデータを挿入
test_data = [
    (1, date(2000, 1, 1), "2000-01-01 - PostgreSQL date epoch"),
    (2, date(1970, 1, 1), "1970-01-01 - Unix epoch"),
    (3, date(2024, 12, 25), "2024-12-25 - Christmas 2024"),
    (4, date(1999, 12, 31), "1999-12-31 - Day before 2000"),
    (5, None, "NULL date value")
]

for row in test_data:
    cur.execute("INSERT INTO test_date_conversion VALUES (%s, %s, %s)", row)

conn.commit()
cur.close()
conn.close()

print("テストデータの準備完了")

# cu_pg_parquet.pyを実行
print("\ncu_pg_parquet.pyを実行中...")
result = subprocess.run([
    "python", "cu_pg_parquet.py",
    "--table", "test_date_conversion",
    "--parallel", "1",
    "--chunks", "1"
], capture_output=True, text=True)

if result.returncode != 0:
    print(f"エラー: {result.stderr}")
    exit(1)

print("変換完了")

# Parquetファイルを読み込んで検証
print("\nParquetファイルを検証中...")
df = cudf.read_parquet("output/test_date_conversion_chunk_0_queue.parquet")

print("\n変換結果:")
print(df)

# データ型の確認
print("\nデータ型:")
print(df.dtypes)

# test_date列がdatetime64[s]になっているか確認
if df['test_date'].dtype.name.startswith('datetime64'):
    print(f"\n✓ test_date列は正しくdatetime型に変換されました: {df['test_date'].dtype}")
else:
    print(f"\n✗ test_date列の型が間違っています: {df['test_date'].dtype}")

# 値の検証
print("\n期待値との比較:")
expected_dates = [
    pd.Timestamp('2000-01-01'),
    pd.Timestamp('1970-01-01'),
    pd.Timestamp('2024-12-25'),
    pd.Timestamp('1999-12-31'),
    pd.NaT
]

# cuDFからPandasに変換して比較
df_pandas = df.to_pandas()
for i, (actual, expected) in enumerate(zip(df_pandas['test_date'], expected_dates)):
    if pd.isna(expected):
        if pd.isna(actual):
            print(f"Row {i}: ✓ NULL値が正しく処理されました")
        else:
            print(f"Row {i}: ✗ NULL値の処理に失敗 (actual: {actual})")
    else:
        if actual == expected:
            print(f"Row {i}: ✓ {actual} == {expected}")
        else:
            print(f"Row {i}: ✗ {actual} != {expected}")

# クリーンアップ
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres", 
    password="postgres",
    host="localhost"
)
cur = conn.cursor()
cur.execute("DROP TABLE test_date_conversion")
conn.commit()
cur.close()
conn.close()

print("\nテスト完了")