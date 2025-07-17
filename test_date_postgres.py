#!/usr/bin/env python3
"""
PostgreSQLでdate型の変換をテストするスクリプト
"""

import subprocess
import cudf
import pandas as pd
from datetime import date
import os
import sys

# テストデータ準備用のSQLスクリプトを作成
sql_script = """
-- 既存のテストテーブルを削除
DROP TABLE IF EXISTS test_date_conversion;

-- date型を含むテーブルを作成
CREATE TABLE test_date_conversion (
    id INTEGER,
    test_date DATE,
    description TEXT
);

-- テストデータを挿入
INSERT INTO test_date_conversion VALUES 
    (1, '2000-01-01', '2000-01-01 - PostgreSQL date epoch'),
    (2, '1970-01-01', '1970-01-01 - Unix epoch'),
    (3, '2024-12-25', '2024-12-25 - Christmas 2024'),
    (4, '1999-12-31', '1999-12-31 - Day before 2000'),
    (5, NULL, 'NULL date value');

-- データを確認
SELECT * FROM test_date_conversion ORDER BY id;
"""

print("PostgreSQLにテストテーブルを作成中...")

# psqlコマンドでSQLを実行
result = subprocess.run(
    ["psql", "-U", "postgres", "-c", sql_script],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"エラー: {result.stderr}")
    sys.exit(1)

print("テストテーブル作成完了")
print(result.stdout)

# 既存のParquetファイルを削除
output_dir = "output"
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        if file.startswith("test_date_conversion") and file.endswith(".parquet"):
            os.remove(os.path.join(output_dir, file))

# cu_pg_parquet.pyを実行
print("\ncu_pg_parquet.pyを実行中...")
result = subprocess.run([
    "python", "cu_pg_parquet.py",
    "--table", "test_date_conversion",
    "--parallel", "1",
    "--chunks", "1"
], capture_output=True, text=True)

if result.returncode != 0:
    print(f"変換エラー: {result.stderr}")
    print(f"stdout: {result.stdout}")
    sys.exit(1)

print("変換完了")
print(result.stdout)

# Parquetファイルを読み込んで検証
print("\nParquetファイルを検証中...")
parquet_file = os.path.join(output_dir, "test_date_conversion_chunk_0_queue.parquet")

if not os.path.exists(parquet_file):
    print(f"エラー: {parquet_file}が見つかりません")
    sys.exit(1)

df = cudf.read_parquet(parquet_file)

print("\n変換結果:")
print(df)

# データ型の確認
print("\nデータ型:")
print(df.dtypes)

# test_date列がdatetime64になっているか確認
if 'test_date' in df.columns:
    if df['test_date'].dtype.name.startswith('datetime64'):
        print(f"\n✓ test_date列は正しくdatetime型に変換されました: {df['test_date'].dtype}")
    else:
        print(f"\n✗ test_date列の型が間違っています: {df['test_date'].dtype}")
else:
    print("\n✗ test_date列が見つかりません")

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
if 'test_date' in df_pandas.columns:
    for i, row in df_pandas.iterrows():
        actual = row['test_date']
        expected = expected_dates[i]
        desc = row['description']
        
        if pd.isna(expected):
            if pd.isna(actual):
                print(f"Row {i+1}: ✓ NULL値が正しく処理されました - {desc}")
            else:
                print(f"Row {i+1}: ✗ NULL値の処理に失敗 (actual: {actual}) - {desc}")
        else:
            if actual == expected:
                print(f"Row {i+1}: ✓ {actual} == {expected} - {desc}")
            else:
                print(f"Row {i+1}: ✗ {actual} != {expected} - {desc}")

# クリーンアップ
print("\nテストテーブルをクリーンアップ中...")
cleanup_result = subprocess.run(
    ["psql", "-U", "postgres", "-c", "DROP TABLE test_date_conversion;"],
    capture_output=True,
    text=True
)

if cleanup_result.returncode == 0:
    print("クリーンアップ完了")
else:
    print(f"クリーンアップエラー: {cleanup_result.stderr}")

print("\nテスト完了")