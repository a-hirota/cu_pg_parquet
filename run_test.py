#!/usr/bin/env python3
"""
テスト実行スクリプト（対話型入力を避ける）
"""
import os
import glob
import subprocess

# 既存のparquetファイルを削除
for f in glob.glob("output/*.parquet"):
    os.remove(f)
    print(f"削除: {f}")

# 環境変数設定
os.environ["GPUPASER_PG_DSN"] = "host=localhost dbname=postgres user=postgres"
os.environ["GPUPGPARSER_TEST_MODE"] = "1"

# テスト実行
result = subprocess.run([
    "python", "cu_pg_parquet.py",
    "--test", "--table", "customer",
    "--parallel", "2", "--chunks", "2"
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)