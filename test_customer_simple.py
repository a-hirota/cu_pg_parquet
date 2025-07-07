#!/usr/bin/env python3
"""
customerテーブルの修正効果を簡単にテスト
"""

import os
import sys
sys.path.append('/home/ubuntu/gpupgparser')

from src.readPostgres.metadata import fetch_column_meta
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2_lite
import psycopg
import cupy as cp

# 環境設定
os.environ['GPUPGPARSER_TEST_MODE'] = '1'
os.environ['TABLE_NAME'] = 'customer'

# PostgreSQL接続
dsn = os.environ.get("GPUPASER_PG_DSN", "")
conn = psycopg.connect(dsn)

# メタデータ取得
columns = fetch_column_meta(conn, "SELECT * FROM customer")
print(f"列数: {len(columns)}")

# ダミーデータで推定テスト
dummy_size = 100 * 1024 * 1024  # 100MB
dummy_data = cp.zeros(dummy_size, dtype=cp.uint8)

print("\n=== 推定計算のテスト ===")
try:
    # parse関数は内部で推定計算を行う
    result = parse_binary_chunk_gpu_ultra_fast_v2_lite(
        dummy_data, 
        columns, 
        header_size=19,
        debug=True,
        test_mode=True
    )
    print("✅ エラーなく完了")
except Exception as e:
    print(f"❌ エラー: {e}")

conn.close()