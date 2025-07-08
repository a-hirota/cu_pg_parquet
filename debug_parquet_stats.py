#!/usr/bin/env python3
"""
Parquetファイル統計のデバッグ
"""
import sys
import os

# 環境変数を設定
os.environ["GPUPASER_PG_DSN"] = "host=localhost dbname=postgres user=postgres"
os.environ["GPUPGPARSER_TEST_MODE"] = "1"
os.environ["TABLE_NAME"] = "customer"

# sys.argvを設定
sys.argv = ["debug_parquet_stats.py", "--table", "customer", "--parallel", "2", "--chunks", "1"]

# 必要なモジュールをインポート
from docs.benchmark.benchmark_rust_gpu_direct import main

try:
    # デバッグ用にmain関数を直接呼び出し
    main(total_chunks=1, table_name="customer", test_mode=True)
except Exception as e:
    print(f"\nエラーの詳細:")
    import traceback
    traceback.print_exc()
    
    print(f"\nエラーメッセージ: {e}")
    print(f"エラーの型: {type(e)}")