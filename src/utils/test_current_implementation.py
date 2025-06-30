#!/usr/bin/env python3
"""
現在の実装での行数を確認するテストスクリプト
"""

import os
import subprocess
import json
import psycopg

def test_single_chunk():
    """単一チャンクでテスト"""
    
    print("=== 単一チャンクテスト ===")
    
    # 環境変数設定
    env = os.environ.copy()
    env["CHUNK_ID"] = "0"
    env["TOTAL_CHUNKS"] = "1"
    env["TABLE_NAME"] = "lineorder"
    env["GPUPGPARSER_TEST_MODE"] = "1"
    
    # Rustプログラムを実行
    print("Rustプログラムを実行中...")
    result = subprocess.run(
        ["/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"エラー: {result.stderr}")
        return
    
    # JSON結果を抽出
    lines = result.stdout.split("\n")
    json_start = False
    json_lines = []
    
    for line in lines:
        if "===CHUNK_RESULT_JSON===" in line:
            json_start = True
            continue
        if "===END_CHUNK_RESULT_JSON===" in line:
            break
        if json_start:
            json_lines.append(line)
    
    if json_lines:
        chunk_info = json.loads("\n".join(json_lines))
        print(f"チャンクファイル: {chunk_info['chunk_file']}")
        print(f"データサイズ: {chunk_info['total_bytes'] / 1024**3:.2f} GB")
    
    # PostgreSQLの実際の行数を確認
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM lineorder")
    actual_rows = cursor.fetchone()[0]
    print(f"\nPostgreSQL実際の行数: {actual_rows:,}")
    
    cursor.close()
    conn.close()
    
    # GPUで処理してみる（簡易版）
    print("\nGPU処理をテスト中...")
    
    # benchmark_rust_gpu_direct.pyの一部を実行
    cmd = f"""
cd /home/ubuntu/gpupgparser && python -c "
import os
os.environ['TOTAL_CHUNKS'] = '1'
os.environ['TABLE_NAME'] = 'lineorder'
os.environ['GPUPGPARSER_TEST_MODE'] = '1'

# 必要なインポート
import sys
sys.path.append('.')
from docs.benchmark.benchmark_rust_gpu_direct import main

# 1チャンクで実行
main(total_chunks=1, table_name='lineorder')
"
"""
    
    subprocess.run(cmd, shell=True)

def quick_duplicate_check():
    """簡易的な重複チェック"""
    
    print("\n=== 重複の可能性チェック ===")
    
    # 32チャンクでの合計が正確に一致することを確認済み
    print("32チャンクでのRust側の合計: 246,012,324行（正確）")
    print("GPUパーサー報告値: 246,031,120行（+18,796行）")
    
    print("\n可能性：")
    print("1. GPUパーサーの重複検出（複数スレッドが同じ行を処理）")
    print("2. チャンク境界での行の重複カウント")
    print("3. ソート無効化による順序の問題")
    print("4. atomic操作の競合条件")
    
    print("\n推奨アクション：")
    print("1. GPUパーサーに重複検出ロジックを追加")
    print("2. 各スレッドの担当範囲を厳密に管理")
    print("3. ソートを再有効化して順序を保証")

if __name__ == "__main__":
    # test_single_chunk()
    quick_duplicate_check()