#!/usr/bin/env python3
"""Rust実装とPython実装の速度比較ベンチマーク"""
import os
import sys
import time
import psycopg2
import psycopg2.extras
import cudf
import numpy as np
from numba import cuda
from src.rust_integration import PostgresGPUReader, RustStringBuilder
import subprocess

def ensure_test_table():
    """テスト用の大規模テーブルを作成"""
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("ERROR: GPUPASER_PG_DSN環境変数が設定されていません")
        sys.exit(1)
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # テーブルの存在確認
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'benchmark_test'
        )
    """)
    
    if not cur.fetchone()[0]:
        print("大規模テストテーブルを作成中...")
        cur.execute("DROP TABLE IF EXISTS benchmark_test")
        cur.execute("""
            CREATE TABLE benchmark_test (
                id INTEGER,
                name TEXT,
                description TEXT,
                value DECIMAL(15,2),
                created_at TIMESTAMP
            )
        """)
        
        # 100万行のテストデータを生成
        print("100万行のテストデータを生成中...")
        cur.execute("""
            INSERT INTO benchmark_test (id, name, description, value, created_at)
            SELECT 
                i,
                'User_' || i,
                'Description for user ' || i || ' with some additional text to make it longer',
                (random() * 10000)::decimal(15,2),
                NOW() - (random() * interval '365 days')
            FROM generate_series(1, 1000000) AS i
        """)
        conn.commit()
        print("✓ テストテーブル作成完了")
    else:
        print("✓ 既存のテストテーブルを使用")
    
    # 行数確認
    cur.execute("SELECT COUNT(*) FROM benchmark_test")
    row_count = cur.fetchone()[0]
    print(f"  行数: {row_count:,}")
    
    cur.close()
    conn.close()

def benchmark_rust_implementation():
    """Rust実装のベンチマーク"""
    print("\n=== Rust実装のベンチマーク ===")
    
    reader = PostgresGPUReader()
    
    # ウォームアップ
    query = "COPY (SELECT name FROM benchmark_test LIMIT 1000) TO STDOUT WITH BINARY"
    reader.fetch_string_column_to_gpu(query, column_index=0)
    
    # 本番計測
    queries = [
        ("1万行", "COPY (SELECT name FROM benchmark_test LIMIT 10000) TO STDOUT WITH BINARY"),
        ("10万行", "COPY (SELECT name FROM benchmark_test LIMIT 100000) TO STDOUT WITH BINARY"),
        ("100万行", "COPY (SELECT name FROM benchmark_test) TO STDOUT WITH BINARY"),
    ]
    
    results = []
    for label, query in queries:
        start = time.time()
        
        # PostgreSQL → GPU転送
        gpu_buffers = reader.fetch_string_column_to_gpu(query, column_index=0)
        
        # cuDF Series作成
        series = reader.create_cudf_series_from_gpu_buffers(gpu_buffers)
        
        end = time.time()
        elapsed = end - start
        
        print(f"\n{label}:")
        print(f"  時間: {elapsed:.3f}秒")
        print(f"  行数: {gpu_buffers['row_count']:,}")
        print(f"  速度: {gpu_buffers['row_count'] / elapsed:,.0f} 行/秒")
        
        results.append((label, elapsed, gpu_buffers['row_count']))
    
    return results

def benchmark_python_implementation():
    """既存Python実装のベンチマーク"""
    print("\n=== Python実装のベンチマーク（比較用） ===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    queries = [
        ("1万行", "SELECT name FROM benchmark_test LIMIT 10000"),
        ("10万行", "SELECT name FROM benchmark_test LIMIT 100000"),
        ("100万行", "SELECT name FROM benchmark_test"),
    ]
    
    results = []
    for label, query in queries:
        start = time.time()
        
        # PostgreSQLからデータ取得
        cur.execute(query)
        rows = cur.fetchall()
        
        # Python配列に変換
        names = [row[0] for row in rows]
        
        # cuDF Seriesに変換
        series = cudf.Series(names)
        
        end = time.time()
        elapsed = end - start
        
        print(f"\n{label}:")
        print(f"  時間: {elapsed:.3f}秒")
        print(f"  行数: {len(rows):,}")
        print(f"  速度: {len(rows) / elapsed:,.0f} 行/秒")
        
        results.append((label, elapsed, len(rows)))
    
    cur.close()
    conn.close()
    
    return results

def benchmark_copy_binary_python():
    """COPY BINARY + Python処理のベンチマーク"""
    print("\n=== COPY BINARY + Python処理のベンチマーク ===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    import io
    
    queries = [
        ("1万行", "COPY (SELECT name FROM benchmark_test LIMIT 10000) TO STDOUT WITH BINARY"),
        ("10万行", "COPY (SELECT name FROM benchmark_test LIMIT 100000) TO STDOUT WITH BINARY"),
        ("100万行", "COPY (SELECT name FROM benchmark_test) TO STDOUT WITH BINARY"),
    ]
    
    results = []
    for label, query in queries:
        start = time.time()
        
        # COPY BINARYでデータ取得
        output = io.BytesIO()
        cur.copy_expert(query, output)
        binary_data = output.getvalue()
        output.close()
        
        # 簡易パース（実際のパースは省略）
        # ここでは単純にバイト数を行数の推定に使用
        estimated_rows = len(binary_data) // 50  # 1行あたり約50バイトと仮定
        
        end = time.time()
        elapsed = end - start
        
        print(f"\n{label}:")
        print(f"  時間: {elapsed:.3f}秒")
        print(f"  バイト数: {len(binary_data):,}")
        print(f"  推定行数: {estimated_rows:,}")
        print(f"  速度: {len(binary_data) / elapsed / 1024 / 1024:.1f} MB/秒")
        
        results.append((label, elapsed, len(binary_data)))
    
    cur.close()
    conn.close()
    
    return results

def print_comparison(rust_results, python_results, copy_binary_results):
    """結果の比較表示"""
    print("\n" + "=" * 80)
    print("速度比較サマリー")
    print("=" * 80)
    
    print("\n{:<15} {:>15} {:>15} {:>15} {:>15}".format(
        "データサイズ", "Rust実装", "Python実装", "COPY BINARY", "高速化率"
    ))
    print("-" * 80)
    
    for i, (label, _, _) in enumerate(rust_results):
        rust_time = rust_results[i][1]
        python_time = python_results[i][1]
        copy_time = copy_binary_results[i][1]
        
        speedup_vs_python = python_time / rust_time
        speedup_vs_copy = copy_time / rust_time
        
        print("{:<15} {:>13.3f}秒 {:>13.3f}秒 {:>13.3f}秒 {:>10.1f}x/{:.1f}x".format(
            label, rust_time, python_time, copy_time, speedup_vs_python, speedup_vs_copy
        ))

def main():
    print("Rust実装 vs Python実装 速度比較ベンチマーク")
    print("=" * 80)
    
    # 環境確認
    print("環境確認:")
    try:
        import gpupgparser_rust
        print("  ✓ gpupgparser_rust: インポート成功")
    except ImportError:
        print("  ✗ gpupgparser_rust: 見つかりません")
        sys.exit(1)
    
    # GPUメモリ情報
    try:
        cuda_device = cuda.get_current_device()
        print(f"  ✓ GPU: {cuda_device.name}")
        # メモリ情報取得を試行
        try:
            free, total = cuda.current_context().get_memory_info()
            print(f"  ✓ GPUメモリ: {free / 1024**3:.1f}GB / {total / 1024**3:.1f}GB")
        except:
            print(f"  ✓ GPUメモリ: 情報取得不可")
    except:
        print("  ✓ GPU: 情報取得不可")
    
    # テーブル準備
    ensure_test_table()
    
    # ベンチマーク実行
    try:
        rust_results = benchmark_rust_implementation()
        python_results = benchmark_python_implementation()
        copy_binary_results = benchmark_copy_binary_python()
        
        # 結果比較
        print_comparison(rust_results, python_results, copy_binary_results)
        
    except Exception as e:
        print(f"\n✗ ベンチマーク失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()