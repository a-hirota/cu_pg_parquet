#!/usr/bin/env python3
"""Rust実装の簡易ベンチマーク（CPUメモリ版）"""
import os
import sys
import time
import psycopg2
import psycopg2.extras

def benchmark_rust_copy_binary():
    """Rust実装のCOPY BINARY取得速度を計測"""
    print("Rust実装のCOPY BINARY取得速度ベンチマーク")
    print("=" * 60)
    
    # gpupgparser_rustモジュールの確認
    try:
        import gpupgparser_rust
        print("✓ gpupgparser_rust: インポート成功")
    except ImportError:
        print("✗ gpupgparser_rust: 見つかりません")
        return
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("ERROR: GPUPASER_PG_DSN環境変数が設定されていません")
        return
    
    # テスト用クエリ
    queries = [
        ("小規模（1万行）", "COPY (SELECT generate_series(1, 10000) AS id, 'test_' || generate_series(1, 10000) AS name) TO STDOUT WITH BINARY"),
        ("中規模（10万行）", "COPY (SELECT generate_series(1, 100000) AS id, 'test_' || generate_series(1, 100000) AS name) TO STDOUT WITH BINARY"),
        ("大規模（100万行）", "COPY (SELECT generate_series(1, 1000000) AS id, 'test_' || generate_series(1, 1000000) AS name) TO STDOUT WITH BINARY"),
    ]
    
    print("\n=== Rust実装（transfer_to_gpu_numba）===")
    
    for label, query in queries:
        # Python側でCOPY BINARYデータ取得
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        
        import io
        output = io.BytesIO()
        
        start = time.time()
        cur.copy_expert(query, output)
        binary_data = output.getvalue()
        copy_time = time.time() - start
        
        output.close()
        cur.close()
        conn.close()
        
        # Rust実装でメモリ転送
        start = time.time()
        gpu_info = gpupgparser_rust.transfer_to_gpu_numba(binary_data)
        transfer_time = time.time() - start
        
        total_time = copy_time + transfer_time
        
        print(f"\n{label}:")
        print(f"  COPY取得時間: {copy_time:.3f}秒")
        print(f"  転送時間: {transfer_time:.3f}秒")
        print(f"  合計時間: {total_time:.3f}秒")
        print(f"  データサイズ: {len(binary_data) / 1024 / 1024:.1f}MB")
        print(f"  スループット: {len(binary_data) / total_time / 1024 / 1024:.1f}MB/秒")
        print(f"  デバイスポインタ: 0x{gpu_info['device_ptr']:x}")

def benchmark_python_copy_binary():
    """Python実装のCOPY BINARY取得速度を計測（比較用）"""
    print("\n\n=== Python実装（比較用）===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        return
    
    queries = [
        ("小規模（1万行）", "COPY (SELECT generate_series(1, 10000) AS id, 'test_' || generate_series(1, 10000) AS name) TO STDOUT WITH BINARY"),
        ("中規模（10万行）", "COPY (SELECT generate_series(1, 100000) AS id, 'test_' || generate_series(1, 100000) AS name) TO STDOUT WITH BINARY"),
        ("大規模（100万行）", "COPY (SELECT generate_series(1, 1000000) AS id, 'test_' || generate_series(1, 1000000) AS name) TO STDOUT WITH BINARY"),
    ]
    
    for label, query in queries:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        
        import io
        output = io.BytesIO()
        
        start = time.time()
        cur.copy_expert(query, output)
        binary_data = output.getvalue()
        total_time = time.time() - start
        
        output.close()
        cur.close()
        conn.close()
        
        print(f"\n{label}:")
        print(f"  合計時間: {total_time:.3f}秒")
        print(f"  データサイズ: {len(binary_data) / 1024 / 1024:.1f}MB")
        print(f"  スループット: {len(binary_data) / total_time / 1024 / 1024:.1f}MB/秒")

def benchmark_rust_string_builder():
    """RustStringBuilderのベンチマーク"""
    print("\n\n=== RustStringBuilder性能テスト ===")
    
    from src.rust_integration import RustStringBuilder
    
    test_sizes = [
        ("小規模", 10000),
        ("中規模", 100000),
        ("大規模", 1000000),
    ]
    
    for label, size in test_sizes:
        # テスト文字列生成
        test_strings = [f"test_string_{i}".encode() for i in range(size)]
        
        # Python版（比較用）
        start = time.time()
        python_list = []
        for s in test_strings:
            python_list.append(s.decode())
        python_time = time.time() - start
        
        # Rust版
        builder = RustStringBuilder()
        start = time.time()
        for s in test_strings:
            builder.add_string(s)
        rust_time = time.time() - start
        
        print(f"\n{label}（{size:,}文字列）:")
        print(f"  Python時間: {python_time:.3f}秒")
        print(f"  Rust時間: {rust_time:.3f}秒")
        print(f"  高速化率: {python_time / rust_time:.1f}x")

def main():
    benchmark_rust_copy_binary()
    benchmark_python_copy_binary()
    benchmark_rust_string_builder()
    
    print("\n" + "=" * 60)
    print("ベンチマーク完了")

if __name__ == "__main__":
    main()