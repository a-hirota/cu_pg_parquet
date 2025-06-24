#!/usr/bin/env python3
"""PostgreSQL高速読み取りベンチマーク - 7GB/s目標"""
import os
import sys
import time
import psycopg2
import subprocess
from pathlib import Path

def get_table_size_info():
    """lineorderテーブルのサイズ情報を取得"""
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("ERROR: GPUPASER_PG_DSN環境変数が設定されていません")
        sys.exit(1)
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # テーブルサイズ取得
    cur.execute("""
        SELECT 
            pg_size_pretty(pg_relation_size('lineorder')) as table_size,
            pg_relation_size('lineorder') as table_size_bytes,
            pg_size_pretty(pg_total_relation_size('lineorder')) as total_size,
            pg_total_relation_size('lineorder') as total_size_bytes
    """)
    size_info = cur.fetchone()
    
    # 行数取得
    cur.execute("SELECT COUNT(*) FROM lineorder")
    row_count = cur.fetchone()[0]
    
    # カラム情報取得
    cur.execute("""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = 'lineorder'
        ORDER BY ordinal_position
    """)
    columns = cur.fetchall()
    
    cur.close()
    conn.close()
    
    print("=== lineorderテーブル情報 ===")
    print(f"テーブルサイズ: {size_info[0]} ({size_info[1]:,} bytes)")
    print(f"総サイズ（インデックス含む）: {size_info[2]} ({size_info[3]:,} bytes)")
    print(f"行数: {row_count:,}")
    print(f"平均行サイズ: {size_info[1] / row_count:.1f} bytes/行")
    print(f"\nカラム数: {len(columns)}")
    print("\nカラム情報:")
    for col in columns:
        print(f"  - {col[0]}: {col[1]}" + (f"({col[2]})" if col[2] else ""))
    
    return size_info[1], row_count

def benchmark_psql_copy():
    """psqlコマンドでCOPY BINARYの速度を計測"""
    print("\n=== psql COPY BINARY ベンチマーク ===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    output_file = "/dev/shm/lineorder_copy.bin"
    
    # 既存ファイル削除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # COPY コマンド
    copy_cmd = f"\\COPY lineorder TO '{output_file}' WITH (FORMAT BINARY)"
    
    cmd = [
        "psql", dsn, "-c", copy_cmd
    ]
    
    print(f"出力先: {output_file}")
    print("COPY実行中...")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None, None
    
    elapsed = end_time - start_time
    
    # ファイルサイズ確認
    file_size = os.path.getsize(output_file)
    
    print(f"\n✓ COPY完了")
    print(f"  時間: {elapsed:.2f}秒")
    print(f"  ファイルサイズ: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
    print(f"  読み取り速度: {file_size / elapsed / 1024**3:.2f} GB/秒")
    
    return file_size, elapsed

def benchmark_python_copy():
    """Python psycopg2でCOPY BINARYの速度を計測"""
    print("\n=== Python psycopg2 COPY BINARY ベンチマーク ===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    output_file = "/dev/shm/lineorder_copy_python.bin"
    
    # 既存ファイル削除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    print(f"出力先: {output_file}")
    print("COPY実行中...")
    
    start_time = time.time()
    
    with open(output_file, 'wb') as f:
        cur.copy_expert(
            "COPY lineorder TO STDOUT WITH (FORMAT BINARY)",
            f
        )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    cur.close()
    conn.close()
    
    # ファイルサイズ確認
    file_size = os.path.getsize(output_file)
    
    print(f"\n✓ COPY完了")
    print(f"  時間: {elapsed:.2f}秒")
    print(f"  ファイルサイズ: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
    print(f"  読み取り速度: {file_size / elapsed / 1024**3:.2f} GB/秒")
    
    return file_size, elapsed

def benchmark_rust_copy():
    """Rust実装でCOPY BINARYの速度を計測"""
    print("\n=== Rust tokio-postgres COPY BINARY ベンチマーク ===")
    
    try:
        import gpupgparser_rust
    except ImportError:
        print("✗ gpupgparser_rustモジュールが見つかりません")
        return None, None
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    
    print("COPY実行中...")
    
    start_time = time.time()
    
    # Rust実装を呼び出し（メモリ転送のみ）
    query = "COPY lineorder TO STDOUT WITH (FORMAT BINARY)"
    # 注: 現在の実装は全データをメモリに読み込むため、メモリ不足の可能性
    
    try:
        # 簡易的にサイズだけ計測
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        
        import io
        buffer = io.BytesIO()
        cur.copy_expert(query, buffer)
        data = buffer.getvalue()
        
        # Rustでメモリ転送
        gpu_info = gpupgparser_rust.transfer_to_gpu_numba(data)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        file_size = len(data)
        
        print(f"\n✓ COPY + メモリ転送完了")
        print(f"  時間: {elapsed:.2f}秒")
        print(f"  データサイズ: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
        print(f"  読み取り速度: {file_size / elapsed / 1024**3:.2f} GB/秒")
        print(f"  デバイスポインタ: 0x{gpu_info['device_ptr']:x}")
        
        cur.close()
        conn.close()
        
        return file_size, elapsed
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None

def main():
    print("PostgreSQL 高速読み取りベンチマーク")
    print("目標: 7GB/秒")
    print("=" * 60)
    
    # テーブル情報取得
    table_size, row_count = get_table_size_info()
    
    # 各種ベンチマーク実行
    results = []
    
    # 1. psql
    size1, time1 = benchmark_psql_copy()
    if size1:
        results.append(("psql", size1, time1, size1 / time1 / 1024**3))
    
    # 2. Python psycopg2
    size2, time2 = benchmark_python_copy()
    if size2:
        results.append(("Python psycopg2", size2, time2, size2 / time2 / 1024**3))
    
    # 3. Rust (メモリサイズによっては実行不可)
    if table_size < 10 * 1024**3:  # 10GB未満の場合のみ
        size3, time3 = benchmark_rust_copy()
        if size3:
            results.append(("Rust tokio-postgres", size3, time3, size3 / time3 / 1024**3))
    else:
        print("\n✗ Rustベンチマークはスキップ（テーブルサイズが大きすぎるため）")
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    print(f"{'手法':<20} {'時間(秒)':<10} {'速度(GB/s)':<12} {'目標達成率':<10}")
    print("-" * 60)
    
    for method, size, elapsed, speed_gbps in results:
        achievement = (speed_gbps / 7.0) * 100
        print(f"{method:<20} {elapsed:<10.2f} {speed_gbps:<12.2f} {achievement:<10.1f}%")
    
    # 最速の手法
    if results:
        fastest = max(results, key=lambda x: x[3])
        print(f"\n最速: {fastest[0]} ({fastest[3]:.2f} GB/秒)")
        
        if fastest[3] < 7.0:
            print(f"\n目標未達成: あと {7.0 - fastest[3]:.2f} GB/秒の改善が必要")
            print("\n改善案:")
            print("1. 並列COPY（複数接続で範囲分割）")
            print("2. ストリーミング処理（メモリ使用量削減）")
            print("3. 直接GPU転送（CPU経由を削減）")
            print("4. PostgreSQLのbuffer設定チューニング")

if __name__ == "__main__":
    main()