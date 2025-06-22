#!/usr/bin/env python3
"""PostgreSQL読み取り速度の詳細測定"""
import os
import time
import psycopg2
import subprocess

def measure_table_info():
    """lineorderテーブルの詳細情報を取得"""
    dsn = os.environ.get('GPUPASER_PG_DSN', "dbname=postgres user=postgres host=localhost port=5432")
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # テーブルサイズ
    cur.execute("""
        SELECT 
            pg_size_pretty(pg_relation_size('lineorder')) as table_size,
            pg_relation_size('lineorder') as table_size_bytes
    """)
    size_info = cur.fetchone()
    
    # 行数（LIMITで高速化）
    cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname='lineorder'")
    estimated_rows = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    print("=== lineorderテーブル情報 ===")
    print(f"テーブルサイズ: {size_info[0]} ({size_info[1] / 1024**3:.2f} GB)")
    print(f"推定行数: {estimated_rows:,}")
    print(f"推定平均行サイズ: {size_info[1] / estimated_rows:.1f} bytes/行")
    
    return size_info[1], estimated_rows

def benchmark_psql_limit(limit=1000000):
    """psqlでLIMIT付きCOPY速度測定"""
    print(f"\n=== psql COPY BINARY (LIMIT {limit:,}) ===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN', "dbname=postgres user=postgres host=localhost port=5432")
    output_file = f"/dev/shm/lineorder_sample_{limit}.bin"
    
    # 既存ファイル削除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # COPY コマンド
    copy_cmd = f"\\COPY (SELECT * FROM lineorder LIMIT {limit}) TO '{output_file}' WITH (FORMAT BINARY)"
    
    cmd = ["psql", dsn, "-c", copy_cmd]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None, None, None
    
    elapsed = end_time - start_time
    
    # ファイルサイズ確認
    file_size = os.path.getsize(output_file)
    
    print(f"✓ COPY完了")
    print(f"  時間: {elapsed:.2f}秒")
    print(f"  ファイルサイズ: {file_size:,} bytes ({file_size / 1024**2:.2f} MB)")
    print(f"  読み取り速度: {file_size / elapsed / 1024**2:.2f} MB/秒")
    print(f"  推定フルテーブル速度: {file_size / elapsed / 1024**3 * (estimated_rows / limit):.2f} GB/秒")
    
    return file_size, elapsed, limit

def benchmark_parallel_copy():
    """並列COPY実験"""
    print("\n=== 並列COPY実験 (4並列) ===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN', "dbname=postgres user=postgres host=localhost port=5432")
    
    # テーブルのページ数を取得
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute("SELECT relpages FROM pg_class WHERE relname='lineorder'")
    total_pages = cur.fetchone()[0]
    cur.close()
    conn.close()
    
    pages_per_worker = total_pages // 4
    
    # 並列COPYコマンド生成
    commands = []
    for i in range(4):
        start_page = i * pages_per_worker
        end_page = (i + 1) * pages_per_worker if i < 3 else total_pages
        
        output_file = f"/dev/shm/lineorder_part_{i}.bin"
        
        # ctidによる範囲指定
        copy_cmd = f"""
        psql "{dsn}" -c "\\COPY (
            SELECT * FROM lineorder 
            WHERE ctid >= '({start_page},0)'::tid 
            AND ctid < '({end_page},0)'::tid
        ) TO '{output_file}' WITH (FORMAT BINARY)"
        """
        
        commands.append(copy_cmd)
    
    # 並列実行
    start_time = time.time()
    
    processes = []
    for cmd in commands:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(p)
    
    # 全プロセス待機
    for p in processes:
        p.wait()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 合計サイズ計算
    total_size = 0
    for i in range(4):
        file_path = f"/dev/shm/lineorder_part_{i}.bin"
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)
    
    print(f"✓ 並列COPY完了")
    print(f"  時間: {elapsed:.2f}秒")
    print(f"  合計サイズ: {total_size:,} bytes ({total_size / 1024**3:.2f} GB)")
    print(f"  読み取り速度: {total_size / elapsed / 1024**3:.2f} GB/秒")

def main():
    print("PostgreSQL読み取り速度測定")
    print("目標: 7GB/秒")
    print("=" * 60)
    
    global estimated_rows
    table_size, estimated_rows = measure_table_info()
    
    # 1. 小サンプル測定
    limits = [100000, 1000000, 10000000]
    for limit in limits:
        benchmark_psql_limit(limit)
    
    # 2. 並列COPY実験
    benchmark_parallel_copy()
    
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("現在の単一接続での読み取り速度は目標の7GB/秒に達していません。")
    print("\n改善案:")
    print("1. PostgreSQL設定の最適化")
    print("   - shared_buffers増加")
    print("   - effective_io_concurrency調整")
    print("2. 並列COPY実装（上記実験参照）")
    print("3. ストリーミング処理でメモリ効率改善")
    print("4. GPU Direct Storage活用")

if __name__ == "__main__":
    main()