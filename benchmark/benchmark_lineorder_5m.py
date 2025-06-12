#!/usr/bin/env python3
"""
GPUパーサー ベンチマークメイン
=============================

GPUソート最適化の性能を測定するためのベンチマークツール

使用方法:
    python benchmark_main.py --rows 5000000    # 500万行
    python benchmark_main.py --rows 10000000   # 1000万行
    python benchmark_main.py --test gpu_sort   # GPUソート性能テスト
"""

import argparse
import sys
import os
import time
import subprocess
from pathlib import Path

# プロジェクトルートをPATHに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_table(rows: int):
    """指定行数のテストテーブルを作成"""
    
    print(f"📊 {rows:,}行のテストテーブルを作成中...")
    
    # PostgreSQL接続設定
    dsn = os.environ.get('GPUPASER_PG_DSN', 'dbname=postgres user=postgres host=localhost port=5432')
    
    # テストテーブル作成SQLを生成
    sql_commands = f"""
-- テストテーブル削除（存在する場合）
DROP TABLE IF EXISTS lineorder_test_{rows};

-- テストテーブル作成
CREATE TABLE lineorder_test_{rows} AS
SELECT 
    (random() * 1000000)::int as lo_orderkey,
    (random() * 100000)::int as lo_linenumber,
    (random() * 200000)::int as lo_custkey,
    (random() * 40000)::int as lo_partkey,
    (random() * 10000)::int as lo_suppkey,
    ('1992-01-01'::date + (random() * 2500)::int)::date as lo_orderdate,
    ('P'::char || (random() * 9 + 1)::int::text)::char(1) as lo_orderpriority,
    (random() * 100000)::int as lo_shippriority,
    (random() * 1000000 + 100000)::numeric(15,2) as lo_quantity,
    (random() * 10000000 + 1000000)::numeric(15,2) as lo_extendedprice,
    (random() * 100 + 1)::numeric(15,2) as lo_ordtotalprice,
    (random() * 50 + 1)::numeric(15,2) as lo_discount,
    (random() * 100000000 + 10000000)::numeric(15,2) as lo_revenue,
    (random() * 100000 + 10000)::numeric(15,2) as lo_supplycost,
    (random() * 1000000 + 100000)::numeric(15,2) as lo_tax,
    ('1995-01-01'::date + (random() * 1000)::int)::date as lo_commitdate,
    ('SHIP' || (random() * 999 + 1)::int::text)::char(10) as lo_shipmode
FROM generate_series(1, {rows});

-- インデックス作成（高速化）
CREATE INDEX idx_lineorder_test_{rows}_orderkey ON lineorder_test_{rows}(lo_orderkey);

-- 統計情報更新
ANALYZE lineorder_test_{rows};

-- 確認
SELECT 'テーブル作成完了:', count(*) as row_count FROM lineorder_test_{rows};
"""
    
    # SQLファイルに保存
    sql_file = f"input/create_test_table_{rows}.sql"
    os.makedirs("input", exist_ok=True)
    
    with open(sql_file, 'w', encoding='utf-8') as f:
        f.write(sql_commands)
    
    print(f"📝 SQLファイル作成: {sql_file}")
    
    # PostgreSQLでSQL実行
    try:
        print("🔧 PostgreSQLでテーブル作成中...")
        result = subprocess.run([
            'psql', dsn, '-f', sql_file
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ テストテーブル作成完了")
            print(result.stdout.split('\n')[-3])  # 行数確認行を表示
        else:
            print(f"❌ テーブル作成エラー: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ テーブル作成がタイムアウトしました")
        return False
    except FileNotFoundError:
        print("❌ psqlコマンドが見つかりません。PostgreSQLがインストールされていることを確認してください。")
        return False
    
    return True

def run_gpu_parser_benchmark(rows: int):
    """GPUパーサーベンチマーク実行"""
    
    print(f"\n🚀 GPUパーサーベンチマーク開始 ({rows:,}行)")
    
    # ベンチマークスクリプト作成
    benchmark_script = f"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# プロジェクトパス設定
sys.path.insert(0, '/home/ubuntu/gpupgparser')

from src.main_postgres_to_parquet import process_postgres_table_to_parquet_optimized

def run_benchmark():
    table_name = 'lineorder_test_{rows}'
    output_file = f'benchmark/lineorder_test_{{rows}}_gpu_optimized.parquet'
    
    print(f"📊 ベンチマーク開始: {{table_name}}")
    print(f"📁 出力ファイル: {{output_file}}")
    
    start_time = time.perf_counter()
    
    try:
        # GPUパーサー実行（最新の最適化版）
        result_df = process_postgres_table_to_parquet_optimized(
            table_name=table_name,
            output_path=output_file,
            use_integrated_parser=True,  # 統合パーサー使用
            debug=True
        )
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        print(f"✅ ベンチマーク完了")
        print(f"⏱️  実行時間: {{elapsed_time:.2f}}秒")
        print(f"📊 処理行数: {{len(result_df):,}}行")
        print(f"🔥 スループット: {{len(result_df)/elapsed_time:.0f}}行/秒")
        
        # ファイルサイズ確認
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024 / 1024
            print(f"📁 出力ファイルサイズ: {{file_size:.2f}}MB")
        
        return elapsed_time, len(result_df)
        
    except Exception as e:
        print(f"❌ ベンチマークエラー: {{e}}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == '__main__':
    run_benchmark()
"""
    
    # 一時的なベンチマークスクリプトファイル作成
    benchmark_file = f"benchmark_temp_{rows}.py"
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        f.write(benchmark_script)
    
    try:
        # ベンチマーク実行
        print("🔧 GPUパーサー実行中...")
        result = subprocess.run([
            sys.executable, benchmark_file
        ], timeout=1800, capture_output=True, text=True)  # 30分タイムアウト
        
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
            
        if result.returncode != 0:
            print(f"❌ ベンチマーク実行エラー (終了コード: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ ベンチマークがタイムアウトしました")
        return False
    finally:
        # 一時ファイル削除
        if os.path.exists(benchmark_file):
            os.remove(benchmark_file)
    
    return True

def run_gpu_sort_performance_test():
    """GPUソート性能テスト実行"""
    
    print("\n🧪 GPUソート性能テスト開始")
    
    try:
        result = subprocess.run([
            sys.executable, "test/test_gpu_sort_simple.py"
        ], timeout=300, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
            
        if result.returncode == 0:
            print("✅ GPUソート性能テスト完了")
        else:
            print(f"❌ テスト実行エラー (終了コード: {result.returncode})")
            
    except subprocess.TimeoutExpired:
        print("⏰ テストがタイムアウトしました")
    except FileNotFoundError:
        print("❌ テストファイルが見つかりません")

def main():
    parser = argparse.ArgumentParser(
        description="GPUパーサー ベンチマークツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python benchmark_main.py --rows 5000000          # 500万行のベンチマーク
  python benchmark_main.py --rows 10000000         # 1000万行のベンチマーク
  python benchmark_main.py --test gpu_sort         # GPUソート性能テスト
  python benchmark_main.py --rows 1000000 --create # テーブル作成のみ
        """
    )
    
    parser.add_argument(
        '--rows', 
        type=int, 
        help='テストデータの行数 (例: 5000000)'
    )
    parser.add_argument(
        '--test', 
        choices=['gpu_sort'], 
        help='実行するテストタイプ'
    )
    parser.add_argument(
        '--create', 
        action='store_true', 
        help='テーブル作成のみ実行'
    )
    
    args = parser.parse_args()
    
    print("🚀 GPUパーサー ベンチマークツール")
    print("=" * 50)
    
    if args.test == 'gpu_sort':
        run_gpu_sort_performance_test()
        return
    
    if not args.rows:
        parser.print_help()
        print("\n❌ --rows または --test オプションが必要です")
        return
    
    # 環境確認
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("⚠️  警告: GPUPASER_PG_DSN環境変数が設定されていません")
        print("例: export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'")
    
    # テーブル作成
    if not create_test_table(args.rows):
        print("❌ テーブル作成に失敗しました")
        return
    
    if args.create:
        print("✅ テーブル作成完了（--createオプションのため、ベンチマークは実行しません）")
        return
    
    # ベンチマーク実行
    if not run_gpu_parser_benchmark(args.rows):
        print("❌ ベンチマークに失敗しました")
        return
    
    print("\n🎉 すべての処理が完了しました！")

if __name__ == '__main__':
    main()
