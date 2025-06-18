"""
PostgreSQL → kvikio直接ヒープファイル読み込み → GPU Processing → Arrow/Parquet

kvikioを使用してPostgreSQLのヒープファイルを直接GPUメモリに読み込み、
既存のgpupgparserパイプラインでArrow/Parquet形式に高速変換します。

環境変数:
GPUPASER_PG_DSN       : PostgreSQL接続文字列（メタデータ取得用）
POSTGRES_DATA_DIR     : PostgreSQLデータディレクトリ（/var/lib/postgresql/data等）
USE_KVIKIO_COMPAT     : kvikio互換モード（True/False, optional）

使用方法（postgresユーザ権限で実行）:
# 1. postgresユーザに切り替え
sudo su - postgres

# 2. 環境変数設定
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'
export POSTGRES_DATA_DIR='/var/lib/postgresql/data'

# 3. ベンチマーク実行
python /home/ubuntu/gpupgparser/benchmark/benchmark_kvikio_heap.py --table lineorder --database postgres

注意: PostgreSQLヒープファイルへのアクセスにはpostgresユーザ権限が必要です。
"""

import os
import time
import argparse
import warnings
from pathlib import Path
import psycopg
import cudf
from numba import cuda
import rmm

from src.metadata import fetch_column_meta
from src.types import ColumnMeta
from src.build_cudf_from_buf import integrate_kvikio_pipeline
from src.heap_file_reader import HeapFileReaderError

DEFAULT_DATABASE = "postgres"
DEFAULT_TABLE = "lineorder"
OUTPUT_DIR = "benchmark"

def check_kvikio_support():
    """kvikioサポート確認"""
    print("\n=== kvikio サポート確認 ===")
    
    try:
        import kvikio
        print(f"✅ kvikio バージョン: {kvikio.__version__}")
        
        # GPU Direct Storage確認
        if os.path.exists("/proc/driver/nvidia-fs/stats"):
            print("✅ nvidia-fs ドライバ検出")
        else:
            print("⚠️  nvidia-fs ドライバが見つかりません（互換モードで動作）")
            os.environ["KVIKIO_COMPAT_MODE"] = "ON"
        
        compat_mode = os.environ.get("KVIKIO_COMPAT_MODE", "OFF")
        print(f"✅ KVIKIO_COMPAT_MODE: {compat_mode}")
        
        return True
    except ImportError:
        print("❌ kvikio がインストールされていません")
        return False

def check_postgres_permissions(data_dir: str, database: str = "postgres", table: str = "lineorder"):
    """PostgreSQLデータディレクトリの読み取り権限確認"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"PostgreSQLデータディレクトリが見つかりません: {data_dir}")
    
    if not os.access(data_path, os.R_OK):
        import getpass
        current_user = getpass.getuser()
        raise PermissionError(
            f"PostgreSQLデータディレクトリへの読み取り権限がありません: {data_dir}\n"
            f"現在のユーザー: {current_user}\n"
            f"解決方法: 以下のコマンドでpostgresユーザに切り替えてください:\n"
            f"  sudo su - postgres\n"
            f"  export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser\n"
            f"  python /home/ubuntu/gpupgparser/benchmark/benchmark_kvikio_heap.py --table {table} --database {database}"
        )

def find_heap_file(data_dir: str, database: str, table: str):
    """PostgreSQLヒープファイルを検索"""
    print(f"\n=== ヒープファイル検索 ===")
    print(f"データディレクトリ: {data_dir}")
    print(f"データベース: {database}")
    print(f"テーブル: {table}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"PostgreSQLデータディレクトリが見つかりません: {data_dir}")
    
    # データベースOIDを取得するためにPostgreSQLに接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise ValueError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            # データベースOID取得
            cur.execute("SELECT oid FROM pg_database WHERE datname = %s", (database,))
            db_result = cur.fetchone()
            if not db_result:
                raise ValueError(f"データベース '{database}' が見つかりません")
            db_oid = db_result[0]
            
            # テーブルOID取得
            cur.execute("SELECT relfilenode FROM pg_class WHERE relname = %s AND relkind = 'r'", (table,))
            table_result = cur.fetchone()
            if not table_result:
                raise ValueError(f"テーブル '{table}' が見つかりません")
            relfilenode = table_result[0]
            
    finally:
        conn.close()
    
    # ヒープファイルパス構築
    heap_file_path = data_path / str(db_oid) / str(relfilenode)
    
    if not heap_file_path.exists():
        raise FileNotFoundError(f"ヒープファイルが見つかりません: {heap_file_path}")
    
    file_size = heap_file_path.stat().st_size
    print(f"✅ ヒープファイル発見: {heap_file_path}")
    print(f"   ファイルサイズ: {file_size / (1024*1024):.2f} MB")
    
    return str(heap_file_path)

def run_kvikio_benchmark(database: str, table: str, output_path: str = None):
    """kvikio版ベンチマーク実行"""
    
    print(f"\n=== PostgreSQL → kvikio直接ヒープ読み込み ベンチマーク ===")
    print(f"データベース: {database}")
    print(f"テーブル: {table}")
    
    if output_path is None:
        output_path = f"{OUTPUT_DIR}/{table}_kvikio_heap.output.parquet"
    
    print(f"出力先: {output_path}")
    
    # kvikioサポート確認
    if not check_kvikio_support():
        print("❌ kvikioサポートが利用できません")
        return
    
    # RMM初期化
    try:
        if not rmm.is_initialized():
            rmm.reinitialize(
                pool_allocator=True, 
                initial_pool_size=4*1024**3  # 4GB
            )
            print("✅ RMM メモリプール初期化完了 (4GB)")
    except Exception as e:
        print(f"❌ RMM初期化エラー: {e}")
        return
    
    start_total_time = time.time()
    
    # メタデータ取得
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("❌ 環境変数 GPUPASER_PG_DSN が設定されていません")
        return
    
    conn = psycopg.connect(dsn)
    try:
        print("\nメタデータを取得中...")
        start_meta_time = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {table}")
        meta_time = time.time() - start_meta_time
        print(f"✅ メタデータ取得完了 ({meta_time:.4f}秒)")
        print(f"   列数: {len(columns)}")
        
        for i, col in enumerate(columns[:5]):  # 最初の5列を表示
            print(f"   列{i+1}: {col.name} (OID:{col.pg_oid}, Arrow:{col.arrow_id})")
        if len(columns) > 5:
            print(f"   ... 他{len(columns)-5}列")
            
    finally:
        conn.close()
    
    # ヒープファイル検索
    data_dir = os.environ.get("POSTGRES_DATA_DIR")
    if not data_dir:
        print("❌ 環境変数 POSTGRES_DATA_DIR が設定されていません")
        return
    
    try:
        # PostgreSQL権限チェック
        check_postgres_permissions(data_dir, database, table)
        
        start_find_time = time.time()
        heap_file_path = find_heap_file(data_dir, database, table)
        find_time = time.time() - start_find_time
        print(f"✅ ヒープファイル検索完了 ({find_time:.4f}秒)")
    except PermissionError as e:
        print(f"❌ 権限エラー:\n{e}")
        return
    except Exception as e:
        print(f"❌ ヒープファイル検索エラー: {e}")
        return
    
    # kvikio統合パイプライン実行
    print(f"\nkvikio統合パイプライン実行中...")
    start_processing_time = time.time()
    
    try:
        cudf_df = integrate_kvikio_pipeline(heap_file_path, columns)
        processing_time = time.time() - start_processing_time
        rows = len(cudf_df)
        
        print(f"✅ kvikio統合パイプライン完了 ({processing_time:.4f}秒)")
        print(f"   処理行数: {rows:,} 行")
        
    except HeapFileReaderError as e:
        print(f"❌ ヒープファイル読み込みエラー: {e}")
        return
    except Exception as e:
        print(f"❌ kvikio統合パイプラインエラー: {e}")
        return
    
    # Parquet書き込み
    print(f"\nParquetファイル書き込み中...")
    start_write_time = time.time()
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cudf_df.to_parquet(output_path, compression='snappy', engine='cudf')
        write_time = time.time() - start_write_time
        
        file_size = os.path.getsize(output_path)
        print(f"✅ Parquet書き込み完了 ({write_time:.4f}秒)")
        print(f"   出力ファイルサイズ: {file_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ Parquet書き込みエラー: {e}")
        return
    
    total_time = time.time() - start_total_time
    decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
    
    # 結果表示
    print(f"\n=== kvikio直接ヒープ読み込みベンチマーク完了 ===")
    print(f"総時間: {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得      : {meta_time:.4f} 秒")
    print(f"  ヒープファイル検索  : {find_time:.4f} 秒")
    print(f"  kvikio統合パイプライン: {processing_time:.4f} 秒")
    print(f"  Parquet書き込み     : {write_time:.4f} 秒")
    print("--- 統計情報 ---")
    print(f"  処理行数    : {rows:,} 行")
    print(f"  処理列数    : {len(columns)} 列")
    print(f"  Decimal列数 : {decimal_cols} 列")
    
    # 従来のCOPY BINARY方式との比較用メトリクス
    heap_file_size = os.path.getsize(heap_file_path)
    throughput_mbps = (heap_file_size / (1024*1024)) / processing_time if processing_time > 0 else 0
    total_cells = rows * len(columns)
    cell_throughput = total_cells / processing_time if processing_time > 0 else 0
    
    print(f"  ヒープファイルサイズ: {heap_file_size / (1024*1024):.2f} MB")
    print(f"  処理スループット    : {throughput_mbps:.2f} MB/sec")
    print(f"  セル処理速度        : {cell_throughput:,.0f} cells/sec")
    
    # 性能評価
    baseline_copy_speed = 129.1  # PostgreSQL COPY BINARY baseline (MB/sec)
    improvement_ratio = throughput_mbps / baseline_copy_speed if baseline_copy_speed > 0 else 0
    
    print(f"  性能向上倍率        : {improvement_ratio:.1f}倍 (対COPY BINARY {baseline_copy_speed} MB/sec)")
    
    if throughput_mbps > 2000:
        performance_class = "🏆 超高速 (2GB/s+)"
    elif throughput_mbps > 1000:
        performance_class = "🥇 高速 (1GB/s+)"
    elif throughput_mbps > 500:
        performance_class = "🥈 中速 (500MB/s+)"
    else:
        performance_class = "🥉 改善中"
    
    print(f"  性能クラス          : {performance_class}")
    print("========================================")
    
    # 検証用読み込み
    print(f"\ncuDF検証用読み込み...")
    try:
        start_verify_time = time.time()
        verify_df = cudf.read_parquet(output_path)
        verify_time = time.time() - start_verify_time
        
        print(f"✅ cuDF検証完了 ({verify_time:.4f}秒)")
        print(f"   読み込み確認: {len(verify_df):,} 行 × {len(verify_df.columns)} 列")
        
        # 基本データ確認
        print("--- データ型確認 ---")
        for col_name, dtype in list(verify_df.dtypes.items())[:5]:
            print(f"  {col_name}: {dtype}")
        if len(verify_df.columns) > 5:
            print(f"  ... 他{len(verify_df.columns)-5}列")
            
    except Exception as e:
        print(f"❌ cuDF検証エラー: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL kvikio直接ヒープ読み込みベンチマーク')
    parser.add_argument('--database', type=str, default=DEFAULT_DATABASE, help='データベース名')
    parser.add_argument('--table', type=str, default=DEFAULT_TABLE, help='テーブル名')
    parser.add_argument('--output', type=str, help='出力Parquetファイルパス')
    parser.add_argument('--check-support', action='store_true', help='kvikioサポート確認のみ')
    
    args = parser.parse_args()
    
    # CUDA初期化確認
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    if args.check_support:
        check_kvikio_support()
        return
    
    # kvikio版ベンチマーク実行
    run_kvikio_benchmark(
        database=args.database,
        table=args.table,
        output_path=args.output
    )

if __name__ == "__main__":
    main()