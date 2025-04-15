"""
lineorderテーブルのデータをParquet形式に変換するテスト
"""

import os
import sys
import time
import numpy as np
import cudf
import psycopg2
from typing import Dict

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpupaser.pg_connector import PostgresConnector
from gpupaser.binary_parser import BinaryParser
from gpupaser.output_handler import OutputHandler

def test_lineorder_small_parquet():
    """少量のlineorderデータをParquetに変換するテスト"""
    # 出力ディレクトリ設定
    output_dir = "test/lineorder_test_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/lineorder_small.parquet"
    
    # PostgreSQL接続情報
    db_params = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost"
    }
    
    # lineorderテーブルに対する接続
    pg_conn = PostgresConnector(**db_params)
    
    # テーブル情報の取得
    table_name = "lineorder"
    if not pg_conn.check_table_exists(table_name):
        print(f"テーブル {table_name} が見つかりません")
        return
        
    columns = pg_conn.get_table_info(table_name)
    
    # バイナリデータの取得 (最初の10行)
    binary_data, _ = pg_conn.get_binary_data(table_name, limit=10)
    
    # バイナリパーサーの初期化
    parser = BinaryParser()
    
    # 出力ハンドラの初期化
    output_handler = OutputHandler(parquet_output=output_path)
    
    # バイナリデータの解析と変換
    start_time = time.time()
    result = parser.parse_postgres_binary(binary_data, columns)
    print(f"解析時間: {time.time() - start_time:.4f}秒")
    
    if result:
        # 結果をParquetに出力
        output_handler.process_chunk_result(result)
        output_handler.close()
        
        # cuDFでParquetファイルを読み込んで内容確認
        try:
            print("\n=== cuDFでParquetファイルを読み込み ===")
            gdf = cudf.read_parquet(output_path)
            print(f"行数: {len(gdf)}")
            print(f"列名: {gdf.columns.tolist()}")
            print("\n最初の数行:")
            print(gdf.head())
            
            # カラムのデータ型を出力
            print("\nカラムデータ型:")
            for col_name, dtype in zip(gdf.columns, gdf.dtypes):
                print(f"{col_name}: {dtype}")
                
            return True
        except Exception as e:
            print(f"Parquet読み込みエラー: {e}")
            return False
    else:
        print("データの解析に失敗しました")
        return False

def test_lineorder_performance():
    """lineorderテーブルの性能テスト (6,000,000行)"""
    # 出力ディレクトリ設定
    output_dir = "test/lineorder_perf_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # PostgreSQL接続情報
    db_params = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost"
    }
    
    # lineorderテーブルに対する接続
    pg_conn = PostgresConnector(**db_params)
    
    # テーブル情報の取得
    table_name = "lineorder"
    if not pg_conn.check_table_exists(table_name):
        print(f"テーブル {table_name} が見つかりません")
        return False
        
    columns = pg_conn.get_table_info(table_name)
    total_rows = pg_conn.get_table_row_count(table_name)
    
    # 行数制限（最大6,000,000行）
    max_rows = min(6_000_000, total_rows)
    print(f"処理対象行数: {max_rows}")
    
    # チャンクサイズの設定
    chunk_size = 500_000
    num_chunks = (max_rows + chunk_size - 1) // chunk_size
    
    # 出力ハンドラの初期化（チャンク単位でParquetファイルを作成）
    output_handlers = {}
    
    # バイナリパーサーの初期化
    parser = BinaryParser()
    
    # 時間計測開始
    total_start_time = time.time()
    processed_rows = 0
    
    # チャンク単位で処理
    for chunk_idx in range(num_chunks):
        chunk_start_time = time.time()
        
        # チャンクの範囲設定
        offset = chunk_idx * chunk_size
        limit = min(chunk_size, max_rows - offset)
        
        if limit <= 0:
            break
            
        print(f"\n=== チャンク {chunk_idx + 1}/{num_chunks} 処理中 (オフセット: {offset}, 行数: {limit}) ===")
        
        # 出力ファイル設定
        output_path = f"{output_dir}/lineorder_chunk_{chunk_idx}.parquet"
        output_handlers[chunk_idx] = OutputHandler(parquet_output=output_path)
        
        # バイナリデータの取得
        binary_data, _ = pg_conn.get_binary_data(table_name, limit=limit, offset=offset)
        
        # データの解析と変換
        chunk_parse_start = time.time()
        result = parser.parse_postgres_binary(binary_data, columns)
        chunk_parse_time = time.time() - chunk_parse_start
        
        if not result:
            print(f"チャンク {chunk_idx} の解析に失敗しました")
            continue
            
        # 結果をParquetに出力
        chunk_write_start = time.time()
        output_handlers[chunk_idx].process_chunk_result(result)
        output_handlers[chunk_idx].close()
        chunk_write_time = time.time() - chunk_write_start
        
        # 行数の取得
        row_count = len(next(iter(result.values())))
        processed_rows += row_count
        
        # チャンク処理時間の出力
        chunk_total_time = time.time() - chunk_start_time
        print(f"チャンク処理時間: {chunk_total_time:.4f}秒")
        print(f"  解析時間: {chunk_parse_time:.4f}秒")
        print(f"  書き込み時間: {chunk_write_time:.4f}秒")
        print(f"  処理速度: {row_count / chunk_total_time:.2f} rows/sec")
        
    # 全体の処理時間を計算
    total_time = time.time() - total_start_time
    print(f"\n=== 処理完了 ===")
    print(f"合計処理時間: {total_time:.4f}秒")
    print(f"処理行数: {processed_rows}")
    print(f"全体の処理速度: {processed_rows / total_time:.2f} rows/sec")
    
    return True

if __name__ == "__main__":
    # まず少量のデータでテスト
    print("=== 少量のlineorderデータテスト ===")
    if test_lineorder_small_parquet():
        # 正常に処理できたら性能テスト実行
        print("\n=== lineorderテーブル性能テスト ===")
        test_lineorder_performance()
