"""
lineorderテーブルのデータをマルチGPUでParquet形式に変換するテスト
"""

import os
import sys
import time
import numpy as np
import cudf
import psycopg2
from typing import Dict, List, Any
import concurrent.futures
from numba import cuda

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpupaser.pg_connector import PostgresConnector
from gpupaser.binary_parser import BinaryParser
from gpupaser.output_handler import OutputHandler

def get_gpu_count():
    """利用可能なGPU数を取得"""
    try:
        count = len(cuda.gpus)
        print(f"利用可能なGPU数: {count}")
        return count
    except Exception as e:
        print(f"GPU情報の取得に失敗: {e}")
        return 1  # デフォルトは1

def process_chunk(chunk_idx: int, offset: int, limit: int, table_name: str, 
                 columns: List, output_dir: str, db_params: Dict, gpu_id: int):
    """チャンクを処理する関数（GPUごとに並行実行）"""
    # 使用するGPUを設定
    try:
        cuda.select_device(gpu_id)
        print(f"GPU {gpu_id} を選択しました（チャンク {chunk_idx}）")
    except Exception as e:
        print(f"GPU {gpu_id} の選択に失敗: {e}")
        return False
    
    try:
        # PostgreSQLに接続
        pg_conn = PostgresConnector(**db_params)
        
        # 出力設定
        output_path = f"{output_dir}/lineorder_chunk_{chunk_idx}_gpu{gpu_id}.parquet"
        output_handler = OutputHandler(parquet_output=output_path)
        
        # バイナリデータの取得（重要：offset指定）
        print(f"チャンク {chunk_idx} データ取得開始: オフセット={offset}, 行数={limit}")
        start_time = time.time()
        binary_data, _ = pg_conn.get_binary_data(table_name, limit=limit, offset=offset)
        fetch_time = time.time() - start_time
        print(f"チャンク {chunk_idx} データ取得完了: {len(binary_data)} バイト, {fetch_time:.2f}秒")
        
        # バイナリパーサーの初期化（GPUデバイスを指定）
        parser = BinaryParser(use_gpu=True)
        
        # データの解析と変換
        parse_start = time.time()
        result = parser.parse_postgres_binary(binary_data, columns)
        parse_time = time.time() - parse_start
        
        if not result:
            print(f"チャンク {chunk_idx} の解析に失敗しました")
            return False
            
        # 行数の確認
        row_count = len(next(iter(result.values())))
        print(f"チャンク {chunk_idx} 解析成功: {row_count}行, {parse_time:.2f}秒")
        
        # 結果をParquetに出力
        write_start = time.time()
        output_handler.process_chunk_result(result)
        output_handler.close()
        write_time = time.time() - write_start
        
        # 処理時間の計算
        total_time = fetch_time + parse_time + write_time
        
        print(f"チャンク {chunk_idx} 処理完了 (GPU {gpu_id})")
        print(f"  取得時間: {fetch_time:.2f}秒")
        print(f"  解析時間: {parse_time:.2f}秒")
        print(f"  書込時間: {write_time:.2f}秒")
        print(f"  合計時間: {total_time:.2f}秒")
        print(f"  処理速度: {row_count / total_time:.2f} rows/sec")
        
        return row_count
    except Exception as e:
        print(f"チャンク {chunk_idx} の処理中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return 0

def test_lineorder_multigpu():
    """lineorderテーブルのマルチGPU処理テスト"""
    # 出力ディレクトリ設定
    output_dir = "test/lineorder_multigpu_output"
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
    
    # 利用可能なGPU数を取得
    gpu_count = get_gpu_count()
    
    # チャンクサイズの設定（GPU数を考慮）
    # 各GPUが処理するチャンク数（少ないが大きなチャンクの方が効率的）
    chunks_per_gpu = 2
    # 大幅に拡大したチャンクサイズ（最大100万行）
    chunk_size = max(100000, min(1_000_000, max_rows // (gpu_count * chunks_per_gpu)))
    
    # GPUメモリ容量に基づく動的調整
    try:
        # 利用可能なGPUメモリを取得
        context = cuda.current_context()
        free_memory = context.get_memory_info()[0]
        # 1行あたりの推定メモリ使用量（lineorderテーブルの列数と型に基づく）
        bytes_per_row = 1000  # 概算：1KB/行
        # GPUメモリの70%までを使用
        max_rows_by_memory = int(0.7 * free_memory / bytes_per_row)
        # 安全側に立った上限値設定
        chunk_size = min(chunk_size, max_rows_by_memory)
        print(f"GPUメモリに基づく調整後のチャンクサイズ: {chunk_size}行")
    except Exception as e:
        print(f"GPUメモリ情報取得エラー（デフォルト値を使用）: {e}")
    
    # チャンク数の計算
    num_chunks = (max_rows + chunk_size - 1) // chunk_size
    print(f"チャンク設定: サイズ={chunk_size}行, 数={num_chunks}個")
    print(f"GPU数: {gpu_count}, GPU当たりチャンク数: {chunks_per_gpu}")
    
    # 時間計測開始
    total_start_time = time.time()
    
    # GPUとチャンクの割り当て
    tasks = []
    for chunk_idx in range(num_chunks):
        offset = chunk_idx * chunk_size
        limit = min(chunk_size, max_rows - offset)
        
        if limit <= 0:
            break
            
        gpu_id = chunk_idx % gpu_count
        tasks.append((chunk_idx, offset, limit, table_name, columns, output_dir, db_params, gpu_id))
    
    # 並列実行（ThreadPoolExecutorを使用）
    processed_rows = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(process_chunk, *task))
        
        # 各タスクの結果を取得
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if isinstance(result, int):
                    processed_rows += result
            except Exception as e:
                print(f"タスク実行中にエラー: {e}")
    
    # 全体の処理時間を計算
    total_time = time.time() - total_start_time
    print(f"\n=== 処理完了 ===")
    print(f"合計処理時間: {total_time:.2f}秒")
    print(f"処理行数: {processed_rows}")
    print(f"全体の処理速度: {processed_rows / total_time:.2f} rows/sec")
    
    return True

def test_lineorder_parquet_combined():
    """複数のParquetファイルを結合して読み込むテスト"""
    # 出力ディレクトリ設定
    input_dir = "test/lineorder_multigpu_output"
    output_path = f"{input_dir}/lineorder_combined.parquet"
    
    # 入力ファイルの検索
    parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                     if f.startswith("lineorder_chunk_") and f.endswith(".parquet")]
    
    if not parquet_files:
        print("結合するParquetファイルがありません")
        return False
    
    print(f"結合対象ファイル数: {len(parquet_files)}")
    
    # 結合テーブルを初期化
    combined_df = None
    total_rows = 0
    
    # 時間計測開始
    start_time = time.time()
    
    # ファイルを順次読み込んで結合
    for file_path in parquet_files:
        try:
            print(f"読み込み中: {file_path}")
            df = cudf.read_parquet(file_path)
            print(f"  行数: {len(df)}")
            
            if combined_df is None:
                combined_df = df
            else:
                combined_df = cudf.concat([combined_df, df], ignore_index=True)
                
            total_rows += len(df)
        except Exception as e:
            print(f"ファイル {file_path} の読み込み中にエラー: {e}")
    
    if combined_df is None:
        print("結合に失敗しました")
        return False
    
    # 結合したデータフレームを保存
    combined_df.to_parquet(output_path)
    
    # 処理時間の計算
    total_time = time.time() - start_time
    print(f"\n=== 結合完了 ===")
    print(f"出力ファイル: {output_path}")
    print(f"合計行数: {total_rows}")
    print(f"処理時間: {total_time:.2f}秒")
    
    return True

if __name__ == "__main__":
    # マルチGPUテスト実行
    if test_lineorder_multigpu():
        # 成功したら結合テストを実行
        test_lineorder_parquet_combined()
