#!/usr/bin/env python
"""
複数GPUを使用してPostgreSQLからデータを取得しParquetファイルに変換するスクリプト
各GPUが独立したプロセスで動作し、CUDAコンテキストの競合を避けます
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Process, Queue

# gpuPaserパッケージのパスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from gpupaser.main import PgGpuProcessor
from gpupaser.pg_connector import PostgresConnector
from gpupaser.utils import get_available_gpus


def process_chunk_on_gpu(gpu_id, table_name, start_row, chunk_size, output_path, db_params, queue):
    """
    単一GPUでデータチャンクを処理する関数
    
    Args:
        gpu_id: 使用するGPU ID
        table_name: 処理対象のテーブル名
        start_row: 処理開始行
        chunk_size: 処理する行数
        output_path: 出力ディレクトリパス
        db_params: データベース接続パラメータ
        queue: マルチプロセス間通信用キュー
    """
    try:
        # GPUを指定
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"GPU {gpu_id}: チャンク処理開始 (行 {start_row} から {start_row + chunk_size - 1})")
        
        # 処理時間計測
        start_time = time.time()
        
        # PostgreSQL接続
        pg_conn = PostgresConnector(
            dbname=db_params.get("dbname", "postgres"),
            user=db_params.get("user", "postgres"),
            password=db_params.get("password", "postgres"),
            host=db_params.get("host", "localhost")
        )
        
        # GPUプロセッサーを初期化
        processor = PgGpuProcessor(
            dbname=db_params.get("dbname", "postgres"),
            user=db_params.get("user", "postgres"),
            password=db_params.get("password", "postgres"),
            host=db_params.get("host", "localhost")
        )
        
        # クエリを実行（LIMIT と OFFSET を使用して指定範囲のみ取得）
        query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {start_row}"
        print(f"GPU {gpu_id}: 実行クエリ: {query}")
        
        # GPUでデータを処理
        result_df = processor.process_query(query)
        
        if result_df is not None and not result_df.empty:
            # 処理行数を確認
            processed_rows = len(result_df)
            print(f"GPU {gpu_id}: {processed_rows}行のデータを処理しました")
            
            # データフレームをPyArrowテーブルに変換してParquetに保存
            output_file = os.path.join(output_path, f"{table_name}_chunk_{gpu_id}.parquet")
            table = pa.Table.from_pandas(result_df)
            pq.write_table(table, output_file)
            
            print(f"GPU {gpu_id}: 結果をParquetファイルに保存: {output_file}")
            
            # 処理時間
            elapsed_time = time.time() - start_time
            rows_per_second = processed_rows / elapsed_time
            print(f"GPU {gpu_id}: 処理時間: {elapsed_time:.2f}秒 (スループット: {rows_per_second:.2f}行/秒)")
            
            # 結果をキューに入れる
            queue.put({
                "gpu_id": gpu_id,
                "start_row": start_row,
                "processed_rows": processed_rows,
                "output_file": output_file,
                "elapsed_time": elapsed_time,
                "success": True
            })
        else:
            print(f"GPU {gpu_id}: データが空または処理に失敗しました")
            queue.put({
                "gpu_id": gpu_id,
                "start_row": start_row,
                "success": False,
                "error": "空のデータセットまたは処理失敗"
            })
    
    except Exception as e:
        print(f"GPU {gpu_id}: エラー発生: {str(e)}")
        queue.put({
            "gpu_id": gpu_id,
            "start_row": start_row,
            "success": False,
            "error": str(e)
        })
        
def main():
    """
    メイン関数 - コマンドライン引数の解析とマルチGPU処理の実行
    """
    parser = argparse.ArgumentParser(description="複数GPUを使用してPostgreSQLデータをParquetに変換")
    parser.add_argument("--table", "-t", required=True, help="処理するテーブル名")
    parser.add_argument("--rows", "-r", type=int, default=10000, help="処理する総行数 (デフォルト: 10000)")
    parser.add_argument("--output", "-o", default="./ray_output", help="出力ディレクトリ (デフォルト: ./ray_output)")
    parser.add_argument("--gpus", "-g", type=int, help="使用するGPU数 (指定しない場合は自動検出)")
    parser.add_argument("--gpu_ids", "-i", help="使用するGPU IDのカンマ区切りリスト (例: '0,2')")
    parser.add_argument("--chunk_size", "-c", type=int, help="チャンクサイズ (指定しない場合は自動計算)")
    
    # PostgreSQL接続パラメータ
    parser.add_argument("--db_name", "-d", default="postgres", help="データベース名")
    parser.add_argument("--db_user", "-u", default="postgres", help="データベースユーザー")
    parser.add_argument("--db_password", "-p", default="postgres", help="データベースパスワード")
    parser.add_argument("--db_host", "-H", default="localhost", help="データベースホスト")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output, exist_ok=True)
    
    # 利用可能なGPUの検出
    available_gpus = get_available_gpus()
    
    if args.gpu_ids:
        # カンマ区切りのGPU IDリストを解析
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]
        print(f"指定されたGPU IDを使用: {gpu_ids}")
    elif args.gpus:
        # 指定された数のGPUを使用
        num_gpus = min(args.gpus, len(available_gpus))
        gpu_ids = available_gpus[:num_gpus]
        print(f"{num_gpus}個のGPUを使用: {gpu_ids}")
    else:
        # 利用可能な全てのGPUを使用
        gpu_ids = available_gpus
        print(f"利用可能な全GPU {len(gpu_ids)}個を使用: {gpu_ids}")
    
    num_gpus = len(gpu_ids)
    
    if num_gpus == 0:
        print("エラー: 使用可能なGPUが見つかりません")
        return
    
    # チャンクサイズの決定
    if args.chunk_size:
        chunk_size = args.chunk_size
    else:
        # デフォルトは1GPUあたり均等に分割
        chunk_size = (args.rows + num_gpus - 1) // num_gpus
    
    print(f"総行数: {args.rows}, GPU数: {num_gpus}, チャンクサイズ: {chunk_size}")
    
    # PostgreSQL接続パラメータ
    db_params = {
        "dbname": args.db_name,
        "user": args.db_user,
        "password": args.db_password,
        "host": args.db_host
    }
    
    # マルチプロセス間通信用のキュー
    result_queue = Queue()
    
    # 処理プロセスのリスト
    processes = []
    
    # 各GPUに処理を割り当て
    for i, gpu_id in enumerate(gpu_ids):
        start_row = i * chunk_size
        # 最大行数を超えないようにする
        current_chunk_size = min(chunk_size, args.rows - start_row)
        
        if current_chunk_size <= 0:
            # このGPUには割り当てるデータがない
            continue
        
        # 新しいプロセスを作成
        process = Process(
            target=process_chunk_on_gpu,
            args=(gpu_id, args.table, start_row, current_chunk_size, args.output, db_params, result_queue)
        )
        processes.append(process)
        
        # プロセスを開始
        process.start()
        print(f"プロセス開始: GPU {gpu_id}, 開始行: {start_row}, チャンクサイズ: {current_chunk_size}")
    
    # 結果の収集
    results = []
    for _ in range(len(processes)):
        result = result_queue.get()
        results.append(result)
    
    # 全プロセスの終了を待機
    for process in processes:
        process.join()
    
    # 結果を表示
    successful_files = []
    total_rows = 0
    total_time = 0
    
    for result in results:
        if result["success"]:
            successful_files.append(result["output_file"])
            total_rows += result["processed_rows"]
            total_time = max(total_time, result["elapsed_time"])
    
    print("\n=== 処理結果 ===")
    print(f"成功したGPU: {len(successful_files)}/{len(processes)}")
    print(f"処理された合計行数: {total_rows}")
    print(f"最大処理時間: {total_time:.2f}秒")
    
    if total_time > 0:
        print(f"総合スループット: {total_rows / total_time:.2f}行/秒")
    
    print(f"出力ファイル: {successful_files}")
    
    # 全てのParquetファイルを結合することも可能
    # 以下のコメントを解除すると結合処理が実行されます
    """
    if len(successful_files) > 0:
        print("\n全Parquetファイルを結合中...")
        combined_file = os.path.join(args.output, f"{args.table}_combined.parquet")
        
        # 各ファイルを読み込んでDataFrameのリストを作成
        dfs = []
        for file in successful_files:
            df = pd.read_parquet(file)
            dfs.append(df)
        
        # DataFrameを結合
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # 結合したDataFrameをParquetとして保存
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, combined_file)
            
            print(f"結合ファイルを保存しました: {combined_file} (合計 {len(combined_df)}行)")
    """

if __name__ == "__main__":
    main()
