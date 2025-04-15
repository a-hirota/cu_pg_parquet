"""
lineorderテーブルのパイプライン処理によるParquet出力
"""

import os
import sys
import time
import numpy as np
import cudf
import psycopg2
import argparse
from typing import Dict, List, Any
import concurrent.futures
from numba import cuda

# プロジェクトルートを追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpupaser.pg_connector import PostgresConnector
from gpupaser.binary_parser import PipelinedProcessor

def get_gpu_count():
    """利用可能なGPU数を取得"""
    try:
        count = len(cuda.gpus)
        print(f"利用可能なGPU数: {count}")
        return count
    except Exception as e:
        print(f"GPU情報の取得に失敗: {e}")
        return 1  # デフォルトは1

def test_lineorder_pipeline_small():
    """lineorderテーブルの小規模パイプライン処理テスト"""
    print("\n=== 少量のlineorderデータでパイプラインテスト ===")
    
    # 出力ディレクトリ設定
    output_dir = "test/lineorder_pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # PostgreSQL接続情報
    db_params = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost"
    }
    
    # パイプラインプロセッサの初期化
    processor = PipelinedProcessor(
        table_name="lineorder",
        db_params=db_params,
        output_dir=output_dir,
        chunk_size=50,  # 少量テストなので小さく設定
        gpu_count=get_gpu_count()
    )
    
    # 最大10行で処理を実行
    result = processor.run(max_rows=10)
    
    # 結果の表示
    print("\n=== パイプライン処理結果 ===")
    print(f"処理行数: {result['processed_rows']}")
    print(f"処理チャンク数: {result['processed_chunks']}")
    print(f"処理時間: {result['processing_time']:.2f}秒")
    print(f"処理速度: {result['throughput']:.2f} rows/sec")
    
    # 結果の結合
    combine_result = processor.combine_outputs()
    if combine_result:
        print(f"結合ファイル: {combine_result['output_path']}")
        print(f"結合行数: {combine_result['row_count']}")
        
        # 結果の確認
        df = cudf.read_parquet(combine_result['output_path'])
        print("\n=== Parquetファイル内容 ===")
        print(f"行数: {len(df)}")
        print(f"列名: {df.columns.values}")
        print("\n最初の数行:")
        print(df.head(5))
        
        # データ型の表示
        print("\nカラムデータ型:")
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")
        
        return True
    else:
        print("結合に失敗しました")
        return False

def test_lineorder_pipeline_performance():
    """lineorderテーブルの大規模パイプライン処理テスト"""
    print("\n=== lineorderテーブルの大規模パイプラインテスト ===")
    
    # 出力ディレクトリ設定
    output_dir = "test/lineorder_pipeline_perf"
    os.makedirs(output_dir, exist_ok=True)
    
    # PostgreSQL接続情報
    db_params = {
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost"
    }
    
    # 行数を取得
    pg_conn = PostgresConnector(**db_params)
    table_name = "lineorder"
    total_rows = pg_conn.get_table_row_count(table_name)
    
    # GPU数を取得
    gpu_count = get_gpu_count()
    
    # 利用可能なGPUメモリに基づくチャンクサイズ設定
    chunk_size = 1_000_000  # デフォルト値
    
    try:
        # GPU情報の取得
        context = cuda.current_context()
        free_memory = context.get_memory_info()[0]
        # 1行あたりの推定メモリ使用量
        bytes_per_row = 1000  # 概算：1KB/行
        # GPUメモリの70%までを使用
        max_rows_by_memory = int(0.7 * free_memory / bytes_per_row)
        # 安全側に立った上限値設定
        chunk_size = min(chunk_size, max_rows_by_memory)
        print(f"GPUメモリに基づく調整後のチャンクサイズ: {chunk_size}行")
    except Exception as e:
        print(f"GPUメモリ情報取得エラー（デフォルト値を使用）: {e}")
    
    # 処理行数の設定（最大600万行）
    max_rows = min(6_000_000, total_rows)
    print(f"処理対象行数: {max_rows} / {total_rows}")
    
    # パイプラインプロセッサの初期化
    processor = PipelinedProcessor(
        table_name=table_name,
        db_params=db_params,
        output_dir=output_dir,
        chunk_size=chunk_size,
        gpu_count=gpu_count
    )
    
    # 処理の実行
    result = processor.run(max_rows=max_rows)
    
    # 結果の表示
    print("\n=== パイプライン処理結果 ===")
    print(f"処理行数: {result['processed_rows']} / {max_rows}")
    print(f"処理チャンク数: {result['processed_chunks']} / {result['total_chunks']}")
    print(f"処理時間: {result['processing_time']:.2f}秒")
    print(f"処理速度: {result['throughput']:.2f} rows/sec")
    
    # 結果の結合
    combine_result = processor.combine_outputs()
    if combine_result:
        print(f"結合ファイル: {combine_result['output_path']}")
        print(f"結合行数: {combine_result['row_count']}")
        return True
    else:
        print("結合に失敗しました")
        return False

if __name__ == "__main__":
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='lineorderテーブルのパイプライン処理テスト')
    parser.add_argument('--small', action='store_true', help='少量データでテスト')
    parser.add_argument('--perf', action='store_true', help='大規模データでパフォーマンステスト')
    args = parser.parse_args()
    
    # 引数に応じてテスト実行
    if args.small:
        test_lineorder_pipeline_small()
    elif args.perf:
        test_lineorder_pipeline_performance()
    else:
        # デフォルトは両方実行
        print("少量データテストを実行...")
        test_lineorder_pipeline_small()
        
        print("\n大規模データテストを実行...")
        test_lineorder_pipeline_performance()
