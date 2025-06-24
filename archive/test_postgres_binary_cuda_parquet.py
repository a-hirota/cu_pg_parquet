"""
PostgreSQLバイナリデータのGPU解析とParquet出力のテスト
"""

import time
import os
import argparse
from gpupaser.main import PgGpuProcessor

def test_parquet_output(table_name, limit=None, output_path=None):
    """ParquetファイルとしてPostgreSQLデータを出力するテスト"""
    if output_path is None:
        output_path = f"{table_name}_output.parquet"
    
    print(f"テーブル {table_name} を処理し、結果を {output_path} に保存します")
    
    # 開始時間
    start_time = time.time()
    
    # gpuParserを使用してテーブルを処理
    processor = PgGpuProcessor(parquet_output=output_path)
    try:
        processor.process_table(table_name, limit)
    finally:
        processor.close()
    
    # 処理時間を表示
    total_time = time.time() - start_time
    print(f"\n処理時間: {total_time:.3f}秒")
    
    # Parquetファイルがあればcudfで読み込んでテスト
    if os.path.exists(output_path):
        try:
            import cudf
            print("\nParquetファイルの読み込みテスト:")
            df = cudf.read_parquet(output_path)
            
            print(f"\n行数: {len(df)}")
            
            print("\n最初の5行:")
            print(df.head(5))
            
            print("\n最後の5行:")
            print(df.tail(5))
            
            return True
        except ImportError:
            print("cuDFがインストールされていないため、読み込みテストをスキップします")
        except Exception as e:
            print(f"Parquetファイル読み込みテストエラー: {e}")
            
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PostgreSQL GPU Parser with Parquet Output')
    parser.add_argument('--table', required=True, help='Table name to process')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows')
    parser.add_argument('--output', help='Output Parquet file path')
    args = parser.parse_args()
    
    test_parquet_output(args.table, args.limit, args.output)
