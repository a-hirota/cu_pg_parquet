#!/usr/bin/env python
"""
単一チャンクのParquet出力テスト
"""

import os
import sys
import time
import psycopg2
import cudf
import pyarrow as pa
import pyarrow.parquet as pq

# カレントディレクトリをPYTHONPATHに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpupaser.main import PgGpuProcessor

def test_direct_parquet(output_file='./ray_output/customer_direct.parquet'):
    """直接SQL実行からcuDFでParquet出力"""
    print("\n=== 直接SQL実行からParquet出力テスト ===")
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 既存ファイルの削除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # データベース接続
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='postgres',
        host='localhost'
    )
    
    try:
        # SQL実行
        start_time = time.time()
        query = "SELECT * FROM customer LIMIT 10000"
        print(f"SQL実行: {query}")
        
        # cuDFでクエリ実行
        df = cudf.read_sql(query, conn)
        print(f"読み込み完了: {len(df)}行, {len(df.columns)}列")
        print(f"列: {df.columns.tolist()}")
        
        # Parquet出力
        df.to_parquet(output_file)
        processing_time = time.time() - start_time
        
        print(f"出力完了: {output_file}")
        print(f"処理時間: {processing_time:.3f}秒")
        
        # 検証
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"ファイルサイズ: {file_size:.2f} MB")
            
            # 読み込みテスト
            verify_df = cudf.read_parquet(output_file)
            print(f"検証: {len(verify_df)}行")
            print("\n最初の5行:")
            print(verify_df.head(5))
            
            return True
    except Exception as e:
        print(f"エラー: {e}")
        return False
    finally:
        conn.close()

def test_gpu_processor(output_file='./ray_output/customer_processor.parquet'):
    """PgGpuProcessor使用のParquet出力テスト"""
    print("\n=== PgGpuProcessor使用のParquet出力テスト ===")
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 既存ファイルの削除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    start_time = time.time()
    processor = PgGpuProcessor(parquet_output=output_file)
    
    try:
        print(f"処理開始: customerテーブル 10000行")
        result = processor.process_table("customer", 10000)
        
        processing_time = time.time() - start_time
        print(f"処理完了: {processing_time:.3f}秒")
        
        # 検証
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"ファイルサイズ: {file_size:.2f} MB")
            
            # 読み込みテスト
            verify_df = cudf.read_parquet(output_file)
            print(f"検証: {len(verify_df)}行")
            print("\n最初の5行:")
            print(verify_df.head(5))
            
            return True
        else:
            print(f"ファイルが生成されていません: {output_file}")
            return False
    except Exception as e:
        print(f"エラー: {e}")
        return False
    finally:
        processor.close()

def test_single_chunk(output_file='./ray_output/customer_chunk.parquet'):
    """単一チャンク処理テスト"""
    print("\n=== 単一チャンク処理テスト ===")
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 既存ファイルの削除
    if os.path.exists(output_file):
        os.remove(output_file)
    
    start_time = time.time()
    processor = PgGpuProcessor(parquet_output=output_file)
    
    try:
        chunk_size = 10000
        offset = 0
        print(f"チャンク処理: offset={offset}, size={chunk_size}")
        
        result = processor.process_table_chunk("customer", chunk_size, offset, output_file)
        
        processing_time = time.time() - start_time
        print(f"処理完了: {processing_time:.3f}秒")
        
        # 検証
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"ファイルサイズ: {file_size:.2f} MB")
            
            # 読み込みテスト
            verify_df = cudf.read_parquet(output_file)
            print(f"検証: {len(verify_df)}行")
            print("\n最初の5行:")
            print(verify_df.head(5))
            
            return True
        else:
            print(f"ファイルが生成されていません: {output_file}")
            return False
    except Exception as e:
        print(f"エラー: {e}")
        return False
    finally:
        processor.close()

if __name__ == "__main__":
    # 直接SQL実行からParquet出力テスト
    direct_success = test_direct_parquet()
    print(f"\n直接SQL実行テスト: {'成功' if direct_success else '失敗'}")
    
    # PgGpuProcessor使用のParquet出力テスト
    processor_success = test_gpu_processor()
    print(f"\nPgGpuProcessor使用テスト: {'成功' if processor_success else '失敗'}")
    
    # 単一チャンク処理テスト
    chunk_success = test_single_chunk()
    print(f"\n単一チャンク処理テスト: {'成功' if chunk_success else '失敗'}")
    
    # 全体結果
    print("\n=== テスト結果 ===")
    print(f"直接SQL実行テスト: {'成功' if direct_success else '失敗'}")
    print(f"PgGpuProcessor使用テスト: {'成功' if processor_success else '失敗'}")
    print(f"単一チャンク処理テスト: {'成功' if chunk_success else '失敗'}")
