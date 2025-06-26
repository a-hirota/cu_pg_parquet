#!/usr/bin/env python
"""
lineorderテーブルでBINARY FORMATの検証を行うスクリプト
- 少ない行数でのテスト
- COPY構文の動作確認
- GPUデコードパイプラインの検証
"""

import os
import sys
import time
import numpy as np

# プロジェクトルートを追加
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Assuming src is importable directly now

from src.pg_connector import PostgresConnector, get_binary_data # Assuming get_binary_data is still in pg_connector
# from src.binary_parser import BinaryParser # TODO: Verify BinaryParser equivalent (e.g., gpu_parse_wrapper?)
from src.gpu_memory_manager_v2 import GPUMemoryManager # TODO: Verify class name and functionality match old GPUMemoryManager
from src.gpu_decoder_v2 import decode_chunk # TODO: Verify usage. Old GPUDecoder might be different from gpu_decoder_v2.decode_chunk
from src.output_handler import OutputHandler
from src.type_map import ColumnMeta # TODO: Verify if ColumnInfo should be replaced by ColumnMeta

def test_lineorder_binary_format(limit=5000, parquet_output="lineorder_binary_test.parquet"):
    """
    lineorderテーブルに対してCOPY BINARY構文のテストを実行
    
    Args:
        limit: 処理する行数
        parquet_output: 出力ファイルパス（Noneの場合は出力なし）
    """
    print(f"\n=== lineorderテーブルCOPY BINARY構文テスト (LIMIT {limit}) ===")
    
    # PostgreSQL接続
    pg_conn = PostgresConnector()
    
    # テーブル情報取得
    table_name = "lineorder"
    if not pg_conn.check_table_exists(table_name):
        print(f"テーブル {table_name} が存在しません")
        return False
    
    columns = pg_conn.get_table_info(table_name)
    print(f"列数: {len(columns)}")
    
    # 行数取得（大規模テーブルの場合はCOUNT(*)が遅いため注意）
    try:
        total_rows = pg_conn.get_table_row_count(table_name)
        print(f"テーブル合計行数: {total_rows}")
    except KeyboardInterrupt:
        print("行数カウントがキャンセルされました。処理を続行します。")
    
    # 開始時間
    start_time = time.time()
    
    # バイナリデータ取得（COPY BINARY構文利用）
    print(f"\n1. バイナリデータ取得（LIMIT {limit}）")
    data_start = time.time()
    
    # カスタムクエリの作成（LIMITを指定）
    custom_query = f"SELECT * FROM {table_name} LIMIT {limit}"
    
    # バイナリデータ取得
    buffer_data, buffer = pg_conn.get_binary_data(table_name, query=custom_query)
    
    data_time = time.time() - data_start
    print(f"バイナリデータ取得完了: {len(buffer_data)} バイト, {data_time:.2f}秒")
    
    # バイナリデータをダンプ（デバッグ用）
    debug_file = "lineorder_debug.bin"
    with open(debug_file, "wb") as f:
        f.write(buffer_data)
    print(f"デバッグファイル出力: {debug_file}")
    
    # バイナリパーサーの初期化
    parser = BinaryParser(use_gpu=True)
    memory_manager = GPUMemoryManager()
    gpu_decoder = GPUDecoder()
    
    # バイナリデータの解析
    print(f"\n2. GPUでのバイナリデータ解析")
    parse_start = time.time()
    
    # バイナリデータの行構造抽出
    chunk_array, field_offsets, field_lengths, rows_in_chunk = parser.parse_chunk(
        buffer_data, 
        max_chunk_size=len(buffer_data),
        num_columns=len(columns),
        start_row=0,
        max_rows=limit
    )
    
    parse_struct_time = time.time() - parse_start
    print(f"バイナリ構造解析完了: {rows_in_chunk}行, {parse_struct_time:.2f}秒")
    
    # GPUバッファの初期化
    buffers = memory_manager.initialize_device_buffers(columns, rows_in_chunk)
    
    # GPUでのデコード処理
    decode_start = time.time()
    try:
        # カラム情報の転送
        d_col_types = memory_manager.transfer_to_device(buffers["col_types"], np.int32)
        d_col_lengths = memory_manager.transfer_to_device(buffers["col_lengths"], np.int32)
        
        # GPUデコード実行
        result = gpu_decoder.decode_chunk(
            buffers,
            chunk_array,
            field_offsets,
            field_lengths,
            rows_in_chunk,
            columns
        )
        
        decode_time = time.time() - decode_start
        print(f"GPUデコード完了: {rows_in_chunk}行, {decode_time:.2f}秒")
        
        # 結果の表示
        if result:
            print("\n処理結果サマリー:")
            for col_name, data in result.items():
                if isinstance(data, np.ndarray):
                    print(f"  {col_name}: {type(data).__name__} [長さ={len(data)}]")
                else:
                    print(f"  {col_name}: {type(data).__name__}")
        
        # Parquet出力（指定された場合）
        if parquet_output:
            print(f"\n3. Parquet出力: {parquet_output}")
            output_start = time.time()
            
            output_handler = OutputHandler(parquet_output=parquet_output)
            output_handler.process_chunk_result(result)
            output_handler.close()
            
            output_time = time.time() - output_start
            print(f"Parquet出力完了: {output_time:.2f}秒")
            
            # 出力検証
            if os.path.exists(parquet_output):
                file_size = os.path.getsize(parquet_output) / (1024 * 1024)  # MB
                print(f"出力ファイルサイズ: {file_size:.2f} MB")
                
                try:
                    import cudf
                    verify_df = cudf.read_parquet(parquet_output)
                    print(f"検証: {len(verify_df)}行, {len(verify_df.columns)}列")
                    print("最初の5行:")
                    print(verify_df.head(5))
                except ImportError:
                    print("cuDFがインストールされていないため検証をスキップ")
                except Exception as e:
                    print(f"Parquet検証中にエラー: {e}")
        
        # 全体の処理時間
        total_time = time.time() - start_time
        print(f"\n=== 処理完了 ===")
        print(f"取得時間: {data_time:.2f}秒")
        print(f"構造解析時間: {parse_struct_time:.2f}秒")
        print(f"デコード時間: {decode_time:.2f}秒")
        print(f"全体処理時間: {total_time:.2f}秒")
        print(f"1秒あたり処理行数: {rows_in_chunk / total_time:.2f} rows/sec")
        
        return True
        
    except Exception as e:
        print(f"処理中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # GPUリソースのクリーンアップ
        cleanup_dict = {
            "d_col_types": locals().get("d_col_types"),
            "d_col_lengths": locals().get("d_col_lengths")
        }
        memory_manager.cleanup_buffers(cleanup_dict)
        memory_manager.cleanup_buffers(buffers)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='lineorderテーブルCOPY BINARY構文テスト')
    parser.add_argument('--limit', type=int, default=5000, help='取得する行数')
    parser.add_argument('--output', default="lineorder_binary_test.parquet", help='Parquet出力ファイル名')
    args = parser.parse_args()
    
    test_lineorder_binary_format(args.limit, args.output)
