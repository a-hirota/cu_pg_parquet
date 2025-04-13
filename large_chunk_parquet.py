"""
大規模データセット用Parquet出力スクリプト
"""

import time
import os
import numpy as np
import pandas as pd
import cudf
import pyarrow as pa
import pyarrow.parquet as pq
from gpupaser.pg_connector import connect_to_postgres, get_table_info

def process_large_table(table_name, limit=None, output_path=None, chunk_size=65000):
    """
    大規模テーブルをチャンク単位で処理し、Parquetファイルに出力する
    
    Args:
        table_name: 処理するテーブル名
        limit: 処理する最大行数（Noneの場合は全行）
        output_path: 出力Parquetファイルパス（Noneの場合は{table_name}_output.parquet）
        chunk_size: 1チャンクあたりの行数
        
    Returns:
        出力されたParquetファイルのパス
    """
    # 出力パスの設定
    if output_path is None:
        output_path = f"{table_name}_output.parquet"
    
    print(f"テーブル {table_name} を処理し、{output_path} に保存します")
    
    # 接続
    conn = connect_to_postgres()
    cursor = conn.cursor()
    
    # テーブル情報の取得
    columns = get_table_info(conn, table_name)
    column_names = [col.name for col in columns]
    print(f"カラム数: {len(columns)}")
    
    # テーブルの行数を取得
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]
    if limit:
        total_rows = min(total_rows, limit)
    print(f"テーブル行数: {total_rows}")
    
    # 処理開始時間
    start_time = time.time()
    
    # チャンク情報の初期化
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    has_written_once = False
    processed_rows = 0
    writer = None  # Parquet writer
    
    print(f"チャンクサイズ: {chunk_size}, チャンク数: {num_chunks}")
    
    for chunk_idx in range(num_chunks):
        # 現在のチャンクの開始行と行数を計算
        offset = chunk_idx * chunk_size
        current_chunk_size = min(chunk_size, total_rows - offset)
        
        print(f"\n== チャンク {chunk_idx+1}/{num_chunks}: {offset+1}～{offset+current_chunk_size}行 ==")
        
        # SQLクエリの作成（LIMIT とOFFSETで範囲指定）
        sql = f"SELECT * FROM {table_name} LIMIT {current_chunk_size} OFFSET {offset}"
        
        # データ取得の開始時間
        chunk_start_time = time.time()
        
        # クエリ実行
        cursor.execute(sql)
        
        # 結果をリストに格納
        rows = []
        for row in cursor:
            rows.append(row)
        
        # Pandas DataFrameに変換
        df = pd.DataFrame(rows, columns=column_names)
        print(f"pandas DataFrame作成: {len(df)}行")
        
        # cuDF DataFrameに変換
        gdf = cudf.DataFrame.from_pandas(df)
        print(f"cuDF DataFrame作成: {len(gdf)}行")
        
        # Parquetに書き込み（PyArrowを使用）
        # cuDFからPyArrow Tableに変換
        pa_table = gdf.to_arrow()
        
        if not has_written_once:
            # 最初のチャンクでは、ParquetWriterを初期化
            print(f"ParquetWriter初期化: {output_path}")
            writer = pq.ParquetWriter(output_path, pa_table.schema)
            writer.write_table(pa_table)
            has_written_once = True
        else:
            # 2回目以降は既存のwriterにテーブルを追加
            # ParquetWriterが閉じられている場合は再度開く
            try:
                writer.write_table(pa_table)
            except Exception as e:
                print(f"既存のWriterで書き込みに失敗: {e}")
                print("新しいWriterを作成して追記モードで書き込みます")
                # 一時ファイルに書き込み
                temp_path = f"{output_path}.temp"
                pq.write_table(pa_table, temp_path)
                
                # PyArrowのDatasetを使用して結合
                from pyarrow.dataset import dataset
                pq_dataset = dataset([output_path, temp_path], format="parquet")
                combined_table = pq_dataset.to_table()
                
                # 結合したテーブルを書き込み
                pq.write_table(combined_table, output_path)
                
                # 一時ファイルを削除
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        # チャンク処理時間
        chunk_time = time.time() - chunk_start_time
        processed_rows += len(df)
        print(f"チャンク処理時間: {chunk_time:.3f}秒")
        print(f"累計処理行数: {processed_rows}/{total_rows}")
    
    # Parquet writer と接続を閉じる
    if writer:
        try:
            writer.close()
            print(f"ParquetWriter closed")
        except Exception as e:
            print(f"ParquetWriter close error: {e}")
    
    cursor.close()
    conn.close()
    
    # 総処理時間
    total_time = time.time() - start_time
    print(f"\n処理完了: {processed_rows}行を {total_time:.3f}秒で処理")
    
    # Parquet検証（PyArrowを使用）
    print("\nParquetファイル検証:")
    try:
        # PyArrowでファイルを読み込む
        pa_table = pq.read_table(output_path)
        print(f"行数（PyArrow）: {pa_table.num_rows}")
        
        # cuDFを使用する場合は、単一のrow groupとして読み込むため、
        # PyArrowテーブルを一度メモリ内で変換する
        print("cuDFでの読み込みテスト:")
        temp_path = f"{output_path}.verify.parquet"
        pq.write_table(pa_table, temp_path)
        
        verify_df = cudf.read_parquet(temp_path)
        print(f"行数（cuDF）: {len(verify_df)}")
        
        print("\n最初の5行:")
        print(verify_df.head(5))
        
        print("\n最後の5行:")
        print(verify_df.tail(5))
        
        # 一時ファイルを削除
        try:
            os.remove(temp_path)
        except:
            pass
            
    except Exception as e:
        print(f"検証中にエラー: {e}")
        # 代わりにPyArrowでの簡易検証を行う
        try:
            pa_table = pq.read_table(output_path)
            print(f"PyArrow検証: {pa_table.num_rows}行")
            
            # 最初と最後の数行を表示
            print("\n最初の5行（PyArrow）:")
            print(pa_table.slice(0, 5).to_pandas())
            
            print("\n最後の5行（PyArrow）:")
            print(pa_table.slice(pa_table.num_rows - 5, 5).to_pandas())
        except Exception as inner_e:
            print(f"PyArrow検証でもエラー: {inner_e}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process large PostgreSQL table to Parquet')
    parser.add_argument('--table', default='customer', help='Table name to process')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows')
    parser.add_argument('--output', default=None, help='Output Parquet file path')
    parser.add_argument('--chunk_size', type=int, default=65000, help='Chunk size in rows')
    args = parser.parse_args()
    
    process_large_table(args.table, args.limit, args.output, args.chunk_size)
