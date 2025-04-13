"""
シンプルなテスト用スクリプト - PostgreSQLからcuDFへの直接取り込みとParquet出力
"""

import time
import os
import cudf
import numpy as np
from gpupaser.pg_connector import connect_to_postgres, get_table_info, get_binary_data

def test_simple_parquet_output(table_name='customer', limit=1000, output_path='simple_output.parquet'):
    """
    シンプル化されたテスト - PostgreSQLからデータを取得してParquetとして保存
    """
    print(f"シンプル化されたテスト: {table_name}テーブルをParquetとして保存")
    
    # 接続とデータ取得
    print("接続中...")
    conn = connect_to_postgres()
    
    # テーブル情報取得
    columns = get_table_info(conn, table_name)
    print(f"カラム数: {len(columns)}")
    
    # バイナリデータ取得
    print(f"{limit}行のデータを取得中...")
    buffer_data, buffer = get_binary_data(conn, table_name, limit)
    print(f"バイナリデータサイズ: {len(buffer_data)} バイト")
    
    # 処理開始時間
    start_time = time.time()
    
    # 単純にpandasでデータを読み込み、cuDFに変換
    print("pandasでデータを読み込み中...")
    import pandas as pd
    import io
    import re
    
    # SQLを実行して結果を取得
    sql = f"SELECT * FROM {table_name}"
    if limit:
        sql += f" LIMIT {limit}"
    
    print(f"SQL実行: {sql}")
    
    # メモリバッファを準備
    buffer = io.StringIO()
    
    # COPY TO を使用してCSV形式でデータを取得
    cur = conn.cursor()
    cur.execute(sql)
    
    # 手動でCSVフォーマットに変換
    column_names = [desc[0] for desc in cur.description]
    
    # ヘッダー行を書き込み
    buffer.write(','.join(column_names) + '\n')
    
    # データ行を書き込み
    row_count = 0
    for row in cur:
        # カンマ区切りで値を連結、文字列にはクォーテーションを付ける
        csv_row = []
        for val in row:
            if val is None:
                csv_row.append('')
            elif isinstance(val, (int, float)):
                csv_row.append(str(val))
            else:
                # エスケープ処理（カンマやダブルクォートを含む場合）
                val_str = str(val).replace('"', '""')
                csv_row.append(f'"{val_str}"')
        
        buffer.write(','.join(csv_row) + '\n')
        row_count += 1
    
    print(f"{row_count}行のデータを取得")
    buffer.seek(0)
    
    # pandasで読み込み
    df = pd.read_csv(buffer)
    print(f"pandas DataFrame作成: {len(df)}行")
    
    # cuDFに変換
    gdf = cudf.DataFrame.from_pandas(df)
    print(f"cuDF DataFrame作成: {len(gdf)}行")
    
    # Parquetとして保存
    gdf.to_parquet(output_path)
    print(f"Parquetファイルに保存: {output_path}")
    
    # 処理時間
    elapsed_time = time.time() - start_time
    print(f"処理時間: {elapsed_time:.3f}秒")
    
    # 読み込みテスト
    verify_df = cudf.read_parquet(output_path)
    print(f"\nParquetファイル検証: {len(verify_df)}行")
    print("\n最初の5行:")
    print(verify_df.head(5))
    print("\n最後の5行:")
    print(verify_df.tail(5))
    
    # 接続解放
    conn.close()
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple PostgreSQL to Parquet test')
    parser.add_argument('--table', default='customer', help='Table name to process')
    parser.add_argument('--limit', type=int, default=2000, help='Limit number of rows')
    parser.add_argument('--output', default='test_simple_output.parquet', help='Output Parquet file path')
    args = parser.parse_args()
    
    test_simple_parquet_output(args.table, args.limit, args.output)
