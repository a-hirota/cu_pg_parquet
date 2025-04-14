#!/usr/bin/env python
"""
GPUパーサーのテスト用スクリプト
"""

import numpy as np
import time
import sys
from pathlib import Path
import psycopg2
from gpupaser.gpu_decoder import GPUDecoder

# GPUパーサーのデバッグテスト
def test_gpu_parser(table="customer", limit=1000):
    """GPUパーサーの基本機能テスト"""
    print(f"GPUパーサーテスト: {table}テーブル {limit}行")
    
    # PostgreSQLに接続
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost"
    )
    
    cursor = conn.cursor()
    
    # テーブルのカラム情報を取得
    cursor.execute(f"""
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position
    """)
    
    columns = cursor.fetchall()
    print(f"テーブル {table} のカラム情報:")
    for col in columns:
        print(f"  {col[0]}: {col[1]}" + (f" (長さ: {col[2]})" if col[2] else ""))
    
    num_columns = len(columns)
    print(f"カラム数: {num_columns}")
    
    # PostgreSQLからバイナリデータを取得
    import io
    buffer = io.BytesIO()
    cursor.copy_expert(f"COPY (SELECT * FROM {table} LIMIT {limit}) TO STDOUT WITH (FORMAT BINARY)", buffer)
    buffer.seek(0)
    binary_data = buffer.read()
    
    # ファイルに保存（デバッグ用）
    with open("gpu_parser_test.bin", "wb") as f:
        f.write(binary_data)
    
    print(f"取得したバイナリデータ: {len(binary_data)} バイト")
    
    # バイト配列に変換
    chunk_array = np.frombuffer(binary_data, dtype=np.uint8)
    
    # GPUパーサーのインスタンス化
    gpu_decoder = GPUDecoder()
    
    # 処理タイミングの計測
    start_time = time.time()
    
    # 異なる予測行数でテスト
    estimated_rows = [limit // 10, limit // 2, limit, limit * 2]
    
    for est_rows in estimated_rows:
        print(f"\n--- 予測行数: {est_rows} ---")
        try:
            # GPUパーサーでパース
            field_offsets, field_lengths, rows_in_chunk = gpu_decoder.parse_binary_data(
                chunk_array, est_rows, num_columns
            )
            
            if rows_in_chunk > 0:
                # 最初の行だけ検証
                first_row_offsets = field_offsets[:num_columns]
                first_row_lengths = field_lengths[:num_columns]
                
                print(f"パース結果: {rows_in_chunk}行")
                print(f"最初の行: {num_columns}フィールド")
                
                # 最初の行のフィールドを表示
                for i in range(num_columns):
                    offset = first_row_offsets[i]
                    length = first_row_lengths[i]
                    
                    if length == -1:
                        value = "NULL"
                    elif offset > 0 and offset + length <= len(chunk_array):
                        # フィールドの値を取得
                        if length > 100:
                            # 大きすぎる値は切り詰める
                            value = f"<バイナリデータ {length}バイト>"
                        else:
                            try:
                                # 文字列として表示してみる（バイナリなので正確ではない）
                                value = bytes(chunk_array[offset:offset+length]).decode('utf-8', errors='replace')
                                if len(value) > 30:
                                    value = value[:30] + "..."
                            except:
                                value = f"<バイナリデータ>"
                    else:
                        value = f"<無効なオフセット/長さ: {offset}/{length}>"
                    
                    print(f"  フィールド {i}: オフセット={offset}, 長さ={length}, 値={value}")
            else:
                print(f"パース失敗: 0行")
            
        except Exception as e:
            print(f"エラー: {e}")
    
    elapsed = time.time() - start_time
    print(f"\n処理時間: {elapsed:.3f}秒")
    
    # 終了
    conn.close()
    print("テスト完了")

if __name__ == "__main__":
    # 引数処理
    limit = 1000
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except:
            pass
    
    test_gpu_parser(limit=limit)
