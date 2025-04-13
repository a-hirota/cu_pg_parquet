"""
PostgreSQLへの接続とデータ取得モジュール
"""

import psycopg2
import io
from typing import List, Optional, Tuple

from .utils import ColumnInfo

def connect_to_postgres(dbname='postgres', user='postgres', password='postgres', host='localhost'):
    """PostgreSQLへの接続を確立する"""
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host
    )
    return conn

def check_table_exists(conn, table_name: str) -> bool:
    """テーブルの存在確認"""
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        )
    """, (table_name,))
    exists = cur.fetchone()[0]
    cur.close()
    return exists

def get_table_info(conn, table_name: str) -> List[ColumnInfo]:
    """テーブル情報の取得"""
    cur = conn.cursor()
    cur.execute("""
        SELECT column_name, data_type, 
               CASE WHEN character_maximum_length IS NOT NULL 
                    THEN character_maximum_length 
                    ELSE NULL 
               END as max_length
        FROM information_schema.columns 
        WHERE table_name = %s 
        ORDER BY ordinal_position
    """, (table_name,))
    
    columns = []
    for name, type_, length in cur.fetchall():
        print(f"Column: {name}, Type: {type_}, Length: {length}")  # デバッグ出力
        columns.append(ColumnInfo(name, type_, length))
    
    cur.close()
    return columns

def get_table_row_count(conn, table_name: str) -> int:
    """テーブルの行数取得"""
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cur.fetchone()[0]
    cur.close()
    print(f"Table {table_name} has {row_count} rows")  # デバッグ出力
    return row_count

def get_binary_data(conn, table_name: str, limit: Optional[int] = None) -> Tuple[bytes, io.BytesIO]:
    """テーブルのバイナリデータを取得"""
    cur = conn.cursor()
    
    # バイナリデータを一時的にメモリに保存
    buffer = io.BytesIO()
    limit_clause = f"LIMIT {limit}" if limit is not None else ""
    cur.copy_expert(f"COPY (SELECT * FROM {table_name} {limit_clause}) TO STDOUT WITH (FORMAT binary)", buffer)
    
    # バッファをメモリに固定
    buffer_data = buffer.getvalue()
    
    # バイナリファイルに書き込む
    with open('output_debug.bin', 'wb') as f:
        f.write(buffer_data)
    
    print("バイナリファイルに書き込みました。")
    
    # バッファをリセットして読み取り用に準備
    buffer = io.BytesIO(buffer_data)
    
    # バッファサイズの確認
    total_size = buffer.getbuffer().nbytes
    print(f"Total binary data size: {total_size} bytes")
    
    return buffer_data, buffer
