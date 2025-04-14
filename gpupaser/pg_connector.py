"""
PostgreSQLへの接続とデータ取得モジュール
"""

import psycopg2
import io
from typing import List, Optional, Tuple

from .utils import ColumnInfo

class PostgresConnector:
    """PostgreSQLとの接続を管理するクラス"""
    
    def __init__(self, dbname='postgres', user='postgres', password='postgres', host='localhost'):
        """接続を初期化"""
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.conn = None
        self.connect()
        
    def connect(self):
        """PostgreSQL接続を確立"""
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host
            )
            return True
        except Exception as e:
            print(f"PostgreSQL接続エラー: {e}")
            return False
            
    def check_table_exists(self, table_name):
        """テーブルの存在チェック"""
        return check_table_exists(self.conn, table_name)
        
    def get_table_info(self, table_name):
        """テーブル情報の取得"""
        return get_table_info(self.conn, table_name)
        
    def get_table_row_count(self, table_name):
        """テーブルの行数を取得"""
        return get_table_row_count(self.conn, table_name)
        
    def get_binary_data(self, table_name, limit=None, offset=None, query=None):
        """テーブルのバイナリデータを取得"""
        return get_binary_data(self.conn, table_name, limit, offset, query)
        
    def close(self):
        """接続を閉じる"""
        if self.conn:
            self.conn.close()

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

def get_binary_data(conn, table_name: str, limit: Optional[int] = None, offset: Optional[int] = None, query: Optional[str] = None) -> Tuple[bytes, io.BytesIO]:
    """テーブルのバイナリデータを取得
    
    Args:
        conn: PostgreSQL接続
        table_name: テーブル名
        limit: 取得する最大行数
        offset: 取得開始位置（行オフセット）
        query: カスタムSQLクエリ（指定された場合は他のパラメータより優先）
    
    Returns:
        (bytes, BytesIO): バイナリデータとバッファ
    """
    cur = conn.cursor()
    
    # バイナリデータを一時的にメモリに保存
    buffer = io.BytesIO()
    
    if query is not None:
        # カスタムクエリが指定された場合はそれを使用
        sql_query = query
    else:
        # LIMITとOFFSETの設定
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        offset_clause = f"OFFSET {offset}" if offset is not None else ""
        sql_query = f"SELECT * FROM {table_name} {limit_clause} {offset_clause}"
    
    print(f"実行クエリ: {sql_query}")
    cur.copy_expert(f"COPY ({sql_query}) TO STDOUT WITH (FORMAT binary)", buffer)
    
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
