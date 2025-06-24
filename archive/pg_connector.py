"""
PostgreSQLへの接続とデータ取得モジュール
"""

import psycopg # Use only psycopg (v3)
import io
import os
from typing import List, Optional, Tuple

# from .utils import ColumnInfo # Removed incorrect import
from .meta_fetch import fetch_column_meta, ColumnMeta # Import ColumnMeta from meta_fetch
# from .type_map import ColumnMeta # Removed import from type_map
# from .psql_copy_stream import copy_binary_to_gpu_chunks # Commented out non-existent module import

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
            # Construct DSN string for psycopg (v3)
            dsn = f"dbname='{self.dbname}' user='{self.user}' password='{self.password}' host='{self.host}'"
            self.conn = psycopg.connect(dsn)
            return True
        except Exception as e:
            print(f"PostgreSQL接続エラー: {e}")
            return False

    def check_table_exists(self, table_name):
        """テーブルの存在チェック"""
        # This function is defined globally, ensure it's compatible or update it
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

    def copy_to_gpu(self,
        query: str,
        process_chunk,
        chunk_bytes: int = 32 << 20
    ):
        """
        COPY BINARY を psycopg3 + pinned buffer でチャンク受信し GPU へ転送
        Parameters
        ----------
        query : str
            SELECT クエリ文字列
        process_chunk : callable
            (gpu_dev_array, nbytes) を受け取るコールバック
        chunk_bytes : int
            1チャンクのバイト数
        """
        # DSN文字列を再構築
        dsn = f"dbname={self.dbname} user={self.user} password={self.password} host={self.host}"
        copy_binary_to_gpu_chunks(dsn, query, chunk_bytes, process_chunk)

def connect_to_postgres(dbname='postgres', user='postgres', password='postgres', host='localhost'):
    """PostgreSQLへの接続を確立する"""
    dsn = f"dbname='{dbname}' user='{user}' password='{password}' host='{host}'"
    conn = psycopg.connect(dsn)
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

def get_table_info(conn, table_name: str) -> List[ColumnMeta]: # Changed ColumnInfo to ColumnMeta
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
        # Assuming ColumnMeta constructor matches (name, type_, length) or similar
        # Need to verify ColumnMeta definition if this fails
        columns.append(ColumnMeta(name, type_, length)) # Changed ColumnInfo to ColumnMeta

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

def get_query_column_info(conn, query: str) -> List[ColumnMeta]: # Changed ColumnInfo to ColumnMeta
    """SQLクエリの結果セットのカラム情報を取得

    Args:
        conn: PostgreSQL接続オブジェクト
        query: 実行するSQLクエリ
    
    Returns:
        カラム情報のリスト (ColumnInfoオブジェクト)
    """
    try:
        # カラム情報を取得するためのサブクエリを作成
        # LIMIT 0を使って結果を返さずにメタデータだけを取得
        metadata_query = f"""
        SELECT * FROM ({query}) AS query_result LIMIT 0
        """
        
        cur = conn.cursor()
        cur.execute(metadata_query)
        
        # カラム情報を取得
        columns = []
        for col in cur.description:
            col_name = col.name
            # PostgreSQLの型OIDをPythonの型に変換
            # 参考: https://www.postgresql.org/docs/current/datatype-oid.html
            col_type_oid = col.type_code
            
            # 一般的なPostgreSQL型OIDを確認
            # 23: int4 (integer), 21: int2 (smallint), 20: int8 (bigint)
            # 1700: numeric, 25: text, 1043: varchar
            if col_type_oid in (20, 21, 23):  # 整数型
                col_type = "integer"
                col_length = 4
            elif col_type_oid == 1700:  # numeric
                col_type = "numeric"
                col_length = 8
            elif col_type_oid in (25, 1043):  # 文字列型
                col_type = "character varying"
                col_length = 256  # デフォルト長さ
            else:
                # その他の型は文字列として扱う
                col_type = "text"
                col_length = 1024

            # Assuming ColumnMeta constructor matches (name, type_, length) or similar
            # Need to verify ColumnMeta definition if this fails
            columns.append(ColumnMeta(col_name, col_type, col_length)) # Changed ColumnInfo to ColumnMeta

        # 詳細なログ出力
        print(f"クエリのカラム情報を取得: {len(columns)}カラム")
        for col in columns:
            print(f"  - {col.name}: {col.type}" + (f" (長さ: {col.length})" if col.length else ""))
            
        return columns
        
    except Exception as e:
        print(f"クエリのカラム情報取得中にエラー: {e}")
        # エラーの場合は空のリストを返す
        return []

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
    # Use cursor.copy() for psycopg (v3)
    with cur.copy(f"COPY ({sql_query}) TO STDOUT (FORMAT BINARY)") as copy:
        for data_chunk in copy: # Iterate over data chunks from the COPY operation
            buffer.write(data_chunk)
    
    # バッファをメモリに固定
    buffer_data = buffer.getvalue()
    
    # デバッグ設定に基づいてファイル出力（環境変数で制御）
    debug_files = os.environ.get('GPUPASER_DEBUG_FILES', '1') == '1'
    
    if debug_files:
        # バイナリファイルに書き込む
        with open('output_debug.bin', 'wb') as f:
            f.write(buffer_data)
        print("デバッグ用バイナリファイルに書き込みました。")
    else:
        print("デバッグファイル出力はスキップされました。")
    
    # バッファをリセットして読み取り用に準備
    buffer = io.BytesIO(buffer_data)
    
    # バッファサイズの確認
    total_size = buffer.getbuffer().nbytes
    print(f"Total binary data size: {total_size} bytes")
    
    return buffer_data, buffer


# ----------------------------------------------------------------------
# Arrow ColumnMeta ベースでカラムメタデータを取得する新関数
# ----------------------------------------------------------------------
def get_query_column_meta(conn, query: str) -> List[ColumnMeta]:
    """SQLクエリの RowDescription を利用して ColumnMeta を返す"""
    return fetch_column_meta(conn, query)
