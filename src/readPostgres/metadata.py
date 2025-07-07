"""RowDescription → Arrow ColumnMeta 変換"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any, Protocol

# psycopg2/3 動的インポート
try:
    import psycopg
    CursorBase = psycopg.Cursor
except ImportError:
    import psycopg2
    from psycopg2.extensions import cursor as CursorBase

from ..types import (
    ColumnMeta, PG_OID_TO_ARROW, DECIMAL128, UTF8, UNKNOWN,
)

def _decode_numeric_pg_typmod(pg_typmod: int) -> Tuple[int, int]:
    """PostgreSQL numeric(p,s) pg_typmod 値から (precision, scale) を返す"""
    if pg_typmod <= 0:
        # 精度が指定されていない場合は最大精度を使用
        return (38, 0)
    mod = pg_typmod - 4
    precision = (mod >> 16) & 0xFFFF
    scale = mod & 0xFFFF
    # 無効な値の場合はデフォルトに戻す
    if precision == 0:
        precision = 38
    return precision, scale

class CursorDescription(Protocol):
    """psycopg2/3 cursor.description[i] 互換プロトコル"""
    @property
    def name(self) -> str: ...
    @property
    def type_code(self) -> int: ...
    @property
    def internal_size(self) -> Optional[int]: ...

def fetch_column_meta(conn: Any, sql: str) -> List[ColumnMeta]:
    """任意の SELECT クエリに対し RowDescription のみを取得して ColumnMeta のリストを返す"""
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM ({sql}) AS __t LIMIT 0")
    metas: List[ColumnMeta] = []

    for desc in cur.description:
        name = desc.name
        pg_oid: int = desc.type_code
        # まずinternal_sizeを試す（numericの場合は有効な値が入っている可能性）
        pg_typmod = desc.internal_size or 0

        if pg_typmod < 0:
            pg_typmod = -pg_typmod

        arrow_id, elem = PG_OID_TO_ARROW.get(pg_oid, (UNKNOWN, None))
        elem_size = elem or 0

        arrow_param: Optional[Tuple[int, int] | int] = None

        if arrow_id == DECIMAL128:
            precision, scale = _decode_numeric_pg_typmod(pg_typmod)
            arrow_param = (precision, scale)
        elif arrow_id == UTF8:
            # bpchar/varcharの場合、display_sizeが実際の文字長
            if desc.display_size and desc.display_size > 0:
                arrow_param = desc.display_size

        metas.append(
            ColumnMeta(
                name=name,
                pg_oid=pg_oid,
                pg_typmod=pg_typmod,
                arrow_id=arrow_id,
                elem_size=elem_size,
                arrow_param=arrow_param,
            )
        )

    cur.close()
    return metas

def get_postgresql_table_metadata(conn, table_name: str) -> List[ColumnMeta]:
    """PostgreSQLテーブルのメタデータを取得（旧名互換）"""
    return fetch_column_meta(conn, f"SELECT * FROM {table_name}")

def get_table_metadata(conn, table_name: str) -> List[ColumnMeta]:
    """PostgreSQLテーブルのメタデータを取得"""
    return fetch_column_meta(conn, f"SELECT * FROM {table_name}")

__all__ = ["fetch_column_meta", "get_postgresql_table_metadata", "get_table_metadata"]