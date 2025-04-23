"""
RowDescription から Arrow 変換用メタデータ (ColumnMeta) を生成するモジュール

使い方例:
    import psycopg2
    from meta_fetch import fetch_column_meta

    conn = psycopg2.connect("dbname=postgres user=postgres")
    metas = fetch_column_meta(conn, "SELECT * FROM lineorder LIMIT 100")
    for m in metas:
        print(m)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Any, Protocol

# psycopg2/3 動的インポート
try:
    import psycopg
    CursorBase = psycopg.Cursor
except ImportError:
    import psycopg2
    from psycopg2.extensions import cursor as CursorBase

from .type_map import (
    ColumnMeta,
    PG_OID_TO_ARROW,
    DECIMAL128,
    UTF8,
    UNKNOWN,
)


def _decode_numeric_typmod(typmod: int) -> Tuple[int, int]:
    """
    PostgreSQL numeric(p,s) typmod 値から (precision, scale) を返すヘルパ
    typmod は ( (p << 16) | s ) + 4 という内部表現
    """
    if typmod <= 0:  # typmod==0 は未指定 (= variable)
        return (0, 0)
    mod = typmod - 4
    precision = (mod >> 16) & 0xFFFF
    scale = mod & 0xFFFF
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
    """
    任意の SELECT クエリに対し ``SELECT * FROM (sql) AS t LIMIT 0`` を発行し、
    RowDescription のみを取得して ColumnMeta のリストを返す。

    Notes
    -----
    * psycopg2 の `cursor.description` は sequence のようなタプルで、
      - name            : 列名
      - type_code       : OID
      - internal_size   : typmod または fixed size
      などを含む。
    * ``internal_size < 0`` の場合は typmod が -internal_size として渡される。
      詳細: https://www.psycopg.org/docs/cursor.html#cursor.description
    """
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM ({sql}) AS __t LIMIT 0")
    metas: List[ColumnMeta] = []

    for desc in cur.description:
        name = desc.name
        pg_oid: int = desc.type_code
        typmod = desc.internal_size or 0  # None の場合は 0

        # psycopg2/3 の仕様:
        #   可変長型           typmod は -1 (固定サイズ不明) または -N (実質 typmod=N)
        #   固定長型 (int4)    typmod は 4 のように固定バイト長正値
        if typmod < 0:
            typmod = -typmod
        # typmod==0 or typmod==4 等もあり得る

        # OID → Arrow 型 ID と要素サイズ
        arrow_id, elem = PG_OID_TO_ARROW.get(pg_oid, (UNKNOWN, None))
        elem_size = elem or 0  # 可変長型は 0 で扱う

        arrow_param: Optional[Tuple[int, int] | int] = None

        # typmod による補正
        if arrow_id == DECIMAL128:
            # numeric(p,s) → (precision, scale)
            precision, scale = _decode_numeric_typmod(typmod)
            arrow_param = (precision, scale)
        elif arrow_id == UTF8:
            # VARCHAR(N) の N = typmod-4
            if typmod > 4:
                arrow_param = typmod - 4

        metas.append(
            ColumnMeta(
                name=name,
                pg_oid=pg_oid,
                typmod=typmod,
                arrow_id=arrow_id,
                elem_size=elem_size,
                arrow_param=arrow_param,
            )
        )

    cur.close()
    return metas


__all__ = ["fetch_column_meta"]
