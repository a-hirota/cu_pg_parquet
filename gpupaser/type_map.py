"""
PG OID → Arrow 型マッピングと ColumnMeta 定義モジュール

• ColumnMeta ... RowDescription 取得結果を保持する不変データクラス
• Arrow 型 ID を int 定数で提供（Numba / CUDA で扱いやすい形）
• PG_OID_TO_ARROW ... PostgreSQL の OID を Arrow 型 ID と要素バイト長へ変換する辞書
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# ----------------------------------------------------------------------
# Arrow 型 ID（内部列挙）
# 固定長型はそのままバイト長、可変長型は elem_size=None
# ----------------------------------------------------------------------
INT16, INT32, INT64 = 0, 1, 2
FLOAT32, FLOAT64 = 3, 4
DECIMAL128 = 5
UTF8, BINARY = 6, 7          # UTF8 = 文字列（可変長）, BINARY = バイト列
DATE32, TS64_US = 8, 9
BOOL = 10
UNKNOWN = 255                # 未対応 / フォールバック用

# ----------------------------------------------------------------------
# ColumnMeta
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnMeta:
    """
    PostgreSQL RowDescription の 1 カラム分を Arrow 変換に必要な
    情報にまとめた不変データクラス
    """
    name: str                     # 列名
    pg_oid: int                   # PostgreSQL 型 OID
    typmod: int                   # typmod（-1 の場合は 0 を入れる）
    arrow_id: int                 # 上記 Arrow 型 ID
    elem_size: int                # 固定長型の 1 要素バイト長（可変長型は 0）
    arrow_param: Optional[Tuple[int, int]] = None
    # 例) DECIMAL128 は (precision, scale)
    #    UTF8      は (max_length,) など用途により自由利用

    @property
    def is_variable(self) -> bool:
        """可変長型かどうか"""
        return self.elem_size == 0


# ----------------------------------------------------------------------
# PostgreSQL OID → Arrow 型 ID / 要素サイズ マッピング表
# 必要に応じて随時追加する
# ----------------------------------------------------------------------
PG_OID_TO_ARROW: Dict[int, Tuple[int, Optional[int]]] = {
    20: (INT64, 8),       # int8 / bigint
    21: (INT16, 2),       # int2 / smallint
    23: (INT32, 4),       # int4 / integer
    700: (FLOAT32, 4),    # float4 / real
    701: (FLOAT64, 8),    # float8 / double precision
    1700: (UTF8, None),  # numeric → UTF8 variable length (NUMERIC as text)
    16:  (BOOL, 1),       # boolean
    25:  (UTF8, None),    # text
    1042: (UTF8, None),   # bpchar
    1043: (UTF8, None),   # varchar
    17:  (BINARY, None),  # bytea
    1082: (DATE32, 4),    # date
    1114: (TS64_US, 8),   # timestamp without time zone
    1184: (TS64_US, 8),   # timestamp with time zone
}

__all__ = [
    "INT16", "INT32", "INT64", "FLOAT32", "FLOAT64", "DECIMAL128",
    "UTF8", "BINARY", "DATE32", "TS64_US", "BOOL", "UNKNOWN",
    "ColumnMeta", "PG_OID_TO_ARROW",
]
