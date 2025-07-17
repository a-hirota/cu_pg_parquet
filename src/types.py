"""PostgreSQL OID → Arrow 型マッピングと ColumnMeta 定義"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# Arrow 型 ID（内部列挙）
INT16, INT32, INT64 = 0, 1, 2
FLOAT32, FLOAT64 = 3, 4
DECIMAL128 = 5
UTF8, BINARY = 6, 7
TS64_S, TS64_US = 8, 9  # DATE32を削除し、TS64_Sを追加
BOOL = 10
UNKNOWN = 255

@dataclass(frozen=True)
class ColumnMeta:
    """PostgreSQL RowDescription の 1 カラム分を Arrow 変換に必要な情報にまとめた不変データクラス"""
    name: str
    pg_oid: int
    pg_typmod: int
    arrow_id: int
    elem_size: int
    arrow_param: Optional[Tuple[int, int]] = None

    @property
    def is_variable(self) -> bool:
        """可変長型かどうか"""
        return self.elem_size == 0

# PostgreSQL OID → Arrow 型 ID / 要素サイズ マッピング表
PG_OID_TO_ARROW: Dict[int, Tuple[int, Optional[int]]] = {
    20: (INT64, 8),       # int8 / bigint
    21: (INT16, 2),       # int2 / smallint
    23: (INT32, 4),       # int4 / integer
    700: (FLOAT32, 4),    # float4 / real
    701: (FLOAT64, 8),    # float8 / double precision
    1700: (DECIMAL128, 16), # numeric → Arrow Decimal128 (fixed 16 bytes)
    16:  (BOOL, 1),       # boolean
    25:  (UTF8, None),    # text
    1042: (UTF8, None),   # bpchar
    1043: (UTF8, None),   # varchar
    17:  (BINARY, None),  # bytea
    1082: (TS64_S, 8),    # date → timestamp seconds として扱う
    1114: (TS64_US, 8),   # timestamp without time zone
    1184: (TS64_US, 8),   # timestamp with time zone
}

# PostgreSQL OID → PostgreSQLバイナリでの実際のサイズ マッピング表
PG_OID_TO_BINARY_SIZE: Dict[int, Optional[int]] = {
    20: 8,        # int8 / bigint - 固定8バイト
    21: 2,        # int2 / smallint - 固定2バイト
    23: 4,        # int4 / integer - 固定4バイト
    700: 4,       # float4 / real - 固定4バイト
    701: 8,       # float8 / double precision - 固定8バイト
    16: 1,        # boolean - 固定1バイト
    1082: 4,      # date - PostgreSQLバイナリでは固定4バイト（2000-01-01からの日数）
    1114: 8,      # timestamp without time zone - 固定8バイト
    1184: 8,      # timestamp with time zone - 固定8バイト
    1700: None,   # numeric - 可変長
    25: None,     # text - 可変長
    1042: None,   # bpchar - 可変長
    1043: None,   # varchar - 可変長
    17: None,     # bytea - 可変長
}

__all__ = [
    "INT16", "INT32", "INT64", "FLOAT32", "FLOAT64", "DECIMAL128",
    "UTF8", "BINARY", "TS64_S", "TS64_US", "BOOL", "UNKNOWN",
    "ColumnMeta", "PG_OID_TO_ARROW", "PG_OID_TO_BINARY_SIZE",
]