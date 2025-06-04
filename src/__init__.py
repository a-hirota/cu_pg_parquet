"""
gpupgparser - GPU PostgreSQL Parser
"""

from .metadata import fetch_column_meta
from .column_processor import decode_chunk_integrated
from .binary_parser import parse_binary_chunk_gpu, detect_pg_header_size
from .types import ColumnMeta, PG_OID_TO_ARROW

__version__ = "0.1.0"

__all__ = [
    "fetch_column_meta",
    "decode_chunk_integrated",
    "parse_binary_chunk_gpu",
    "detect_pg_header_size",
    "ColumnMeta",
    "PG_OID_TO_ARROW",
]
