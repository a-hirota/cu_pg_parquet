"""
gpupgparser - GPU PostgreSQL Parser (ZeroCopy版)

RAPIDS cuDF を活用した高速PostgreSQLバイナリデータ処理
ZeroCopy実装をデフォルトとして使用
"""

from .cuda_kernels.postgres_binary_parser import detect_pg_header_size

# 詳細制御用API
from .postgres_to_cudf import DirectColumnExtractor

# メインAPIの公開 (Direct版)
from .postgres_to_parquet_converter import DirectProcessor, convert_postgres_to_parquet_format

# 基盤コンポーネント
from .types import PG_OID_TO_ARROW, ColumnMeta
from .write_parquet_from_cudf import write_cudf_to_parquet, write_cudf_to_parquet_with_options

__version__ = "0.3.0"  # Direct版

__all__ = [
    # メイン処理
    "DirectProcessor",
    "convert_postgres_to_parquet_format",
    # 詳細制御
    "DirectColumnExtractor",
    "detect_pg_header_size",
    "write_cudf_to_parquet",
    "write_cudf_to_parquet_with_options",
    # 基盤
    "ColumnMeta",
    "PG_OID_TO_ARROW",
]
