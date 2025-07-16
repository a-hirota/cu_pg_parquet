"""
gpupgparser - GPU PostgreSQL Parser (ZeroCopy版)

RAPIDS cuDF を活用した高速PostgreSQLバイナリデータ処理
ZeroCopy実装をデフォルトとして使用
"""

# メインAPIの公開 (Direct版)
from .main_postgres_to_parquet import (
    DirectProcessor,
    postgresql_to_cudf_parquet_direct
)

# 詳細制御用API
from .postgres_to_cudf import DirectColumnExtractor
from .cuda_kernels.postgres_binary_parser import (
    detect_pg_header_size
)
from .write_parquet_from_cudf import (
    write_cudf_to_parquet,
    write_cudf_to_parquet_with_options
)

# 基盤コンポーネント
from .types import ColumnMeta, PG_OID_TO_ARROW

__version__ = "0.3.0"  # Direct版

__all__ = [
    # メイン処理
    "DirectProcessor",
    "postgresql_to_cudf_parquet_direct",
    
    # 詳細制御
    "DirectColumnExtractor",
    "detect_pg_header_size",
    "write_cudf_to_parquet",
    "write_cudf_to_parquet_with_options",
    
    # 基盤
    "ColumnMeta",
    "PG_OID_TO_ARROW",
]
