"""
gpupgparser - GPU PostgreSQL Parser (ZeroCopy版)

RAPIDS cuDF を活用した高速PostgreSQLバイナリデータ処理
ZeroCopy実装をデフォルトとして使用
"""

# メインAPIの公開 (ZeroCopy版)
from .main_postgres_to_parquet import (
    ZeroCopyProcessor,
    postgresql_to_cudf_parquet
)

# 詳細制御用API
from .build_cudf_from_buf import CuDFZeroCopyProcessor
from .cuda_kernels.postgres_binary_parser import (
    parse_binary_chunk_gpu_ultra_fast_v2,
    parse_binary_chunk_gpu_ultra_fast_v2_integrated,
    detect_pg_header_size
)
from .write_parquet_from_cudf import (
    write_cudf_to_parquet,
    write_cudf_to_parquet_with_options
)

# 基盤コンポーネント
from .metadata import fetch_column_meta
from .types import ColumnMeta, PG_OID_TO_ARROW
from .memory_manager import GPUMemoryManager

# kvikio直接読み込み機能
from .heap_file_reader import (
    read_heap_file_direct,
    get_heap_file_info,
    HeapFileReaderError,
    KvikioNotInitializedError
)

__version__ = "0.2.0"  # ZeroCopy版

__all__ = [
    # メイン処理
    "ZeroCopyProcessor",
    "postgresql_to_cudf_parquet",
    
    # 詳細制御
    "CuDFZeroCopyProcessor",
    "parse_binary_chunk_gpu_ultra_fast_v2",
    "parse_binary_chunk_gpu_ultra_fast_v2_integrated",
    "detect_pg_header_size",
    "write_cudf_to_parquet",
    "write_cudf_to_parquet_with_options",
    
    # 基盤
    "fetch_column_meta",
    "ColumnMeta",
    "PG_OID_TO_ARROW",
    "GPUMemoryManager",
    
    # kvikio直接読み込み
    "read_heap_file_direct",
    "get_heap_file_info",
    "HeapFileReaderError",
    "KvikioNotInitializedError",
]
