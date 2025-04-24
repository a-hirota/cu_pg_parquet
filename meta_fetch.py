"""
Alias module for gpupaser.meta_fetch.
Provides ColumnMeta and fetch_column_meta at top-level for backward compatibility.
"""
from gpupaser.meta_fetch import ColumnMeta, fetch_column_meta  # fetch_column_meta を追加

__all__ = ["ColumnMeta", "fetch_column_meta"]  # __all__ にも追加
