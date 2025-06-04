"""Arrow 型まわりのユーティリティ関数群"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .types import (
    ColumnMeta, INT16, INT32, INT64, FLOAT32, FLOAT64, DECIMAL128,
    UTF8, BINARY, DATE32, TS64_US, BOOL, UNKNOWN,
)

# Arrow 型 ID → 固定長バイト数
_FIXED_SIZE: dict[int, int] = {
    INT16: 2, INT32: 4, INT64: 8,
    FLOAT32: 4, FLOAT64: 8, DECIMAL128: 16,
    DATE32: 4, TS64_US: 8, BOOL: 1,
}

def arrow_elem_size(arrow_id: int) -> int:
    """Arrow 型 ID から固定長バイト数 (可変長型は 0) を返す"""
    return _FIXED_SIZE.get(arrow_id, 0)

def build_gpu_meta_arrays(
    metas: List[ColumnMeta],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ColumnMeta のリストから GPU 転送向けメタデータ配列を生成"""
    n = len(metas)
    type_ids = np.empty(n, dtype=np.int32)
    elem_sizes = np.empty(n, dtype=np.int32)
    param1 = np.zeros(n, dtype=np.int32)
    param2 = np.zeros(n, dtype=np.int32)

    for i, m in enumerate(metas):
        type_ids[i] = m.arrow_id
        elem_sizes[i] = m.elem_size

        if m.arrow_param is None:
            continue
        if isinstance(m.arrow_param, tuple):
            param1[i], param2[i] = m.arrow_param
        else:
            param1[i] = int(m.arrow_param)

    return type_ids, elem_sizes, param1, param2

__all__ = ["arrow_elem_size", "build_gpu_meta_arrays"]