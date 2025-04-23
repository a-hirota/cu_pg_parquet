"""
Arrow 型まわりのユーティリティ関数群

目的
-----
* ColumnMeta から GPU 転送しやすい int32 配列を組み立てる
* Arrow 型 ID → 固定長バイト数を取得する
* DECIMAL / UTF8 など可変長型の追加パラメータを取り出しやすくする
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .type_map import (
    ColumnMeta,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    DECIMAL128,
    UTF8,
    BINARY,
    DATE32,
    TS64_US,
    BOOL,
    UNKNOWN,
)


# ----------------------------------------------------------------------
# Arrow 型 ID → 固定長バイト数
# 可変長型は 0 を返す
# ----------------------------------------------------------------------
_FIXED_SIZE: dict[int, int] = {
    INT16: 2,
    INT32: 4,
    INT64: 8,
    FLOAT32: 4,
    FLOAT64: 8,
    DECIMAL128: 16,
    DATE32: 4,
    TS64_US: 8,
    BOOL: 1,
}


def arrow_elem_size(arrow_id: int) -> int:
    """Arrow 型 ID から固定長バイト数 (可変長型は 0) を返す"""
    return _FIXED_SIZE.get(arrow_id, 0)


# ----------------------------------------------------------------------
# ColumnMeta 配列 → GPU 転送用 int32 配列群
# ----------------------------------------------------------------------
def build_gpu_meta_arrays(
    metas: List[ColumnMeta],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ColumnMeta のリストから GPU 転送向けメタデータ配列を生成する。

    Returns
    -------
    type_ids : np.ndarray[int32]
        Arrow 型 ID
    elem_sizes : np.ndarray[int32]
        固定長バイト数 (可変長型は 0)
    param1 : np.ndarray[int32]
        DECIMAL の precision / VARCHAR の maxlen 等
    param2 : np.ndarray[int32]
        DECIMAL の scale 等
    """
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


__all__ = [
    "arrow_elem_size",
    "build_gpu_meta_arrays",
]
