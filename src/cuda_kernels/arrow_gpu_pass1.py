"""
GPU Pass‑1 Kernel: 行並列で各列の NULL フラグと可変長列の長さを収集

前提
-----
* `field_lengths` は shape=(rows, ncols) の int32 DeviceNDArray
    - PostgreSQL COPY BINARY 解析カーネル (parse_binary_format_kernel_one_row)
      が出力するフィールド長 (-1 = NULL, 0 以上 = バイト長) を 2D に
      reshape したもの
* `var_indices` は shape=(ncols,) の int32 DeviceNDArray
    - 固定長列      : -1
    - 可変長列      : 0 .. n_var‑1  (可変長列ごとのインデックス)
* `d_var_lens` は shape=(n_var, rows) の int32 DeviceNDArray
    - 可変長列ごとのバイト長を書き込む
* `d_nulls` は shape=(rows, ncols) の uint8 DeviceNDArray
    - NULL なら 0, 有効なら 1 をセット (Arrow 形式)
"""
import numpy as np
from numba import cuda


@cuda.jit
def pass1_len_null(field_lengths, var_indices, d_var_lens, d_nulls):
    """
    Parameters
    ----------
    field_lengths : int32[:, :]
        各行×列のフィールド長 (-1 = NULL)
    var_indices   : int32[:]
        列 → 可変長列インデックス (固定長は -1)
    d_var_lens    : int32[:, :]
        (out) 可変長列 × 行 のバイト長
    d_nulls       : uint8[:, :]
        (out) 行 × 列 の NULL フラグ (NULL=0, Valid=1)
    """
    row = cuda.grid(1)
    rows = field_lengths.shape[0]
    if row >= rows:
        return

    ncols = field_lengths.shape[1]

    for col in range(ncols):
        flen = field_lengths[row, col]
        # Explicitly compare with int32(-1)
        is_null = (flen == np.int32(-1))

        # Arrow の validity ビットマップは
        # 1 = 有効 (NOT NULL), 0 = NULL
        # Use consistent [row, col] indexing
        # Explicitly assign uint8 values
        if is_null:
            d_nulls[row, col] = np.uint8(0)
        else:
            d_nulls[row, col] = np.uint8(1)

        # 可変長列なら長さを保存
        v_idx = var_indices[col]
        if v_idx != -1:
            d_var_lens[v_idx, row] = 0 if is_null else flen


__all__ = ["pass1_len_null"]
