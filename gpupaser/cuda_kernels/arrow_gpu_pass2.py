"""
GPU Pass‑2 Kernel: 可変長列 scatter‑copy
--------------------------------------
1 カーネル呼び出しで「ある 1 つの可変長列」を処理する想定。
launch する側 (gpu_decoder_v2.py) で列ごとに呼び出す。

* UTF8 / BINARY          : raw[off:off+len] → values_buf[offsets[row] : ]
* NUMERIC (UTF8 文字列)  : numeric binary → 10進文字列化 → values_buf

Parameters
----------
raw            : uint8[:]               COPY バイナリ全体
field_offsets  : int32[:] (rows,)       対象列のフィールド先頭オフセット
field_lengths  : int32[:] (rows,)       同上, フィールド長 (-1=NULL)
offsets        : int32[:] (rows,)       pass‑1 prefix‑sum で得た dst オフセット
values_buf     : uint8[:]               書き込み先連結バッファ
numeric_mode   : int32                  0=普通コピー, 1=NUMERIC→文字列
"""

from numba import cuda
import numpy as np
from .numeric_utils import int64_to_decimal_ascii


@cuda.jit(device=True, inline=True)
def _copy_bytes(src, src_pos, dst, dst_pos, length):
    """最大 64byte 毎の単純コピー (未最適化)"""
    for i in range(length):
        dst[dst_pos + i] = src[src_pos + i]


@cuda.jit(device=True, inline=True)
def _parse_numeric_int64(raw, pos, lo_out, scale_out):
    """
    PostgreSQL NUMERIC binary から (int64 lo_val, int scale) を返す簡易版
    * 128bit 超は未対応 → lo_out=0, scale_out=-1 とする
    """
    if pos + 8 > len(raw):
        scale_out[0] = -1
        lo_out[0] = 0
        return

    ndigits = (raw[pos] << 8) | raw[pos + 1]
    weight = (raw[pos + 2] << 8) | raw[pos + 3]
    sign = (raw[pos + 4] << 8) | raw[pos + 5]
    dscale = (raw[pos + 6] << 8) | raw[pos + 7]

    # hi 桁未サポート → ndigits が 9 (=36桁) までを安全域とする
    if ndigits > 9:
        scale_out[0] = -1
        lo_out[0] = 0
        return

    pos += 8
    lo_val = np.int64(0)
    for i in range(ndigits):
        if pos + 1 >= len(raw):
            break
        digit = (raw[pos] << 8) | raw[pos + 1]
        lo_val = lo_val * 10000 + digit
        pos += 2

    exp = weight - (ndigits - 1)
    if exp > 0:
        for _ in range(exp):
            lo_val *= 10000
    elif exp < 0:
        for _ in range(-exp):
            lo_val //= 10000

    if sign == 0x4000:
        lo_val = -lo_val

    lo_out[0] = lo_val
    scale_out[0] = dscale


@cuda.jit
def pass2_scatter_varlen(raw,
                         field_offsets,
                         field_lengths,
                         offsets,
                         values_buf,
                         numeric_mode):
    row = cuda.grid(1)
    rows = field_offsets.size
    if row >= rows:
        return

    src_pos = field_offsets[row]
    flen = field_lengths[row]
    if flen == -1:
        # NULL の場合もoffsetは設定されているが、データは0長
        dst_pos = offsets[row]
        # 空文字列を明示的に設定（長さ=0）
        return

    dst_pos = offsets[row]

    if numeric_mode == 0:
        _copy_bytes(raw, src_pos, values_buf, dst_pos, flen)
    else:
        # NUMERIC → decimal128 (lo int64 + zero padding)
        lo_tmp = cuda.local.array(1, dtype=np.int64)
        sc_tmp = cuda.local.array(1, dtype=np.int32)
        _parse_numeric_int64(raw, src_pos, lo_tmp, sc_tmp)
        if sc_tmp[0] == -1:
            # overflow → zero
            for i in range(16):
                values_buf[dst_pos + i] = 0
            return
        # lo int64 (little-endian)
        lo_val = lo_tmp[0]
        for i in range(8):
            values_buf[dst_pos + i] = lo_val & 0xFF
            lo_val >>= 8
        # hi int64 = 0
        for i in range(8, 16):
            values_buf[dst_pos + i] = 0


__all__ = ["pass2_scatter_varlen"]
