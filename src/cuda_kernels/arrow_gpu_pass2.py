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


@cuda.jit(device=True, inline=True)
def _copy_bytes(src, src_pos, dst, dst_pos, length):
    """最大 64byte 毎の単純コピー (未最適化)"""
    for i in range(length):
        dst[dst_pos + i] = src[src_pos + i]


@cuda.jit
def pass2_scatter_varlen(raw,          # uint8[:]
                         field_offsets,# int32[:] (rows,)
                         field_lengths,# int32[:] (rows,)
                         offsets,      # int32[:] (rows+1,) - Note: kernel uses offsets[row]
                         values_buf):  # uint8[:]
    row = cuda.grid(1)
    rows = field_offsets.size
    if row >= rows:
        return

    src_pos = field_offsets[row]
    flen = field_lengths[row]
    if flen == -1:
        # NULL の場合もoffsetは設定されているが、データは0長
        dst_pos = offsets[row]
        # NULL の場合もoffsetは設定されているが、データは0長なのでコピー不要
        return

    dst_pos = offsets[row]

    # Always perform simple byte copy
    _copy_bytes(raw, src_pos, values_buf, dst_pos, flen)


__all__ = ["pass2_scatter_varlen"]
