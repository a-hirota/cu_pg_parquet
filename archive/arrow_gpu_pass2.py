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
    # デバッグ情報出力（最初の数行のみ）
    row_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if row_id < 5:  # 最初の5行のみデバッグ
        cuda.printf("PASS2: row=%d, src_pos=%d, dst_pos=%d, length=%d\n",
                   row_id, src_pos, dst_pos, length)
        cuda.printf("PASS2: src.size=%d, dst.size=%d\n",
                   src.size, dst.size)
        
        # コピー前の最初の数バイトを表示
        if length > 0 and src_pos < src.size:
            cuda.printf("PASS2: src_data[0-3]: %d %d %d %d\n",
                       src[src_pos] if src_pos < src.size else 255,
                       src[src_pos + 1] if src_pos + 1 < src.size else 255,
                       src[src_pos + 2] if src_pos + 2 < src.size else 255,
                       src[src_pos + 3] if src_pos + 3 < src.size else 255)
    
    for i in range(length):
        dst[dst_pos + i] = src[src_pos + i]
    
    # デバッグ情報を出力（コピー後）
    if row_id < 5 and length > 0:
        cuda.printf("PASS2: dst_data[0-3]: %d %d %d %d\n",
                   dst[dst_pos] if dst_pos < dst.size else 255,
                   dst[dst_pos + 1] if dst_pos + 1 < dst.size else 255,
                   dst[dst_pos + 2] if dst_pos + 2 < dst.size else 255,
                   dst[dst_pos + 3] if dst_pos + 3 < dst.size else 255)


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
