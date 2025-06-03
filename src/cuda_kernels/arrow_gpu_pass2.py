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


@cuda.jit(device=True)
def decode_numeric_postgres_simple(src_buf, src_pos, src_len):
    """
    PostgreSQL NUMERIC バイナリ形式を簡易デコード
    
    Returns
    -------
    (success, value, scale) : (bool, int64, int32)
    """
    if src_len < 8:
        return False, 0, 0
        
    # PostgreSQL NUMERIC binary format (simplified):
    # 2 bytes: ndigits
    # 2 bytes: weight
    # 2 bytes: sign (0x0000=pos, 0x4000=neg, 0xC000=NaN)
    # 2 bytes: dscale
    # 4*ndigits bytes: digits (each digit is 2 bytes, base 10000)
    
    ndigits = (src_buf[src_pos] << 8) | src_buf[src_pos + 1]
    weight = np.int16((src_buf[src_pos + 2] << 8) | src_buf[src_pos + 3])
    sign = (src_buf[src_pos + 4] << 8) | src_buf[src_pos + 5]
    dscale = (src_buf[src_pos + 6] << 8) | src_buf[src_pos + 7]
    
    if sign == 0xC000:  # NaN
        return False, 0, 0
    
    if ndigits == 0:  # Zero
        return True, 0, dscale
        
    # 簡易実装: 最初の数桁のみ処理（int64範囲内）
    if ndigits > 4 or src_len < 8 + ndigits * 2:
        return False, 0, 0
        
    value = 0
    digit_pos = src_pos + 8
    
    for i in range(min(ndigits, 4)):
        if digit_pos + 1 >= src_pos + src_len:
            break
        digit = (src_buf[digit_pos] << 8) | src_buf[digit_pos + 1]
        if digit >= 10000:
            digit = 9999  # clamp
        value = value * 10000 + digit
        digit_pos += 2
    
    # weight調整（簡易版）
    if weight >= 0:
        for i in range(weight):
            value *= 10000
    
    if sign == 0x4000:  # negative
        value = -value
        
    return True, value, dscale


@cuda.jit
def pass2_scatter_varlen(raw,          # uint8[:]
                         field_offsets,# int32[:] (rows,)
                         field_lengths,# int32[:] (rows,)
                         offsets,      # int32[:] (rows+1,) - Note: kernel uses offsets[row]
                         values_buf,   # uint8[:]
                         numeric_mode=0): # int32: 0=normal copy, 1=NUMERIC→string
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

    if numeric_mode == 1:
        # NUMERIC binary → string conversion
        success, numeric_value, scale = decode_numeric_postgres_simple(raw, src_pos, flen)
        if success:
            # Convert to decimal string
            written_len = int64_to_decimal_ascii(numeric_value, scale, values_buf, dst_pos)
            # Update the length in the offsets (this is a hack, proper implementation would pre-calculate)
            # Note: This assumes the caller will handle length updates properly
        else:
            # Fallback: write "NULL" or "ERROR"
            error_msg = b"ERROR"
            for i in range(min(5, len(error_msg))):
                values_buf[dst_pos + i] = error_msg[i]
    else:
        # Normal byte copy for UTF8/BINARY
        _copy_bytes(raw, src_pos, values_buf, dst_pos, flen)


__all__ = ["pass2_scatter_varlen"]
