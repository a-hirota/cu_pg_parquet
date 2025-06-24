"""
numeric_utils.py
================
NUMERIC (<= 64‑bit) の値を GPU 上で文字列化するための補助 device 関数群。

現状 decode_numeric_postgres() は hi=0, lo=int64, scale=int16 を出力する
簡易版であり、38 桁以内の値のみ対象とする。この前提のもと、

    ・符号付き int64 (lo) と scale(0‑38) を 10 進文字列へ変換
    ・戻り値は書き込んだバイト長

※ hi!=0 （>64bit 値）は未対応 → “OVERFLOW” を書き込む
"""

from numba import cuda
import numpy as np


@cuda.jit(device=True, inline=True)
def _reverse_bytes(buf, start, end):
    """in‑place reverse for bytes[ start : end )"""
    l = start
    r = end - 1
    while l < r:
        tmp = buf[l]
        buf[l] = buf[r]
        buf[r] = tmp
        l += 1
        r -= 1


@cuda.jit(device=True)
def int64_to_decimal_ascii(lo_val, scale, out_buf, out_pos):
    """
    Parameters
    ----------
    lo_val : int64
    scale   : int32  (0 ～ 38)
    out_buf : uint8[:]
    out_pos : int32  書き込み開始オフセット

    Returns
    -------
    int32  実際に書き込んだバイト長
    """
    # 0 特別扱い
    if lo_val == 0:
        out_buf[out_pos] = ord('0')
        if scale > 0:
            out_buf[out_pos + 1] = ord('.')
            # "0.000" など scale 桁分 '0'
            for i in range(scale):
                out_buf[out_pos + 2 + i] = ord('0')
            return 2 + scale
        else:
            return 1

    is_neg = lo_val < 0
    if is_neg:
        lo_val = -lo_val

    # バッファに逆順で桁を書き込む
    idx = out_pos
    written = 0
    s = 0  # 桁数カウント

    while lo_val > 0 or s <= scale:
        digit = lo_val % 10
        lo_val //= 10
        out_buf[idx] = np.uint8(ord('0') + digit)
        idx += 1
        written += 1
        s += 1
        # 小数点位置
        if s == scale and scale != 0:
            out_buf[idx] = ord('.')
            idx += 1
            written += 1
            # 小数点後もカウントし続ける (s は整数部カウンタとしてリセットしない)

    # 先頭に負符号
    if is_neg:
        out_buf[idx] = ord('-')
        idx += 1
        written += 1

    # 文字列を正順に並べ替え (現在は逆順 + '.' が逆位置)
    _reverse_bytes(out_buf, out_pos, out_pos + written)

    return written


__all__ = ["int64_to_decimal_ascii"]
