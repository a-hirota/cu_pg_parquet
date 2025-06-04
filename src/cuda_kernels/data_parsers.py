"""
データ解析関数
=============

PostgreSQLバイナリ形式からの各データ型解析を提供
"""

import numpy as np
from numba import cuda, uint64, int64, uint16, int16, uint8, int32
from .math_utils import add128_fast, mul128_u64_fast, neg128_fast, get_pow10_128, apply_scale_fast

@cuda.jit(device=True, inline=True)
def parse_decimal_from_raw(raw, src_offset, d_pow10_table_lo, d_pow10_table_hi, target_scale):
    """rawデータからDecimal値を解析して128ビット整数に変換"""
    
    # バリデーション
    if src_offset == 0 or src_offset + 8 > raw.size:
        return uint64(0), uint64(0)

    # NUMERIC ヘッダ読み取り (8バイト)
    nd = (uint16(raw[src_offset]) << 8) | uint16(raw[src_offset + 1])
    weight = int16((int16(raw[src_offset + 2]) << 8) | int16(raw[src_offset + 3]))
    sign = (uint16(raw[src_offset + 4]) << 8) | uint16(raw[src_offset + 5])
    dscale = (uint16(raw[src_offset + 6]) << 8) | uint16(raw[src_offset + 7])
    
    current_offset = src_offset + 8

    # NaN処理
    if sign == 0xC000:
        return uint64(0), uint64(0)

    # 桁数制限
    if nd > 9:
        return uint64(0), uint64(0)

    # 基数10000桁読み取り
    if current_offset + nd * 2 > raw.size:
        return uint64(0), uint64(0)

    # 基数10000から128ビット整数への変換
    val_hi = uint64(0)
    val_lo = uint64(0)
    
    # 基数1e8最適化実装
    i = 0
    while i < nd:
        if i + 1 < nd:
            digit1 = uint64((uint16(raw[current_offset]) << 8) | uint16(raw[current_offset + 1]))
            digit2 = uint64((uint16(raw[current_offset + 2]) << 8) | uint16(raw[current_offset + 3]))
            combined_digit = digit1 * uint64(10000) + digit2
            
            val_hi, val_lo = mul128_u64_fast(val_hi, val_lo, uint64(100000000))  # 1e8
            val_hi, val_lo = add128_fast(val_hi, val_lo, uint64(0), combined_digit)
            current_offset += 4
            i += 2
        else:
            digit = uint64((uint16(raw[current_offset]) << 8) | uint16(raw[current_offset + 1]))
            val_hi, val_lo = mul128_u64_fast(val_hi, val_lo, uint64(10000))
            val_hi, val_lo = add128_fast(val_hi, val_lo, uint64(0), digit)
            current_offset += 2
            i += 1

    # スケール調整
    pg_scale = int(dscale)
    val_hi, val_lo = apply_scale_fast(val_hi, val_lo, pg_scale, target_scale, d_pow10_table_lo, d_pow10_table_hi)

    # 符号適用
    if sign == 0x4000:
        val_hi, val_lo = neg128_fast(val_hi, val_lo)

    return val_hi, val_lo

@cuda.jit(device=True, inline=True)
def parse_int32_from_raw(raw, src_offset, field_length):
    """rawデータからint32値を解析"""
    
    if field_length != 4:
        return int32(0)
    
    if src_offset + 4 > raw.size:
        return int32(0)
    
    # ビッグエンディアンで読み取り
    value = (int32(raw[src_offset]) << 24) | \
            (int32(raw[src_offset + 1]) << 16) | \
            (int32(raw[src_offset + 2]) << 8) | \
            int32(raw[src_offset + 3])
    
    return value

@cuda.jit(device=True, inline=True)
def parse_int16_from_raw(raw, src_offset, field_length):
    """rawデータからint16値を解析"""
    if src_offset == 0 or field_length != 2 or src_offset + 2 > raw.size:
        return int16(0)
    
    # ビッグエンディアンで読み取り
    value = (int16(raw[src_offset]) << 8) | int16(raw[src_offset + 1])
    return value

@cuda.jit(device=True, inline=True)
def parse_int64_from_raw(raw, src_offset, field_length):
    """rawデータからint64値を解析"""
    if src_offset == 0 or field_length != 8 or src_offset + 8 > raw.size:
        return int64(0)
    
    # ビッグエンディアンで読み取り
    value = int64(0)
    for i in range(8):
        value = (value << 8) | int64(raw[src_offset + i])
    
    return value

__all__ = [
    "parse_decimal_from_raw", "parse_int32_from_raw", 
    "parse_int16_from_raw", "parse_int64_from_raw"
]