"""
数学演算とバッファ操作ユーティリティ
====================================

128ビット演算、データ型変換、バッファ書き込み関数を提供
"""

import numpy as np
from numba import cuda, uint64, int64, uint16, int16, uint8, int32

# 128ビット演算ヘルパー
@cuda.jit(device=True, inline=True)
def add128_fast(a_hi, a_lo, b_hi, b_lo):
    """高速128ビット加算"""
    res_lo = a_lo + b_lo
    carry = uint64(1) if res_lo < a_lo else uint64(0)
    res_hi = a_hi + b_hi + carry
    return res_hi, res_lo

@cuda.jit(device=True, inline=True)
def mul128_u64_fast(a_hi, a_lo, b):
    """高速128ビット × 64ビット乗算"""
    mask32 = uint64(0xFFFFFFFF)
    
    a0 = a_lo & mask32
    a1 = a_lo >> 32
    a2 = a_hi & mask32
    a3 = a_hi >> 32
    
    b0 = b & mask32
    b1 = b >> 32
    
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    p20 = a2 * b0
    p21 = a2 * b1
    p30 = a3 * b0
    
    c0 = p00 >> 32
    r0 = p00 & mask32
    
    temp1 = p01 + p10 + c0
    c1 = temp1 >> 32
    r1 = temp1 & mask32
    
    temp2 = p11 + p20 + c1
    c2 = temp2 >> 32
    r2 = temp2 & mask32
    
    temp3 = p21 + p30 + c2
    r3 = temp3 & mask32
    
    res_lo = (r1 << 32) | r0
    res_hi = (r3 << 32) | r2
    
    return res_hi, res_lo

@cuda.jit(device=True, inline=True)
def neg128_fast(hi, lo):
    """高速128ビット2の補数"""
    neg_lo = (~lo) + uint64(1)
    neg_hi = (~hi) + (uint64(1) if neg_lo == 0 and lo != 0 else uint64(0))
    return neg_hi, neg_lo

@cuda.jit(device=True, inline=True)
def get_pow10_128(n, d_pow10_table_lo, d_pow10_table_hi):
    """10^n を128ビット値として返す"""
    if 0 <= n < d_pow10_table_lo.shape[0]:
        return d_pow10_table_hi[n], d_pow10_table_lo[n]
    else:
        return uint64(0), uint64(0)

@cuda.jit(device=True, inline=True)
def apply_scale_fast(val_hi, val_lo, source_scale, target_scale, d_pow10_table_lo, d_pow10_table_hi):
    """スケール調整の高速実装"""
    if source_scale == target_scale:
        return val_hi, val_lo
    
    scale_diff = target_scale - source_scale
    
    if scale_diff > 0:
        pow_hi, pow_lo = get_pow10_128(scale_diff, d_pow10_table_lo, d_pow10_table_hi)
        if pow_hi == 0 and pow_lo == 0 and scale_diff != 0:
            return uint64(0), uint64(0)
        
        if pow_hi == 0:
            return mul128_u64_fast(val_hi, val_lo, pow_lo)
        else:
            return uint64(0), uint64(0)
    else:
        abs_diff = -scale_diff
        pow_hi, pow_lo = get_pow10_128(abs_diff, d_pow10_table_lo, d_pow10_table_hi)
        if pow_hi == 0 and pow_lo == 0 and abs_diff != 0:
            return uint64(0), uint64(0)

        if pow_hi == 0:
            if pow_lo == 0:
                return uint64(0), uint64(0)
            if val_hi == 0:
                return uint64(0), val_lo // pow_lo
            else:
                quotient_hi = val_hi // pow_lo
                quotient_lo = val_lo // pow_lo
                return quotient_hi, quotient_lo
        else:
            return uint64(0), uint64(0)

# バッファ書き込み関数
@cuda.jit(device=True, inline=True)
def write_int16_to_buffer(buffer, offset, value):
    """int16値をバッファに書き込み（リトルエンディアン）"""
    buffer[offset] = uint8(value & 0xFF)
    buffer[offset + 1] = uint8((value >> 8) & 0xFF)

@cuda.jit(device=True, inline=True)
def write_int32_to_buffer(buffer, offset, value):
    """int32値をバッファに書き込み（リトルエンディアン）"""
    buffer[offset] = uint8(value & 0xFF)
    buffer[offset + 1] = uint8((value >> 8) & 0xFF)
    buffer[offset + 2] = uint8((value >> 16) & 0xFF)
    buffer[offset + 3] = uint8((value >> 24) & 0xFF)

@cuda.jit(device=True, inline=True)
def write_int64_to_buffer(buffer, offset, value):
    """int64値をバッファに書き込み（リトルエンディアン）"""
    for i in range(8):
        buffer[offset + i] = uint8((value >> (i * 8)) & 0xFF)

@cuda.jit(device=True, inline=True)
def write_decimal128_to_buffer(buffer, offset, hi, lo):
    """decimal128値をバッファに書き込み（リトルエンディアン）"""
    # 下位64ビット
    for i in range(8):
        buffer[offset + i] = uint8((lo >> (i * 8)) & 0xFF)
    # 上位64ビット
    for i in range(8):
        buffer[offset + 8 + i] = uint8((hi >> (i * 8)) & 0xFF)

@cuda.jit(device=True, inline=True)
def write_float32_to_buffer(buffer, offset, value):
    """float32値をバッファに書き込み"""
    int_value = cuda.libdevice.float_as_int(value)
    write_int32_to_buffer(buffer, offset, int_value)

@cuda.jit(device=True, inline=True)
def write_float64_to_buffer(buffer, offset, value):
    """float64値をバッファに書き込み"""
    int_value = cuda.libdevice.double_as_longlong(value)
    write_int64_to_buffer(buffer, offset, int_value)

__all__ = [
    "add128_fast", "mul128_u64_fast", "neg128_fast", "get_pow10_128", "apply_scale_fast",
    "write_int16_to_buffer", "write_int32_to_buffer", "write_int64_to_buffer",
    "write_decimal128_to_buffer", "write_float32_to_buffer", "write_float64_to_buffer"
]