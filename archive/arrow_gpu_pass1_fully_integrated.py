"""
GPU Pass1完全統合カーネル: 固定長列統合バッファ版
================================================

ColumnMetaベースで固定長列を統合バッファに配置し、
1回のカーネル起動で全列処理を実現

主な特徴:
1. 固定長列: 統合バッファ（ブロック単位最適化）
2. 可変長列: 従来の個別バッファ処理
3. 1回のカーネル起動で全列処理
4. メモリコアレッシング最適化
"""

import numpy as np
from numba import cuda, uint64, int64, uint16, int16, uint8, boolean, int32

# 型定数（type_map.pyと一致させる）
TYPE_INT16 = 11
TYPE_INT32 = 1
TYPE_INT64 = 2
TYPE_FLOAT32 = 3
TYPE_FLOAT64 = 4
TYPE_DECIMAL128 = 5
TYPE_UTF8 = 6
TYPE_BINARY = 7
TYPE_BOOL = 8
TYPE_DATE32 = 9
TYPE_TS64_US = 10

# 128ビット演算ヘルパー（decimal128処理用）
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
    # float32をint32として解釈してバイト分解
    int_value = cuda.libdevice.float_as_int(value)
    write_int32_to_buffer(buffer, offset, int_value)

@cuda.jit(device=True, inline=True)
def write_float64_to_buffer(buffer, offset, value):
    """float64値をバッファに書き込み"""
    # float64をint64として解釈してバイト分解
    int_value = cuda.libdevice.double_as_longlong(value)
    write_int64_to_buffer(buffer, offset, int_value)

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

@cuda.jit
def pass1_fully_integrated(
    raw,                    # const uint8_t* (入力rawデータ)
    field_offsets,          # const int32_t[:, :] (rows, cols)
    field_lengths,          # const int32_t[:, :] (rows, cols)
    
    # 統合固定長バッファ
    unified_fixed_buffer,   # uint8_t* (全固定長列の統合バッファ)
    row_stride,             # int (1行分のバイト数)
    
    # 固定長レイアウト情報
    fixed_column_count,     # int (固定長列数)
    fixed_column_types,     # int32_t[:] (各列の型ID)
    fixed_column_offsets,   # int32_t[:] (統合バッファ内の各列オフセット)
    fixed_column_sizes,     # int32_t[:] (各列のサイズ)
    fixed_column_indices,   # int32_t[:] (元のテーブル列インデックス)
    fixed_decimal_scales,   # int32_t[:] (Decimal列のスケール情報)
    
    # 可変長バッファ（従来方式）
    var_column_count,       # int (可変長列数)
    var_column_indices,     # int32_t[:] (可変長列の元テーブル列インデックス)
    d_var_lens,             # int32_t[:, :] (可変長列の長さ配列)
    
    # 共通出力
    d_nulls_all,            # uint8_t[:, :] (NULL フラグ)
    
    # Decimal処理用
    d_pow10_table_lo,       # uint64_t[:] (10^n テーブル下位)
    d_pow10_table_hi        # uint64_t[:] (10^n テーブル上位)
):
    """
    Pass1完全統合カーネル
    
    1回のカーネル起動で以下を統合実行:
    1. 全列のNULL判定
    2. 可変長列の長さ記録
    3. 固定長列の解析・変換・統合バッファ書き込み
    """
    row = cuda.grid(1)
    rows, total_cols = field_offsets.shape
    if row >= rows:
        return

    # 統合バッファ内の行開始位置
    row_base_offset = row * row_stride

    # --- 固定長列の統合処理 ---
    for fixed_idx in range(fixed_column_count):
        col_idx = fixed_column_indices[fixed_idx]
        col_type = fixed_column_types[fixed_idx]
        col_offset = fixed_column_offsets[fixed_idx]
        col_size = fixed_column_sizes[fixed_idx]
        
        # フィールド情報取得
        src_offset = field_offsets[row, col_idx]
        field_length = field_lengths[row, col_idx]
        
        # NULL判定
        is_null = (field_length == -1)
        d_nulls_all[row, col_idx] = uint8(0 if is_null else 1)
        
        # NULL の場合はゼロで埋める
        if is_null:
            buffer_offset = row_base_offset + col_offset
            for i in range(col_size):
                unified_fixed_buffer[buffer_offset + i] = uint8(0)
            continue
        
        # 型別の解析・変換・書き込み
        buffer_offset = row_base_offset + col_offset
        
        if col_type == TYPE_DECIMAL128:
            # Decimal128処理
            decimal_scale = fixed_decimal_scales[fixed_idx]
            val_hi, val_lo = parse_decimal_from_raw(raw, src_offset, d_pow10_table_lo, d_pow10_table_hi, decimal_scale)
            write_decimal128_to_buffer(unified_fixed_buffer, buffer_offset, val_hi, val_lo)
            
        elif col_type == TYPE_INT32:
            # INT32処理
            value = parse_int32_from_raw(raw, src_offset, field_length)
            write_int32_to_buffer(unified_fixed_buffer, buffer_offset, value)
            
        elif col_type == TYPE_INT16:
            # INT16処理
            value = parse_int16_from_raw(raw, src_offset, field_length)
            write_int16_to_buffer(unified_fixed_buffer, buffer_offset, value)
            
        elif col_type == TYPE_INT64:
            # INT64処理
            value = parse_int64_from_raw(raw, src_offset, field_length)
            write_int64_to_buffer(unified_fixed_buffer, buffer_offset, value)
            
        elif col_type == TYPE_FLOAT32:
            # FLOAT32処理（基本的な実装）
            if field_length == 4 and src_offset + 4 <= raw.size:
                # ビッグエンディアンで読み取りしてfloat32に変換
                int_value = (int32(raw[src_offset]) << 24) | \
                           (int32(raw[src_offset + 1]) << 16) | \
                           (int32(raw[src_offset + 2]) << 8) | \
                           int32(raw[src_offset + 3])
                # int32をfloat32として解釈
                float_value = cuda.libdevice.int_as_float(int_value)
                write_float32_to_buffer(unified_fixed_buffer, buffer_offset, float_value)
            else:
                # エラーの場合はゼロで埋める
                for i in range(col_size):
                    unified_fixed_buffer[buffer_offset + i] = uint8(0)
            
        elif col_type == TYPE_FLOAT64:
            # FLOAT64処理（基本的な実装）
            if field_length == 8 and src_offset + 8 <= raw.size:
                # ビッグエンディアンで読み取りしてfloat64に変換
                int_value = int64(0)
                for i in range(8):
                    int_value = (int_value << 8) | int64(raw[src_offset + i])
                # int64をfloat64として解釈
                float_value = cuda.libdevice.longlong_as_double(int_value)
                write_float64_to_buffer(unified_fixed_buffer, buffer_offset, float_value)
            else:
                # エラーの場合はゼロで埋める
                for i in range(col_size):
                    unified_fixed_buffer[buffer_offset + i] = uint8(0)
            
        else:
            # 未対応型: ゼロで埋める
            for i in range(col_size):
                unified_fixed_buffer[buffer_offset + i] = uint8(0)

    # --- 可変長列の従来処理 ---
    for var_idx in range(var_column_count):
        col_idx = var_column_indices[var_idx]
        
        # フィールド情報取得
        field_length = field_lengths[row, col_idx]
        
        # NULL判定
        is_null = (field_length == -1)
        d_nulls_all[row, col_idx] = uint8(0 if is_null else 1)
        
        # 長さ記録
        d_var_lens[var_idx, row] = 0 if is_null else field_length

__all__ = ["pass1_fully_integrated"]