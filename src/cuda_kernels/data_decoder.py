"""
データデコーダー
===============

フィールドデータの型変換とバッファ書き込み処理
固定長・可変長データの統合デコード
"""

import numpy as np
from numba import cuda, uint64, int64, uint16, int16, uint8, boolean, int32, uint32
import math

# 型定数（type_map.pyと一致）
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

# 数学演算とバッファ操作関数（統合版）
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

# データ解析関数
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

@cuda.jit(device=True, inline=True)
def copy_string_data_safe(raw_data, src_offset, length, dest_buffer, dest_offset):
    """安全な文字列データコピー"""
    if length <= 0 or src_offset >= raw_data.size or dest_offset >= dest_buffer.size:
        return 0
    
    # 境界チェック
    max_copy_len = min(
        length,
        raw_data.size - src_offset,
        dest_buffer.size - dest_offset
    )
    
    if max_copy_len > 0:
        for i in range(max_copy_len):
            dest_buffer[dest_offset + i] = raw_data[src_offset + i]
    
    return max_copy_len

@cuda.jit
def pass1_column_wise_integrated(
    raw,                      # const uint8_t* (入力rawデータ)
    field_offsets,            # const int32_t[:, :] (rows, cols)
    field_lengths,            # const int32_t[:, :] (rows, cols)
    
    # 列メタデータ配列
    column_types,             # int32_t[:] (各列の型ID)
    column_is_variable,       # uint8_t[:] (可変長フラグ)
    column_indices,           # int32_t[:] (列インデックス)
    
    # 固定長統合バッファ
    unified_fixed_buffer,     # uint8_t* (全固定長列の統合バッファ)
    fixed_column_offsets,     # int32_t[:] (統合バッファ内の各列オフセット)
    fixed_column_sizes,       # int32_t[:] (各列のサイズ)
    fixed_decimal_scales,     # int32_t[:] (Decimal列のスケール情報)
    row_stride,               # int (1行分のバイト数)
    
    # 可変長統合バッファ
    var_data_buffer,          # uint8_t[:] (統合可変長データバッファ)
    var_offset_arrays,        # int32_t[:, :] (各可変長列のオフセット配列)
    var_column_mapping,       # int32_t[:] (列インデックス→可変長インデックス)
    
    # 共通出力
    d_nulls_all,              # uint8_t[:, :] (NULL フラグ)
    
    # Decimal処理用
    d_pow10_table_lo,         # uint64_t[:] (10^n テーブル下位)
    d_pow10_table_hi          # uint64_t[:] (10^n テーブル上位)
):
    """
    統合データデコードカーネル
    
    固定長・可変長両方をSingle Kernelで処理し、
    列順序での段階的処理によりキャッシュ効率を最大化
    """
    
    # 共有メモリバッファ（ブロック内12KB高速処理）
    shared_string_buffer = cuda.shared.array(12288, dtype=uint8)
    
    # グリッド情報
    row = cuda.grid(1)
    rows, total_cols = field_offsets.shape
    thread_id = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    if row >= rows:
        return
    
    # 固定長列インデックス
    fixed_col_idx = 0
    
    # 共有メモリ内でのブロック単位処理
    shared_offset = thread_id * 32  # スレッドごとに32バイト領域（バンク競合回避）
    
    # 固定長・可変長の列順序処理
    for col_idx in range(total_cols):
        if col_idx >= column_types.size:
            break
            
        col_type = column_types[col_idx]
        is_variable = column_is_variable[col_idx]
        
        # フィールド情報取得
        src_offset = field_offsets[row, col_idx]
        field_length = field_lengths[row, col_idx]
        
        # NULL判定
        is_null = (field_length == -1)
        if row < d_nulls_all.shape[0] and col_idx < d_nulls_all.shape[1]:
            d_nulls_all[row, col_idx] = uint8(0 if is_null else 1)
        
        if is_variable and (col_type == TYPE_UTF8 or col_type == TYPE_BINARY):
            # 可変長文字列列の統合処理（共有メモリ最適化版）
            var_idx = var_column_mapping[col_idx] if col_idx < var_column_mapping.size else -1
            
            if var_idx >= 0 and var_idx < var_offset_arrays.shape[0]:
                if is_null:
                    pass
                else:
                    # 共有メモリを使った高速処理
                    if (field_length > 0 and 
                        src_offset < raw.size and 
                        shared_offset + field_length < 12288):  # 共有メモリ境界チェック
                        
                        # 共有メモリに一時コピー（高速）
                        for i in range(field_length):
                            if src_offset + i < raw.size:
                                shared_string_buffer[shared_offset + i] = raw[src_offset + i]
                        
                        # コアレッシングアクセスでグローバルメモリ位置計算
                        var_write_offset = row * 1000 + col_idx * 100  # 一時的な計算方式
                        
                        # 共有メモリからvar_data_bufferに効率的コピー
                        if var_write_offset + field_length <= var_data_buffer.size:
                            actual_copied = copy_string_data_safe(
                                shared_string_buffer, shared_offset, field_length,
                                var_data_buffer, var_write_offset
                            )
                            
                            # この行・この列の開始位置を記録
                            if row < var_offset_arrays.shape[1]:
                                var_offset_arrays[var_idx, row] = var_write_offset
                        else:
                            # バッファオーバーフロー: NULL扱い
                            if row < var_offset_arrays.shape[1]:
                                var_offset_arrays[var_idx, row] = -1  # エラーマーク
                        
                        # 共有メモリオフセットを次の位置に進める
                        shared_offset += field_length
            
        else:
            # 固定長列の統合処理
            if fixed_col_idx < fixed_column_offsets.size:
                # 統合バッファ内の行開始位置
                row_base_offset = row * row_stride
                col_offset = fixed_column_offsets[fixed_col_idx]
                col_size = fixed_column_sizes[fixed_col_idx]
                
                # バッファ書き込み位置
                buffer_offset = row_base_offset + col_offset
                
                if is_null:
                    # NULLの場合はゼロで埋める
                    for i in range(col_size):
                        if buffer_offset + i < unified_fixed_buffer.size:
                            unified_fixed_buffer[buffer_offset + i] = uint8(0)
                else:
                    # 型別の解析・変換・書き込み
                    if col_type == TYPE_DECIMAL128:
                        decimal_scale = fixed_decimal_scales[fixed_col_idx] if fixed_col_idx < fixed_decimal_scales.size else 0
                        val_hi, val_lo = parse_decimal_from_raw(raw, src_offset, d_pow10_table_lo, d_pow10_table_hi, decimal_scale)
                        if buffer_offset + 16 <= unified_fixed_buffer.size:
                            write_decimal128_to_buffer(unified_fixed_buffer, buffer_offset, val_hi, val_lo)
                        
                    elif col_type == TYPE_INT32:
                        value = parse_int32_from_raw(raw, src_offset, field_length)
                        if buffer_offset + 4 <= unified_fixed_buffer.size:
                            write_int32_to_buffer(unified_fixed_buffer, buffer_offset, value)
                        
                    elif col_type == TYPE_INT16:
                        value = parse_int16_from_raw(raw, src_offset, field_length)
                        if buffer_offset + 2 <= unified_fixed_buffer.size:
                            write_int16_to_buffer(unified_fixed_buffer, buffer_offset, value)
                        
                    elif col_type == TYPE_INT64:
                        value = parse_int64_from_raw(raw, src_offset, field_length)
                        if buffer_offset + 8 <= unified_fixed_buffer.size:
                            write_int64_to_buffer(unified_fixed_buffer, buffer_offset, value)
                        
                    elif col_type == TYPE_FLOAT32:
                        if field_length == 4 and src_offset + 4 <= raw.size and buffer_offset + 4 <= unified_fixed_buffer.size:
                            int_value = (int32(raw[src_offset]) << 24) | \
                                       (int32(raw[src_offset + 1]) << 16) | \
                                       (int32(raw[src_offset + 2]) << 8) | \
                                       int32(raw[src_offset + 3])
                            float_value = cuda.libdevice.int_as_float(int_value)
                            write_float32_to_buffer(unified_fixed_buffer, buffer_offset, float_value)
                        else:
                            for i in range(col_size):
                                if buffer_offset + i < unified_fixed_buffer.size:
                                    unified_fixed_buffer[buffer_offset + i] = uint8(0)
                        
                    elif col_type == TYPE_FLOAT64:
                        if field_length == 8 and src_offset + 8 <= raw.size and buffer_offset + 8 <= unified_fixed_buffer.size:
                            int_value = int64(0)
                            for i in range(8):
                                int_value = (int_value << 8) | int64(raw[src_offset + i])
                            float_value = cuda.libdevice.longlong_as_double(int_value)
                            write_float64_to_buffer(unified_fixed_buffer, buffer_offset, float_value)
                        else:
                            for i in range(col_size):
                                if buffer_offset + i < unified_fixed_buffer.size:
                                    unified_fixed_buffer[buffer_offset + i] = uint8(0)
                                    
                    else:
                        # 未対応型: ゼロで埋める
                        for i in range(col_size):
                            if buffer_offset + i < unified_fixed_buffer.size:
                                unified_fixed_buffer[buffer_offset + i] = uint8(0)
            
            fixed_col_idx += 1

@cuda.jit
def build_var_offsets_from_lengths(var_offset_arrays, var_length_arrays, rows):
    """可変長オフセット配列構築"""
    var_idx = cuda.grid(1)
    if var_idx >= var_offset_arrays.shape[0]:
        return
    
    # オフセット配列の最初の要素は0
    if rows > 0:
        var_offset_arrays[var_idx, 0] = 0
        
        # 累積和を計算
        for row in range(1, rows + 1):
            if row <= var_length_arrays.shape[1] and row <= var_offset_arrays.shape[1]:
                prev_offset = var_offset_arrays[var_idx, row - 1]
                if row - 1 < var_length_arrays.shape[1]:
                    length = var_length_arrays[var_idx, row - 1]
                    var_offset_arrays[var_idx, row] = prev_offset + length
                else:
                    var_offset_arrays[var_idx, row] = prev_offset

__all__ = ["pass1_column_wise_integrated", "build_var_offsets_from_lengths"]