"""
GPU Pass1 Decimal統合カーネル: 列ごと処理版
==============================================

従来の列ごと処理を保持しつつ、Pass1段階でDecimal変換を実行
- Numbaの制約を回避（List[DeviceNDArray]を使わない）
- 各Decimal列ごとに個別にカーネル呼び出し
- NULL/長さ収集 + Decimal変換を統合
"""

import numpy as np
from numba import cuda, uint64, int64, uint16, int16, uint8, boolean

# 128ビット演算ヘルパー（最適化版から流用）
@cuda.jit(device=True, inline=True)
def add128_fast(a_hi, a_lo, b_hi, b_lo):
    """高速128ビット加算 (a + b)"""
    res_lo = a_lo + b_lo
    carry = uint64(1) if res_lo < a_lo else uint64(0)
    res_hi = a_hi + b_hi + carry
    return res_hi, res_lo

@cuda.jit(device=True, inline=True)
def mul128_u64_fast(a_hi, a_lo, b):
    """高速128ビット × 64ビット乗算"""
    # 32ビット分割による高速乗算
    mask32 = uint64(0xFFFFFFFF)
    
    # a_lo, a_hi, b を32ビット要素に分割
    a0 = a_lo & mask32
    a1 = a_lo >> 32
    a2 = a_hi & mask32
    a3 = a_hi >> 32
    
    b0 = b & mask32
    b1 = b >> 32
    
    # 部分積計算 (32×32→64)
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    p20 = a2 * b0
    p21 = a2 * b1
    p30 = a3 * b0
    
    # 結果の組み立て
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
def store_decimal128_le(hi, lo, dst_buf, offset):
    """128ビット値をリトルエンディアンで格納"""
    # 下位64ビット
    v = lo
    for i in range(8):
        dst_buf[offset + i] = uint8(v & 0xFF)
        v >>= 8
    
    # 上位64ビット
    v = hi
    for i in range(8):
        dst_buf[offset + 8 + i] = uint8(v & 0xFF)
        v >>= 8

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
        # スケールアップ: 10^scale_diff を乗算
        pow_hi, pow_lo = get_pow10_128(scale_diff, d_pow10_table_lo, d_pow10_table_hi)
        if pow_hi == 0 and pow_lo == 0 and scale_diff != 0:
            return uint64(0), uint64(0)  # オーバーフロー
        
        if pow_hi == 0:  # 乗数が64ビットに収まる場合
            return mul128_u64_fast(val_hi, val_lo, pow_lo)
        else:
            return uint64(0), uint64(0)  # オーバーフロー
    else:  # scale_diff < 0
        # スケールダウン: 10^(-scale_diff) で除算
        abs_diff = -scale_diff
        pow_hi, pow_lo = get_pow10_128(abs_diff, d_pow10_table_lo, d_pow10_table_hi)
        if pow_hi == 0 and pow_lo == 0 and abs_diff != 0:
            return uint64(0), uint64(0)  # アンダーフロー

        if pow_hi == 0:  # 除数が64ビットに収まる場合
            if pow_lo == 0:
                return uint64(0), uint64(0)  # ゼロ除算回避
            # 簡易除算（完全実装が必要な場合は長除算アルゴリズムを使用）
            if val_hi == 0:
                return uint64(0), val_lo // pow_lo
            else:
                # 128ビット除算の簡易実装
                quotient_hi = val_hi // pow_lo
                remainder_hi = val_hi % pow_lo
                quotient_lo = val_lo // pow_lo
                return quotient_hi, quotient_lo
        else:
            return uint64(0), uint64(0)  # アンダーフロー

@cuda.jit
def pass1_len_null_decimal_column_wise(
    raw,                # const uint8_t* __restrict__
    field_offsets,      # const int32_t* __restrict__ (rows,)
    field_lengths,      # const int32_t* __restrict__ (rows,)
    decimal_dst_buf,    # uint8_t* (output buffer for this Decimal column)
    stride,             # int (must be 16 for Decimal128)
    target_scale,       # int (column scale)
    d_pow10_table_lo,   # device array
    d_pow10_table_hi,   # device array
    # Pass1 outputs
    d_nulls_col,        # uint8_t* (NULL flags for this column)
    var_idx,            # int (-1 if not variable, >=0 if variable)
    d_var_lens,         # int32_t[:, :] (variable lengths array)
):
    """
    Pass1統合カーネル: 列ごと処理版
    
    1つのDecimal列に対して以下を実行:
    1. NULL判定
    2. 可変長列の長さ記録（該当する場合）  
    3. Decimalバイナリ解析 + 128ビット変換
    4. 出力バッファに直接書き込み
    """
    row = cuda.grid(1)
    rows = field_offsets.shape[0]
    if row >= rows:
        return

    src_offset = field_offsets[row]
    flen = field_lengths[row]
    dst_byte_offset = row * stride

    # NULL判定
    is_null = (flen == -1)
    if is_null:
        d_nulls_col[row] = uint8(0)  # NULL
        # 可変長列の場合は長さ0を記録
        if var_idx != -1:
            d_var_lens[var_idx, row] = 0
        # NULLの場合は16バイトゼロで埋める
        for i in range(16):
            decimal_dst_buf[dst_byte_offset + i] = uint8(0)
        return
    else:
        d_nulls_col[row] = uint8(1)  # NOT NULL
        # 可変長列の場合は実際の長さを記録
        if var_idx != -1:
            d_var_lens[var_idx, row] = flen

    # --- Decimal処理開始 ---
    
    # バリデーション
    if src_offset == 0 or src_offset + 8 > raw.size:
        # 無効データの場合はゼロで埋める
        for i in range(16):
            decimal_dst_buf[dst_byte_offset + i] = uint8(0)
        return

    # --- NUMERIC ヘッダ読み取り (8バイト) ---
    nd = (uint16(raw[src_offset]) << 8) | uint16(raw[src_offset + 1])
    weight = int16((int16(raw[src_offset + 2]) << 8) | int16(raw[src_offset + 3]))
    sign = (uint16(raw[src_offset + 4]) << 8) | uint16(raw[src_offset + 5])
    dscale = (uint16(raw[src_offset + 6]) << 8) | uint16(raw[src_offset + 7])
    
    current_offset = src_offset + 8

    # NaN処理
    if sign == 0xC000:
        for i in range(16):
            decimal_dst_buf[dst_byte_offset + i] = uint8(0)
        return

    # 桁数制限 (最大9桁 = 基数10000で最大38桁相当)
    if nd > 9:
        for i in range(16):
            decimal_dst_buf[dst_byte_offset + i] = uint8(0)
        return

    # --- 基数10000桁読み取り ---
    if current_offset + nd * 2 > raw.size:
        for i in range(16):
            decimal_dst_buf[dst_byte_offset + i] = uint8(0)
        return

    # 基数10000桁を配列に読み込み
    digits = cuda.local.array(9, uint16)  # 最大9桁
    for i in range(nd):
        digits[i] = (uint16(raw[current_offset]) << 8) | uint16(raw[current_offset + 1])
        current_offset += 2

    # --- 基数10000から128ビット整数への変換 ---
    val_hi = uint64(0)
    val_lo = uint64(0)
    
    # 基数1e8最適化実装（2桁ずつ結合）
    i = 0
    while i < nd:
        if i + 1 < nd:  # digitsが2つ以上残っている場合
            # digits[i] と digits[i+1] を結合して1e8の1桁として扱う
            combined_digit = uint64(digits[i]) * uint64(10000) + uint64(digits[i+1])
            
            # val = val * 1e8 + combined_digit
            val_hi, val_lo = mul128_u64_fast(val_hi, val_lo, uint64(100000000))  # 1e8
            val_hi, val_lo = add128_fast(val_hi, val_lo, uint64(0), combined_digit)
            i += 2
        else:  # digitsが1つだけ残っている場合
            # val = val * 10000 + digits[i]
            val_hi, val_lo = mul128_u64_fast(val_hi, val_lo, uint64(10000))
            val_hi, val_lo = add128_fast(val_hi, val_lo, uint64(0), uint64(digits[i]))
            i += 1

    # --- スケール統一 ---
    pg_scale = int(dscale)  # PostgreSQL側のスケール
    val_hi, val_lo = apply_scale_fast(val_hi, val_lo, pg_scale, target_scale, d_pow10_table_lo, d_pow10_table_hi)

    # --- 符号適用 ---
    if sign == 0x4000:  # 負数
        val_hi, val_lo = neg128_fast(val_hi, val_lo)

    # --- 128ビット値を出力 ---
    store_decimal128_le(val_hi, val_lo, decimal_dst_buf, dst_byte_offset)

# 従来のPass1カーネル（非Decimal列用）
@cuda.jit
def pass1_len_null_non_decimal(
    field_lengths,     # int32[:, :]
    var_indices,       # int32[:]
    d_var_lens,        # int32[:, :]
    d_nulls,           # uint8[:, :]
    decimal_cols_mask  # uint8[:] - Decimal列は1、その他は0
):
    """
    従来のPass1機能（Decimal列以外）
    
    Decimal列は除外し、NULL/長さ収集のみ実行
    """
    row = cuda.grid(1)
    rows, ncols = field_lengths.shape
    
    if row >= rows:
        return
    
    for col in range(ncols):
        # Decimal列はスキップ（別途Pass1統合カーネルで処理）
        if decimal_cols_mask[col] == 1:
            continue
            
        flen = field_lengths[row, col]
        
        # NULL判定
        if flen == -1:
            d_nulls[row, col] = uint8(0)  # NULL
        else:
            d_nulls[row, col] = uint8(1)  # NOT NULL
        
        # 可変長列の長さ記録
        var_idx = var_indices[col]
        if var_idx != -1:
            d_var_lens[var_idx, row] = 0 if flen == -1 else flen

__all__ = [
    "pass1_len_null_decimal_column_wise",
    "pass1_len_null_non_decimal"
]