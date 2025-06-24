"""
GPU Pass1 Ultimate統合カーネル: シンプル版
=========================================

デッドロック問題を回避するため、複雑なPrefix-sumを除去し、
基本的な統合処理のみを実装
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

# Decimal処理関数群（既存のものをインポート）
from .arrow_gpu_pass1_fully_integrated import (
    add128_fast, mul128_u64_fast, neg128_fast, get_pow10_128, apply_scale_fast,
    write_int16_to_buffer, write_int32_to_buffer, write_int64_to_buffer,
    write_decimal128_to_buffer, write_float32_to_buffer, write_float64_to_buffer,
    parse_decimal_from_raw, parse_int32_from_raw, parse_int16_from_raw, parse_int64_from_raw
)

@cuda.jit
def pass1_ultimate_simple(
    raw,                    # const uint8_t* (入力rawデータ)
    field_offsets,          # const int32_t[:, :] (rows, cols)
    field_lengths,          # const int32_t[:, :] (rows, cols)
    
    # 統合固定長バッファ（既存）
    unified_fixed_buffer,   # uint8_t* (全固定長列の統合バッファ)
    row_stride,             # int (1行分のバイト数)
    
    # 固定長レイアウト情報（既存）
    fixed_column_count,     # int (固定長列数)
    fixed_column_types,     # int32_t[:] (各列の型ID)
    fixed_column_offsets,   # int32_t[:] (統合バッファ内の各列オフセット)
    fixed_column_sizes,     # int32_t[:] (各列のサイズ)
    fixed_column_indices,   # int32_t[:] (元のテーブル列インデックス)
    fixed_decimal_scales,   # int32_t[:] (Decimal列のスケール情報)
    
    # 可変長文字列バッファ（シンプル版：長さ収集のみ）
    var_column_count,       # int (可変長文字列列数)
    var_column_indices,     # int32_t[:] (可変長列の元テーブル列インデックス)
    
    # 出力: 可変長列の長さ配列
    d_var_lens,             # int32_t[:, :] (可変長列の長さ配列)
    
    # 共通出力
    d_nulls_all,            # uint8_t[:, :] (NULL フラグ)
    
    # Decimal処理用（既存）
    d_pow10_table_lo,       # uint64_t[:] (10^n テーブル下位)
    d_pow10_table_hi        # uint64_t[:] (10^n テーブル上位)
):
    """
    Pass1 Ultimate統合カーネル - シンプル版
    
    デッドロック問題を回避するため:
    1. 固定長列統合処理のみ実装
    2. 可変長列は長さ収集のみ
    3. 複雑なPrefix-sumは除去
    4. cuda.syncthreads()を最小限に
    """
    
    # グリッド情報
    row = cuda.grid(1)
    rows, total_cols = field_offsets.shape
    
    # ===== 処理1: 固定長列統合処理 =====
    if row < rows:
        # 統合バッファ内の行開始位置
        row_base_offset = row * row_stride
        
        # 固定長列の統合処理
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
                decimal_scale = fixed_decimal_scales[fixed_idx]
                val_hi, val_lo = parse_decimal_from_raw(raw, src_offset, d_pow10_table_lo, d_pow10_table_hi, decimal_scale)
                write_decimal128_to_buffer(unified_fixed_buffer, buffer_offset, val_hi, val_lo)
                
            elif col_type == TYPE_INT32:
                value = parse_int32_from_raw(raw, src_offset, field_length)
                write_int32_to_buffer(unified_fixed_buffer, buffer_offset, value)
                
            elif col_type == TYPE_INT16:
                value = parse_int16_from_raw(raw, src_offset, field_length)
                write_int16_to_buffer(unified_fixed_buffer, buffer_offset, value)
                
            elif col_type == TYPE_INT64:
                value = parse_int64_from_raw(raw, src_offset, field_length)
                write_int64_to_buffer(unified_fixed_buffer, buffer_offset, value)
                
            elif col_type == TYPE_FLOAT32:
                if field_length == 4 and src_offset + 4 <= raw.size:
                    int_value = (int32(raw[src_offset]) << 24) | \
                               (int32(raw[src_offset + 1]) << 16) | \
                               (int32(raw[src_offset + 2]) << 8) | \
                               int32(raw[src_offset + 3])
                    float_value = cuda.libdevice.int_as_float(int_value)
                    write_float32_to_buffer(unified_fixed_buffer, buffer_offset, float_value)
                else:
                    for i in range(col_size):
                        unified_fixed_buffer[buffer_offset + i] = uint8(0)
                
            elif col_type == TYPE_FLOAT64:
                if field_length == 8 and src_offset + 8 <= raw.size:
                    int_value = int64(0)
                    for i in range(8):
                        int_value = (int_value << 8) | int64(raw[src_offset + i])
                    float_value = cuda.libdevice.longlong_as_double(int_value)
                    write_float64_to_buffer(unified_fixed_buffer, buffer_offset, float_value)
                else:
                    for i in range(col_size):
                        unified_fixed_buffer[buffer_offset + i] = uint8(0)
                        
            else:
                # 未対応型: ゼロで埋める
                for i in range(col_size):
                    unified_fixed_buffer[buffer_offset + i] = uint8(0)
        
        # ===== 処理2: 可変長文字列列の長さ収集 =====
        for var_idx in range(var_column_count):
            col_idx = var_column_indices[var_idx]
            
            # フィールド情報取得
            field_length = field_lengths[row, col_idx]
            
            # NULL判定
            is_null = (field_length == -1)
            d_nulls_all[row, col_idx] = uint8(0 if is_null else 1)
            
            # 長さ記録
            d_var_lens[var_idx, row] = 0 if is_null else field_length

__all__ = ["pass1_ultimate_simple"]