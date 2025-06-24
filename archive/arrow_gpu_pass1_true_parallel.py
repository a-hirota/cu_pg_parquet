"""
GPU Pass1 真の並列統合カーネル
============================

課題：現在の実装はPrefix-sum後に逐次コピー
解決：Pass1でPrefix-sum済みオフセット使用して直接並列書き込み
"""

import numpy as np
from numba import cuda, uint64, int64, uint16, int16, uint8, boolean, int32, uint32
import math

# 型定数
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

# 既存のDecimal処理関数をインポート
from .arrow_gpu_pass1_fully_integrated import (
    add128_fast, mul128_u64_fast, neg128_fast, get_pow10_128, apply_scale_fast,
    write_int16_to_buffer, write_int32_to_buffer, write_int64_to_buffer,
    write_decimal128_to_buffer, write_float32_to_buffer, write_float64_to_buffer,
    parse_decimal_from_raw, parse_int32_from_raw, parse_int16_from_raw, parse_int64_from_raw
)

@cuda.jit(device=True, inline=True)
def copy_string_data_parallel(
    raw_data, src_offset, length,
    dest_buffer, dest_offset
):
    """
    文字列データの並列コピー（安全版）
    各スレッドが直接目的地にコピー
    """
    if length <= 0 or src_offset < 0 or dest_offset < 0:
        return
    
    max_src = min(src_offset + length, raw_data.size)
    max_dest = min(dest_offset + length, dest_buffer.size)
    actual_length = min(length, max_src - src_offset, max_dest - dest_offset)
    
    # デバッグ情報を出力（最初の数行のみ）
    row_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if row_id < 5:  # 最初の5行のみデバッグ
        cuda.printf("DEBUG: row=%d, src_offset=%d, dest_offset=%d, length=%d, actual_length=%d\n",
                   row_id, src_offset, dest_offset, length, actual_length)
        cuda.printf("DEBUG: raw_data.size=%d, dest_buffer.size=%d\n",
                   raw_data.size, dest_buffer.size)
        
        # コピー前の最初の数バイトを表示
        if actual_length > 0 and src_offset < raw_data.size:
            cuda.printf("DEBUG: src_data[0-3]: %d %d %d %d\n",
                       raw_data[src_offset] if src_offset < raw_data.size else 255,
                       raw_data[src_offset + 1] if src_offset + 1 < raw_data.size else 255,
                       raw_data[src_offset + 2] if src_offset + 2 < raw_data.size else 255,
                       raw_data[src_offset + 3] if src_offset + 3 < raw_data.size else 255)
    
    # 単純なバイト単位コピー
    for i in range(actual_length):
        if src_offset + i < raw_data.size and dest_offset + i < dest_buffer.size:
            dest_buffer[dest_offset + i] = raw_data[src_offset + i]
    
    # デバッグ情報を出力（コピー後）
    if row_id < 5 and actual_length > 0:
        cuda.printf("DEBUG: dest_data[0-3]: %d %d %d %d\n",
                   dest_buffer[dest_offset] if dest_offset < dest_buffer.size else 255,
                   dest_buffer[dest_offset + 1] if dest_offset + 1 < dest_buffer.size else 255,
                   dest_buffer[dest_offset + 2] if dest_offset + 2 < dest_buffer.size else 255,
                   dest_buffer[dest_offset + 3] if dest_offset + 3 < dest_buffer.size else 255)

@cuda.jit
def pass1_true_parallel_kernel(
    raw,                        # const uint8_t* (入力rawデータ)
    field_offsets,              # const int32_t[:, :] (rows, cols)
    field_lengths,              # const int32_t[:, :] (rows, cols)
    
    # 統合固定長バッファ
    unified_fixed_buffer,       # uint8_t* (全固定長列の統合バッファ)
    row_stride,                 # int (1行分のバイト数)
    
    # 固定長レイアウト情報
    fixed_column_count,         # int (固定長列数)
    fixed_column_types,         # int32_t[:] (各列の型ID)
    fixed_column_offsets,       # int32_t[:] (統合バッファ内の各列オフセット)
    fixed_column_sizes,         # int32_t[:] (各列のサイズ)
    fixed_column_indices,       # int32_t[:] (元のテーブル列インデックス)
    fixed_decimal_scales,       # int32_t[:] (Decimal列のスケール情報)
    
    # 可変長文字列並列処理用（**重要な変更点**）
    var_column_count,           # int (可変長文字列列数)
    var_column_indices,         # int32_t[:] (可変長列の元テーブル列インデックス)
    var_data_buffers,           # uint8_t[:] (可変長データの統合バッファ)
    var_offset_arrays,          # int32_t[:, :] (事前計算済みPrefix-sumオフセット)
    
    # 共通出力
    d_nulls_all,                # uint8_t[:, :] (NULL フラグ)
    
    # Decimal処理用
    d_pow10_table_lo,           # uint64_t[:] (10^n テーブル下位)
    d_pow10_table_hi            # uint64_t[:] (10^n テーブル上位)
):
    """
    Pass1 真の並列統合カーネル
    
    **重要な改良**：
    1. Prefix-sumは事前にCuPyで高速計算済み
    2. 各スレッドが事前計算済みオフセット使用して直接書き込み
    3. 固定長・可変長を完全に並列処理
    """
    
    # グリッド情報
    row = cuda.grid(1)
    rows, total_cols = field_offsets.shape
    
    if row >= rows:
        return
    
    # ===== 処理1: 固定長列統合処理（並列） =====
    row_base_offset = row * row_stride
    
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
        
        # 統合バッファ書き込み位置
        buffer_offset = row_base_offset + col_offset
        
        if is_null:
            # NULLの場合はゼロで埋める
            for i in range(col_size):
                if buffer_offset + i < unified_fixed_buffer.size:
                    unified_fixed_buffer[buffer_offset + i] = uint8(0)
            continue
        
        # 型別の解析・変換・書き込み
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
                    if buffer_offset + i < unified_fixed_buffer.size:
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
                    if buffer_offset + i < unified_fixed_buffer.size:
                        unified_fixed_buffer[buffer_offset + i] = uint8(0)
        else:
            # 未対応型: ゼロで埋める
            for i in range(col_size):
                if buffer_offset + i < unified_fixed_buffer.size:
                    unified_fixed_buffer[buffer_offset + i] = uint8(0)
    
    # ===== 処理2: 可変長文字列列の並列コピー（**真の並列処理**） =====
    for var_idx in range(var_column_count):
        col_idx = var_column_indices[var_idx]
        
        # フィールド情報取得
        src_offset = field_offsets[row, col_idx]
        field_length = field_lengths[row, col_idx]
        
        # NULL判定
        is_null = (field_length == -1)
        d_nulls_all[row, col_idx] = uint8(0 if is_null else 1)
        
        # デバッグ情報出力（最初の数行のみ）
        if row < 5:
            cuda.printf("MAIN: row=%d, var_idx=%d, col_idx=%d, src_offset=%d, field_length=%d, is_null=%d\n",
                       row, var_idx, col_idx, src_offset, field_length, int(is_null))
        
        if is_null or field_length <= 0:
            continue
        
        # **重要**：事前計算済みPrefix-sumオフセットを使用
        if row < var_offset_arrays.shape[1] - 1:
            dest_offset = var_offset_arrays[var_idx, row]
            
            # デバッグ情報: オフセット配列の形状と値
            if row < 5:
                cuda.printf("MAIN: var_offset_arrays.shape=(%d,%d), dest_offset=%d\n",
                           var_offset_arrays.shape[0], var_offset_arrays.shape[1], dest_offset)
            
            # **並列文字列コピー実行**
            copy_string_data_parallel(
                raw, src_offset, field_length,
                var_data_buffers, dest_offset
            )

__all__ = ["pass1_true_parallel_kernel"]