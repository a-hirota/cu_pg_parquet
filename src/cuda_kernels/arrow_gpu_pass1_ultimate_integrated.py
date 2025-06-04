"""
GPU Pass1 Ultimate統合カーネル: Pass2完全廃止版
=============================================

革新的特徴:
1. 固定長列統合処理（Decimal最適化継承）
2. 可変長文字列長さ収集 
3. GPU並列Prefix-sum
4. 可変長文字列データコピー
5. 1回のカーネル起動で全処理完了

技術コア:
- CUDA協調スレッド
- ブロック内同期
- 並列Prefix-sum
- メモリコアレッシング最適化
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

# 文字列処理専用関数
@cuda.jit(device=True, inline=True)
def copy_string_data_optimized(raw_data, src_offset, length, dest_buffer, dest_offset):
    """最適化された文字列データコピー"""
    if length <= 0 or src_offset + length > raw_data.size or dest_offset + length > dest_buffer.size:
        return
    
    # 単純なバイト単位コピー（安全性優先）
    for i in range(length):
        dest_buffer[dest_offset + i] = raw_data[src_offset + i]

@cuda.jit(device=True, inline=True)
def block_prefix_sum_simple(shared_data, tid, block_size):
    """簡単なブロック内Prefix-sum"""
    
    # Up-sweep (reduction)
    step = 1
    while step < block_size:
        if tid >= step and (tid + 1) % (step * 2) == 0:
            shared_data[tid] += shared_data[tid - step]
        
        cuda.syncthreads()
        step *= 2
    
    # Clear last element for exclusive scan
    if tid == block_size - 1:
        shared_data[tid] = 0
        
    cuda.syncthreads()
    
    # Down-sweep
    step = block_size // 2
    while step > 0:
        if tid >= step and (tid + 1) % (step * 2) == 0:
            temp = shared_data[tid - step]
            shared_data[tid - step] = shared_data[tid]
            shared_data[tid] += temp
        
        cuda.syncthreads()
        step //= 2

@cuda.jit
def pass1_ultimate_integrated(
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
    
    # 可変長文字列バッファ（新規）
    var_column_count,       # int (可変長文字列列数)
    var_column_indices,     # int32_t[:] (可変長列の元テーブル列インデックス)
    var_data_buffer,        # uint8_t[:] (統合可変長データバッファ)
    
    # 一時作業領域
    d_var_lens,             # int32_t[:, :] (可変長列の長さ配列)
    d_var_offset_arrays,    # int32_t[:, :] (各可変長列のオフセット配列)
    
    # 共通出力
    d_nulls_all,            # uint8_t[:, :] (NULL フラグ)
    
    # Decimal処理用（既存）
    d_pow10_table_lo,       # uint64_t[:] (10^n テーブル下位)
    d_pow10_table_hi        # uint64_t[:] (10^n テーブル上位)
):
    """
    Pass1 Ultimate統合カーネル - Pass2完全廃止版
    
    3段階統合処理:
    1. 固定長列処理 + 可変長長さ収集
    2. 並列Prefix-sum（可変長オフセット確定）
    3. 可変長データコピー
    """
    
    # グリッド情報
    row = cuda.grid(1)
    rows, total_cols = field_offsets.shape
    
    # 共有メモリ（ブロック内Prefix-sum用）
    shared_lengths = cuda.shared.array(512, dtype=int32)  # 最大512スレッド対応
    
    # ===== 段階1: 固定長列処理 + 可変長長さ収集 =====
    if row < rows:
        # 統合バッファ内の行開始位置
        row_base_offset = row * row_stride
        
        # 固定長列の統合処理（既存実装そのまま）
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
            
            # 型別の解析・変換・書き込み（既存実装）
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
        
        # 可変長文字列列の長さ収集
        for var_idx in range(var_column_count):
            col_idx = var_column_indices[var_idx]
            
            # フィールド情報取得
            field_length = field_lengths[row, col_idx]
            
            # NULL判定
            is_null = (field_length == -1)
            d_nulls_all[row, col_idx] = uint8(0 if is_null else 1)
            
            # 長さ記録
            d_var_lens[var_idx, row] = 0 if is_null else field_length
    
    # ===== 段階1完了: ブロック内同期 =====
    cuda.syncthreads()
    
    # ===== 段階2: 並列Prefix-sum（各可変長列） =====
    tid = cuda.threadIdx.x
    
    for var_idx in range(var_column_count):
        # ブロック内のスレッドで協調してPrefix-sum計算
        if row < rows:
            shared_lengths[tid] = d_var_lens[var_idx, row]
        else:
            shared_lengths[tid] = 0
            
        cuda.syncthreads()
        
        # ブロック内Prefix-sum実行
        block_size = min(cuda.blockDim.x, rows - cuda.blockIdx.x * cuda.blockDim.x)
        if tid < block_size:
            block_prefix_sum_simple(shared_lengths, tid, block_size)
        
        cuda.syncthreads()
        
        # 結果をオフセット配列に格納
        if row < rows:
            d_var_offset_arrays[var_idx, row + 1] = shared_lengths[tid]
            # 最初の要素は0で初期化済み
        
        cuda.syncthreads()
    
    # ===== 段階3: 可変長データコピー =====
    if row < rows:
        for var_idx in range(var_column_count):
            col_idx = var_column_indices[var_idx]
            
            # フィールド情報取得
            src_offset = field_offsets[row, col_idx]
            field_length = field_lengths[row, col_idx]
            
            # NULL でなければデータコピー
            if field_length > 0:
                dest_offset = d_var_offset_arrays[var_idx, row]
                copy_string_data_optimized(raw, src_offset, field_length, var_data_buffer, dest_offset)

__all__ = ["pass1_ultimate_integrated"]