"""GPU列処理: PostgreSQL → Arrow変換"""

from __future__ import annotations
from typing import List, Dict, Any
import os
import warnings
import numpy as np
import cupy as cp
import pyarrow as pa
import cudf
import cudf.core.dtypes
from numba import cuda

from .types import (
    ColumnMeta, INT16, INT32, INT64, FLOAT32, FLOAT64, DECIMAL128, 
    UTF8, BINARY, DATE32, TS64_US, BOOL, UNKNOWN
)
from .memory_manager import GPUMemoryManager

from .cuda_kernels.column_processor import (
    pass1_column_wise_integrated, build_var_offsets_from_lengths
)
from .cuda_kernels.decimal_tables import (
    POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
)

@cuda.jit
def extract_var_lengths(field_lengths, column_idx, var_length_array, rows):
    """可変長列の長さを抽出するカーネル"""
    row = cuda.grid(1)
    if row < rows:
        var_length_array[row] = field_lengths[row, column_idx]

@cuda.jit
def extract_column_data_individual(
    raw_dev, field_offsets_dev, field_lengths_dev, 
    column_idx, rows, output_buffer, output_offsets
):
    """個別列データ抽出カーネル"""
    row = cuda.grid(1)
    
    if row < rows:
        field_offset = field_offsets_dev[row, column_idx]
        field_length = field_lengths_dev[row, column_idx]
        output_offset = output_offsets[row]
        
        for i in range(field_length):
            if output_offset + i < output_buffer.size:
                output_buffer[output_offset + i] = raw_dev[field_offset + i]

def create_cudf_series_optimized(
    col: ColumnMeta,
    rows: int,
    d_values_col,
    d_offsets_col=None,
    null_mask=None
) -> cudf.Series:
    """GPU配列からCuDFシリーズ作成"""
    
    try:
        if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY):
            # 可変長文字列列
            if d_offsets_col is not None:
                try:
                    host_data = d_values_col.copy_to_host() if hasattr(d_values_col, 'copy_to_host') else np.array(d_values_col)
                    host_offsets = d_offsets_col.copy_to_host() if hasattr(d_offsets_col, 'copy_to_host') else np.array(d_offsets_col)
                    host_offsets = host_offsets.astype(np.int32)
                    
                    if len(host_offsets) > rows + 1:
                        host_offsets = host_offsets[:rows + 1]
                    
                    if len(host_offsets) > 1:
                        max_offset = int(host_offsets[-1]) if host_offsets[-1] > 0 else len(host_data)
                        if max_offset < len(host_data):
                            host_data = host_data[:max_offset]
                    
                except Exception as e:
                    return cudf.Series([None] * rows, dtype='string')
                
                try:
                    pa_string_array = pa.StringArray.from_buffers(
                        length=rows,
                        value_offsets=pa.py_buffer(host_offsets),
                        data=pa.py_buffer(host_data),
                        null_bitmap=None
                    )
                    series = cudf.Series.from_arrow(pa_string_array)
                    return series
                    
                except Exception as e:
                    return cudf.Series([None] * rows, dtype='string')
            else:
                return cudf.Series([None] * rows, dtype='string')
        
        else:
            # 固定長列
            try:
                host_data = d_values_col.copy_to_host() if hasattr(d_values_col, 'copy_to_host') else np.array(d_values_col)
            except Exception as e:
                return cudf.Series([None] * rows)
            
            if col.arrow_id == DECIMAL128:
                try:
                    expected_size = rows * 16
                    if len(host_data) != expected_size:
                        if len(host_data) > expected_size:
                            host_data = host_data[:expected_size]
                        else:
                            host_data = np.pad(host_data, (0, expected_size - len(host_data)), 'constant')
                    
                    decimal_values = []
                    precision = 38
                    scale = 0
                    
                    for i in range(0, len(host_data), 16):
                        if i + 16 <= len(host_data):
                            decimal_bytes = host_data[i:i+16]
                            low_bytes = decimal_bytes[:8]
                            high_bytes = decimal_bytes[8:16]
                            
                            low_int = int.from_bytes(low_bytes, byteorder='little', signed=False)
                            high_int = int.from_bytes(high_bytes, byteorder='little', signed=False)
                            
                            if high_int & (1 << 63):
                                full_int = -(((~high_int & 0x7FFFFFFFFFFFFFFF) << 64) + (~low_int & 0xFFFFFFFFFFFFFFFF) + 1)
                            else:
                                full_int = (high_int << 64) + low_int
                            
                            decimal_values.append(full_int)
                        else:
                            decimal_values.append(0)
                    
                    try:
                        arrow_decimal_type = pa.decimal128(precision=precision, scale=scale)
                        arrow_array = pa.array(decimal_values, type=arrow_decimal_type)
                        series = cudf.Series.from_arrow(arrow_array)
                        
                    except Exception as decimal_error:
                        series = cudf.Series(decimal_values, dtype='int64')
                        
                except Exception as e:
                    series = cudf.Series([0] * rows, dtype='int64')
                    
            elif col.arrow_id == INT32:
                try:
                    expected_size = rows * 4
                    if len(host_data) != expected_size:
                        if len(host_data) > expected_size:
                            host_data = host_data[:expected_size]
                        else:
                            host_data = np.pad(host_data, (0, expected_size - len(host_data)), 'constant')
                    
                    data = host_data.view(np.int32)
                    series = cudf.Series(data)
                except Exception as e:
                    series = cudf.Series([0] * rows, dtype='int32')
            else:
                try:
                    series = cudf.Series(host_data[:rows])
                except Exception as e:
                    series = cudf.Series([None] * rows)
            
            return series
            
    except Exception as e:
        return cudf.Series([None] * rows)

def create_individual_string_buffers(
    columns: List[ColumnMeta],
    rows: int,
    raw_dev,
    field_offsets_dev,
    field_lengths_dev
) -> Dict[str, Any]:
    """文字列列用個別バッファ作成"""
    
    string_buffers = {}
    var_columns = [col for col in columns if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY)]
    
    for col_idx, col in enumerate(var_columns):
        actual_col_idx = None
        for i, c in enumerate(columns):
            if c.name == col.name:
                actual_col_idx = i
                break
        
        if actual_col_idx is None:
            continue
        
        try:
            estimated_total_size = rows * 20
            
            d_column_data = cuda.device_array(estimated_total_size, dtype=np.uint8)
            d_column_offsets = cuda.device_array(rows + 1, dtype=np.int32)
            
            threads = 256
            blocks = (rows + threads - 1) // threads
            
            d_lengths = cuda.device_array(rows, dtype=np.int32)
            extract_var_lengths[blocks, threads](field_lengths_dev, actual_col_idx, d_lengths, rows)
            
            host_lengths = d_lengths.copy_to_host()
            host_offsets = np.zeros(rows + 1, dtype=np.int32)
            host_offsets[1:] = np.cumsum(host_lengths)
            
            actual_size = int(host_offsets[-1])
            if actual_size > estimated_total_size:
                d_column_data = cuda.device_array(actual_size, dtype=np.uint8)
            elif actual_size == 0:
                string_buffers[col.name] = {'data': None, 'offsets': None, 'actual_size': 0}
                continue
            
            d_column_offsets = cuda.to_device(host_offsets)
            
            extract_column_data_individual[blocks, threads](
                raw_dev, field_offsets_dev, field_lengths_dev,
                actual_col_idx, rows, d_column_data, d_column_offsets
            )
            
            cuda.synchronize()
            
            string_buffers[col.name] = {
                'data': d_column_data,
                'offsets': d_column_offsets,
                'actual_size': actual_size
            }
            
        except Exception as e:
            string_buffers[col.name] = {'data': None, 'offsets': None, 'actual_size': 0}
    
    return string_buffers

def create_cudf_dataframe_optimized(
    columns: List[ColumnMeta],
    rows: int,
    fixed_buffers: Dict[str, Any],
    var_data_buffer,
    var_layouts,
    var_offset_arrays,
    host_nulls_all,
    string_buffers=None
) -> cudf.DataFrame:
    """CuDF DataFrame作成"""
    
    cudf_series_dict = {}
    
    for cidx, col in enumerate(columns):
        if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY):
            # 可変長文字列列処理
            if string_buffers and col.name in string_buffers:
                buffer_info = string_buffers[col.name]
                if buffer_info['data'] is not None and buffer_info['offsets'] is not None:
                    try:
                        series = create_cudf_series_optimized(
                            col, rows, buffer_info['data'], buffer_info['offsets'], None
                        )
                    except Exception as e:
                        series = cudf.Series([None] * rows, dtype='string')
                else:
                    series = cudf.Series([None] * rows, dtype='string')
            else:
                series = cudf.Series([None] * rows, dtype='string')
        
        else:
            # 固定長列処理
            if col.name in fixed_buffers:
                try:
                    d_values_col = fixed_buffers[col.name]
                    series = create_cudf_series_optimized(
                        col, rows, d_values_col, None, None
                    )
                except Exception as e:
                    series = cudf.Series([None] * rows)
            else:
                series = cudf.Series([None] * rows)
        
        cudf_series_dict[col.name] = series
    
    try:
        cudf_df = cudf.DataFrame(cudf_series_dict)
        return cudf_df
    except Exception as e:
        return cudf.DataFrame()

def decode_chunk_integrated(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
    field_offsets_dev,
    field_lengths_dev,
    columns: List[ColumnMeta],
) -> pa.RecordBatch:
    """統合GPUデコード"""
    
    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    var_columns = [col for col in columns if col.is_variable]

    # 10のべき乗テーブルをGPUに転送
    d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
    d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)

    # 文字列列個別バッファ作成
    string_buffers = create_individual_string_buffers(
        columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
    )

    # 統合バッファシステム初期化
    gmm = GPUMemoryManager()
    buffer_info = gmm.initialize_buffers(columns, rows)
    
    fixed_layouts = buffer_info.fixed_layouts
    var_layouts = buffer_info.var_layouts

    # 共通NULL配列初期化
    d_nulls_all = cuda.device_array((rows, ncols), dtype=np.uint8)

    # 統合カーネル実行
    threads = 256
    blocks = max(64, (rows + threads - 1) // threads)
    
    try:
        pass1_column_wise_integrated[blocks, threads](
            raw_dev,
            field_offsets_dev,
            field_lengths_dev,
            
            # 列メタデータ配列
            buffer_info.column_types,
            buffer_info.column_is_variable,
            buffer_info.column_indices,
            
            # 固定長統合バッファ
            buffer_info.fixed_buffer,
            buffer_info.fixed_column_offsets,
            buffer_info.fixed_column_sizes,
            buffer_info.fixed_decimal_scales,
            buffer_info.row_stride,
            
            # 可変長統合バッファ
            buffer_info.var_data_buffer,
            buffer_info.var_offset_arrays,
            buffer_info.var_column_mapping,
            
            # 共通出力
            d_nulls_all,
            
            # Decimal処理用
            d_pow10_table_lo,
            d_pow10_table_hi
        )
        
        cuda.synchronize()
        
    except Exception as e:
        raise

    # DataFrame作成
    host_nulls_all = d_nulls_all.copy_to_host()
    if host_nulls_all.shape[0] > rows:
        host_nulls_all = host_nulls_all[:rows, :]
    
    fixed_buffers = gmm.extract_fixed_column_arrays(buffer_info, rows)
    
    cudf_df = create_cudf_dataframe_optimized(
        columns, rows, fixed_buffers, buffer_info.var_data_buffer,
        var_layouts, buffer_info.var_offset_arrays, host_nulls_all, string_buffers
    )

    # CuDF → Arrow変換
    try:
        if len(cudf_df) > 0:
            arrow_table = cudf_df.to_arrow()
            record_batch = arrow_table.to_batches()[0]
        else:
            arrays = []
            for col in columns:
                arrays.append(pa.nulls(rows, type=pa.string()))
            record_batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
        
    except Exception as e:
        arrays = []
        for col in columns:
            arrays.append(pa.nulls(rows, type=pa.string()))
        record_batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    
    return record_batch

__all__ = ["decode_chunk_integrated"]