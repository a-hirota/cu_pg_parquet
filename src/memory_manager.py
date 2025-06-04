"""GPU Memory Manager: 列順序ベース統合版"""

from typing import List, Dict, Any, Tuple, NamedTuple
import numpy as np
import cupy as cp
from numba import cuda

from .types import *

class ColumnLayout(NamedTuple):
    """列レイアウト情報"""
    name: str
    column_index: int
    arrow_type_id: int
    is_variable: bool
    
    # 固定長列用
    buffer_offset: int
    element_size: int
    decimal_scale: int
    
    # 可変長列用
    var_index: int

class BufferInfo(NamedTuple):
    """統合バッファの情報"""
    # 固定長バッファ
    fixed_buffer: cuda.cudadrv.devicearray.DeviceNDArray
    row_stride: int
    fixed_layouts: List[ColumnLayout]
    
    # 可変長バッファ
    var_data_buffer: cuda.cudadrv.devicearray.DeviceNDArray
    var_offset_arrays: cuda.cudadrv.devicearray.DeviceNDArray
    var_layouts: List[ColumnLayout]
    
    # 列メタデータ配列（カーネル用）
    column_types: cuda.cudadrv.devicearray.DeviceNDArray
    column_is_variable: cuda.cudadrv.devicearray.DeviceNDArray
    column_indices: cuda.cudadrv.devicearray.DeviceNDArray
    
    # 固定長列情報配列
    fixed_column_offsets: cuda.cudadrv.devicearray.DeviceNDArray
    fixed_column_sizes: cuda.cudadrv.devicearray.DeviceNDArray
    fixed_decimal_scales: cuda.cudadrv.devicearray.DeviceNDArray
    
    # 可変長列マッピング
    var_column_mapping: cuda.cudadrv.devicearray.DeviceNDArray

class GPUMemoryManager:
    """列順序ベース統合バッファ対応のGPUメモリマネージャー"""
    
    def __init__(self):
        self.device_buffers = {}
        self.buffer_info = None
        
    def get_element_size(self, arrow_id: int) -> int:
        """Arrow型IDから要素サイズを取得"""
        size_map = {
            INT16: 2, INT32: 4, INT64: 8,
            FLOAT32: 4, FLOAT64: 8,
            DECIMAL128: 16, BOOL: 1,
            DATE32: 4, TS64_US: 8
        }
        return size_map.get(arrow_id, 4)
    
    def analyze_columns(self, columns, rows: int) -> Tuple[List[ColumnLayout], List[ColumnLayout]]:
        """列分析：順序を保持しながら固定長・可変長を分類"""
        fixed_layouts = []
        var_layouts = []
        
        current_fixed_offset = 0
        var_index = 0
        
        for col_idx, col in enumerate(columns):
            if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY):
                layout = ColumnLayout(
                    name=col.name,
                    column_index=col_idx,
                    arrow_type_id=col.arrow_id,
                    is_variable=True,
                    buffer_offset=-1,
                    element_size=0,
                    decimal_scale=0,
                    var_index=var_index
                )
                var_layouts.append(layout)
                var_index += 1
            else:
                element_size = self.get_element_size(col.arrow_id)
                
                decimal_scale = 0
                if col.arrow_id == DECIMAL128:
                    precision, scale = col.arrow_param or (38, 0)
                    if precision == 0:
                        precision, scale = 38, 0
                    decimal_scale = scale
                
                layout = ColumnLayout(
                    name=col.name,
                    column_index=col_idx,
                    arrow_type_id=col.arrow_id,
                    is_variable=False,
                    buffer_offset=current_fixed_offset,
                    element_size=element_size,
                    decimal_scale=decimal_scale,
                    var_index=-1
                )
                fixed_layouts.append(layout)
                current_fixed_offset += element_size
        
        return fixed_layouts, var_layouts
    
    def create_column_metadata_arrays(self, columns) -> Tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,
        cuda.cudadrv.devicearray.DeviceNDArray,
        cuda.cudadrv.devicearray.DeviceNDArray
    ]:
        """カーネル用の列メタデータ配列を作成"""
        total_cols = len(columns)
        
        column_types = np.array([col.arrow_id for col in columns], dtype=np.int32)
        column_is_variable = np.array([1 if col.is_variable else 0 for col in columns], dtype=np.uint8)
        column_indices = np.array(list(range(total_cols)), dtype=np.int32)
        
        return (cuda.to_device(column_types),
                cuda.to_device(column_is_variable),
                cuda.to_device(column_indices))
    
    def create_fixed_metadata_arrays(self, fixed_layouts: List[ColumnLayout]) -> Tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,
        cuda.cudadrv.devicearray.DeviceNDArray,
        cuda.cudadrv.devicearray.DeviceNDArray
    ]:
        """固定長列用のメタデータ配列を作成"""
        if not fixed_layouts:
            empty_array = np.array([], dtype=np.int32)
            return (cuda.to_device(empty_array), 
                   cuda.to_device(empty_array), 
                   cuda.to_device(empty_array))
        
        fixed_offsets = np.array([layout.buffer_offset for layout in fixed_layouts], dtype=np.int32)
        fixed_sizes = np.array([layout.element_size for layout in fixed_layouts], dtype=np.int32)
        fixed_scales = np.array([layout.decimal_scale for layout in fixed_layouts], dtype=np.int32)
        
        return (cuda.to_device(fixed_offsets),
               cuda.to_device(fixed_sizes),
               cuda.to_device(fixed_scales))
    
    def create_var_column_mapping(self, columns, var_layouts: List[ColumnLayout]) -> cuda.cudadrv.devicearray.DeviceNDArray:
        """可変長列のマッピング配列を作成"""
        total_cols = len(columns)
        mapping = np.full(total_cols, -1, dtype=np.int32)
        
        for var_layout in var_layouts:
            mapping[var_layout.column_index] = var_layout.var_index
        
        return cuda.to_device(mapping)
    
    def estimate_variable_total_size(self, var_layouts: List[ColumnLayout], rows: int) -> int:
        """可変長データ総サイズ推定"""
        if not var_layouts:
            return 1
        
        avg_string_length = 50
        safety_factor = 2.0
        
        total_estimated = len(var_layouts) * rows * avg_string_length * safety_factor
        return int(total_estimated)
    
    def create_fixed_buffer(self, fixed_layouts: List[ColumnLayout], rows: int) -> Tuple[cuda.cudadrv.devicearray.DeviceNDArray, int]:
        """固定長統合バッファを作成"""
        if not fixed_layouts:
            return cuda.device_array(1, dtype=np.uint8), 0
        
        row_stride = sum(layout.element_size for layout in fixed_layouts)
        total_bytes = rows * row_stride
        fixed_buffer = cuda.device_array(total_bytes, dtype=np.uint8)
        
        return fixed_buffer, row_stride
    
    def create_variable_buffers(self, var_layouts: List[ColumnLayout], rows: int) -> Tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,
        cuda.cudadrv.devicearray.DeviceNDArray
    ]:
        """可変長統合バッファシステムを作成"""
        
        if not var_layouts:
            dummy_buffer = cuda.device_array(1, dtype=np.uint8)
            dummy_offsets = cuda.device_array((1, rows + 1), dtype=np.int32)
            return dummy_buffer, dummy_offsets
        
        total_size = self.estimate_variable_total_size(var_layouts, rows)
        var_data_buffer = cuda.device_array(total_size, dtype=np.uint8)
        
        var_count = len(var_layouts)
        var_offset_arrays = cuda.device_array((var_count, rows + 1), dtype=np.int32)
        
        @cuda.jit
        def init_var_offsets(offset_arrays, var_count, rows_plus_one):
            var_idx = cuda.grid(1)
            if var_idx < var_count:
                offset_arrays[var_idx, 0] = 0
        
        init_threads = min(256, var_count)
        init_blocks = max(64, (var_count + init_threads - 1) // init_threads)
        
        if var_count > 0:
            init_var_offsets[init_blocks, init_threads](var_offset_arrays, var_count, rows + 1)
            cuda.synchronize()
        
        return var_data_buffer, var_offset_arrays
    
    def initialize_buffers(self, columns, rows: int) -> BufferInfo:
        """統合バッファシステムの初期化"""
        
        fixed_layouts, var_layouts = self.analyze_columns(columns, rows)
        
        d_column_types, d_column_is_variable, d_column_indices = \
            self.create_column_metadata_arrays(columns)
        
        d_fixed_offsets, d_fixed_sizes, d_fixed_scales = \
            self.create_fixed_metadata_arrays(fixed_layouts)
        
        d_var_mapping = self.create_var_column_mapping(columns, var_layouts)
        
        fixed_buffer, row_stride = self.create_fixed_buffer(fixed_layouts, rows)
        var_data_buffer, var_offset_arrays = self.create_variable_buffers(var_layouts, rows)
        
        self.buffer_info = BufferInfo(
            fixed_buffer=fixed_buffer,
            row_stride=row_stride,
            fixed_layouts=fixed_layouts,
            var_data_buffer=var_data_buffer,
            var_offset_arrays=var_offset_arrays,
            var_layouts=var_layouts,
            column_types=d_column_types,
            column_is_variable=d_column_is_variable,
            column_indices=d_column_indices,
            fixed_column_offsets=d_fixed_offsets,
            fixed_column_sizes=d_fixed_sizes,
            fixed_decimal_scales=d_fixed_scales,
            var_column_mapping=d_var_mapping
        )
        
        return self.buffer_info
    
    def extract_fixed_column_arrays(self, buffer_info: BufferInfo, rows: int) -> Dict[str, cuda.cudadrv.devicearray.DeviceNDArray]:
        """固定長統合バッファから個別列バッファを抽出"""
        extracted_buffers = {}
        
        for layout in buffer_info.fixed_layouts:
            column_buffer = cuda.device_array(rows * layout.element_size, dtype=np.uint8)
            
            @cuda.jit
            def extract_column(unified_buf, row_stride, col_offset, col_size, output_buf, num_rows):
                row = cuda.grid(1)
                if row >= num_rows:
                    return
                
                src_base = row * row_stride + col_offset
                dest_base = row * col_size
                
                for i in range(col_size):
                    if src_base + i < unified_buf.size and dest_base + i < output_buf.size:
                        output_buf[dest_base + i] = unified_buf[src_base + i]
            
            threads = 256
            blocks = max(64, (rows + threads - 1) // threads)
            
            extract_column[blocks, threads](
                buffer_info.fixed_buffer, buffer_info.row_stride,
                layout.buffer_offset, layout.element_size,
                column_buffer, rows
            )
            cuda.synchronize()
            
            extracted_buffers[layout.name] = column_buffer
        
        return extracted_buffers

__all__ = ["GPUMemoryManager", "BufferInfo"]