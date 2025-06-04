"""
GPU Memory Manager V7: 列順序ベース統合版（Grid size最適化）
========================================

V7カーネル専用の最適化されたメモリ管理システム:
1. 列順序処理対応
2. 段階的メモリ確保
3. 可変長データの動的サイズ調整
4. ブロック内Prefix-sum対応
5. Grid size完全最適化（GPU並列性確保）
"""

from typing import List, Dict, Any, Tuple, NamedTuple
import numpy as np
import cupy as cp
from numba import cuda

from .type_map import *

class ColumnLayoutV7(NamedTuple):
    """V7用の列レイアウト情報"""
    name: str
    column_index: int        # 元のテーブル列インデックス
    arrow_type_id: int       # Arrow型ID
    is_variable: bool        # 可変長フラグ
    
    # 固定長列用
    buffer_offset: int       # 統合バッファ内のオフセット（固定長のみ）
    element_size: int        # 要素サイズ（固定長のみ）
    decimal_scale: int       # Decimal列のスケール（固定長のみ）
    
    # 可変長列用
    var_index: int          # 可変長列内のインデックス（可変長のみ）

class V7BufferInfo(NamedTuple):
    """V7統合バッファの情報"""
    # 固定長バッファ
    fixed_buffer: cuda.cudadrv.devicearray.DeviceNDArray
    row_stride: int
    fixed_layouts: List[ColumnLayoutV7]
    
    # 可変長バッファ
    var_data_buffer: cuda.cudadrv.devicearray.DeviceNDArray
    var_offset_arrays: cuda.cudadrv.devicearray.DeviceNDArray  # 2D配列
    var_layouts: List[ColumnLayoutV7]
    
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

class GPUMemoryManagerV7ColumnWise:
    """V7列順序ベース統合バッファ対応のGPUメモリマネージャー（Grid size最適化版）"""
    
    def __init__(self):
        self.device_buffers = {}
        self.v7_buffer_info = None
        
    def get_element_size(self, arrow_id: int) -> int:
        """Arrow型IDから要素サイズを取得"""
        size_map = {
            INT16: 2,
            INT32: 4, 
            INT64: 8,
            FLOAT32: 4,
            FLOAT64: 8,
            DECIMAL128: 16,
            BOOL: 1,
            DATE32: 4,
            TS64_US: 8
        }
        return size_map.get(arrow_id, 4)
    
    def analyze_columns_v7(self, columns, rows: int) -> Tuple[List[ColumnLayoutV7], List[ColumnLayoutV7]]:
        """V7用の列分析：順序を保持しながら固定長・可変長を分類"""
        fixed_layouts = []
        var_layouts = []
        
        current_fixed_offset = 0
        var_index = 0
        fixed_index = 0
        
        for col_idx, col in enumerate(columns):
            if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY):
                # 可変長文字列列
                layout = ColumnLayoutV7(
                    name=col.name,
                    column_index=col_idx,
                    arrow_type_id=col.arrow_id,
                    is_variable=True,
                    buffer_offset=-1,  # 可変長では未使用
                    element_size=0,    # 可変長では未使用
                    decimal_scale=0,   # 可変長では未使用
                    var_index=var_index
                )
                var_layouts.append(layout)
                var_index += 1
            else:
                # 固定長列
                element_size = self.get_element_size(col.arrow_id)
                
                # Decimal列のスケール取得
                decimal_scale = 0
                if col.arrow_id == DECIMAL128:
                    precision, scale = col.arrow_param or (38, 0)
                    if precision == 0:
                        precision, scale = 38, 0
                    decimal_scale = scale
                
                layout = ColumnLayoutV7(
                    name=col.name,
                    column_index=col_idx,
                    arrow_type_id=col.arrow_id,
                    is_variable=False,
                    buffer_offset=current_fixed_offset,
                    element_size=element_size,
                    decimal_scale=decimal_scale,
                    var_index=-1  # 固定長では未使用
                )
                fixed_layouts.append(layout)
                current_fixed_offset += element_size
                fixed_index += 1
        
        return fixed_layouts, var_layouts
    
    def create_column_metadata_arrays(self, columns) -> Tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,  # column_types
        cuda.cudadrv.devicearray.DeviceNDArray,  # column_is_variable  
        cuda.cudadrv.devicearray.DeviceNDArray   # column_indices
    ]:
        """カーネル用の列メタデータ配列を作成"""
        total_cols = len(columns)
        
        column_types = np.array([col.arrow_id for col in columns], dtype=np.int32)
        column_is_variable = np.array([1 if col.is_variable else 0 for col in columns], dtype=np.uint8)
        column_indices = np.array(list(range(total_cols)), dtype=np.int32)
        
        d_column_types = cuda.to_device(column_types)
        d_column_is_variable = cuda.to_device(column_is_variable)
        d_column_indices = cuda.to_device(column_indices)
        
        return d_column_types, d_column_is_variable, d_column_indices
    
    def create_fixed_metadata_arrays(self, fixed_layouts: List[ColumnLayoutV7]) -> Tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,  # fixed_column_offsets
        cuda.cudadrv.devicearray.DeviceNDArray,  # fixed_column_sizes
        cuda.cudadrv.devicearray.DeviceNDArray   # fixed_decimal_scales
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
    
    def create_var_column_mapping(self, columns, var_layouts: List[ColumnLayoutV7]) -> cuda.cudadrv.devicearray.DeviceNDArray:
        """可変長列のマッピング配列を作成（列インデックス→可変長インデックス）"""
        total_cols = len(columns)
        mapping = np.full(total_cols, -1, dtype=np.int32)  # デフォルト-1（非可変長）
        
        for var_layout in var_layouts:
            mapping[var_layout.column_index] = var_layout.var_index
        
        return cuda.to_device(mapping)
    
    def estimate_variable_total_size_v7(self, var_layouts: List[ColumnLayoutV7], rows: int) -> int:
        """V7用可変長データ総サイズ推定（ブロック内Prefix-sumを考慮）"""
        if not var_layouts:
            return 1
        
        # ブロックサイズを考慮した推定
        avg_string_length = 50
        safety_factor = 2.0  # ブロック内処理のため少し大きめ
        
        total_estimated = len(var_layouts) * rows * avg_string_length * safety_factor
        return int(total_estimated)
    
    def create_fixed_buffer_v7(self, fixed_layouts: List[ColumnLayoutV7], rows: int) -> Tuple[cuda.cudadrv.devicearray.DeviceNDArray, int]:
        """V7用固定長統合バッファを作成"""
        if not fixed_layouts:
            return cuda.device_array(1, dtype=np.uint8), 0
        
        # 1行分のバイト数計算
        row_stride = sum(layout.element_size for layout in fixed_layouts)
        
        # 統合バッファ確保
        total_bytes = rows * row_stride
        fixed_buffer = cuda.device_array(total_bytes, dtype=np.uint8)
        
        print(f"V7固定長統合バッファ作成: {len(fixed_layouts)}列, 行ストライド={row_stride}バイト, 総サイズ={total_bytes}バイト")
        
        return fixed_buffer, row_stride
    
    def create_variable_buffers_v7(self, var_layouts: List[ColumnLayoutV7], rows: int) -> Tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,  # 統合データバッファ
        cuda.cudadrv.devicearray.DeviceNDArray   # オフセット配列（2D）
    ]:
        """V7用可変長統合バッファシステムを作成（Grid size最適化版）"""
        
        if not var_layouts:
            dummy_buffer = cuda.device_array(1, dtype=np.uint8)
            dummy_offsets = cuda.device_array((1, rows + 1), dtype=np.int32)
            return dummy_buffer, dummy_offsets
        
        # 統合データバッファ
        total_size = self.estimate_variable_total_size_v7(var_layouts, rows)
        var_data_buffer = cuda.device_array(total_size, dtype=np.uint8)
        
        # 2Dオフセット配列（var_count × (rows + 1)）
        var_count = len(var_layouts)
        var_offset_arrays = cuda.device_array((var_count, rows + 1), dtype=np.int32)
        
        # 【重要修正】初期化カーネルのGrid size最適化
        @cuda.jit
        def init_var_offsets_optimized(offset_arrays, var_count, rows_plus_one):
            var_idx = cuda.grid(1)
            if var_idx < var_count:
                offset_arrays[var_idx, 0] = 0
        
        # Grid size最適化（最小64ブロック保証）
        init_threads = min(256, var_count)
        init_blocks = max(64, (var_count + init_threads - 1) // init_threads)
        print(f"   🔧 可変長初期化Grid最適化: var_count={var_count} → blocks={init_blocks}, threads={init_threads}")
        
        if var_count > 0:
            init_var_offsets_optimized[init_blocks, init_threads](var_offset_arrays, var_count, rows + 1)
            cuda.synchronize()
            print(f"   ✅ 可変長初期化完了: {init_blocks}ブロック並列実行")
        
        print(f"V7可変長統合バッファ作成: {var_count}列, 統合データサイズ={total_size}バイト")
        
        return var_data_buffer, var_offset_arrays
    
    def initialize_v7_buffers(self, columns, rows: int) -> V7BufferInfo:
        """V7統合バッファシステムの初期化（Grid size最適化版）"""
        
        print(f"\n=== V7列順序ベース統合バッファ初期化（Grid size最適化） ===")
        print(f"総列数: {len(columns)}, 総行数: {rows:,}")
        
        # 列分析
        fixed_layouts, var_layouts = self.analyze_columns_v7(columns, rows)
        
        print(f"固定長列: {len(fixed_layouts)}列")
        print(f"可変長列: {len(var_layouts)}列")
        
        # 列メタデータ配列作成
        d_column_types, d_column_is_variable, d_column_indices = \
            self.create_column_metadata_arrays(columns)
        
        # 固定長列情報配列作成
        d_fixed_offsets, d_fixed_sizes, d_fixed_scales = \
            self.create_fixed_metadata_arrays(fixed_layouts)
        
        # 可変長列マッピング作成
        d_var_mapping = self.create_var_column_mapping(columns, var_layouts)
        
        # バッファ作成（Grid size最適化版）
        fixed_buffer, row_stride = self.create_fixed_buffer_v7(fixed_layouts, rows)
        var_data_buffer, var_offset_arrays = self.create_variable_buffers_v7(var_layouts, rows)
        
        # V7統合バッファ情報
        self.v7_buffer_info = V7BufferInfo(
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
        
        print("V7統合バッファ初期化完了（Grid size最適化）")
        return self.v7_buffer_info
    
    def extract_fixed_column_arrays_v7(self, v7_info: V7BufferInfo, rows: int) -> Dict[str, cuda.cudadrv.devicearray.DeviceNDArray]:
        """V7固定長統合バッファから個別列バッファを抽出（Grid size最適化版）"""
        extracted_buffers = {}
        
        for layout in v7_info.fixed_layouts:
            # GPU上でデータ抽出
            column_buffer = cuda.device_array(rows * layout.element_size, dtype=np.uint8)
            
            @cuda.jit
            def extract_v7_column_optimized(unified_buf, row_stride, col_offset, col_size, output_buf, num_rows):
                row = cuda.grid(1)
                if row >= num_rows:
                    return
                
                src_base = row * row_stride + col_offset
                dest_base = row * col_size
                
                for i in range(col_size):
                    if src_base + i < unified_buf.size and dest_base + i < output_buf.size:
                        output_buf[dest_base + i] = unified_buf[src_base + i]
            
            # 【重要修正】Grid size最適化（最小64ブロック保証）
            threads = 256
            blocks = max(64, (rows + threads - 1) // threads)
            print(f"   🔧 列抽出Grid最適化 {layout.name}: rows={rows} → blocks={blocks}, threads={threads}")
            
            extract_v7_column_optimized[blocks, threads](
                v7_info.fixed_buffer, v7_info.row_stride,
                layout.buffer_offset, layout.element_size,
                column_buffer, rows
            )
            cuda.synchronize()
            print(f"   ✅ 列抽出完了 {layout.name}: {blocks}ブロック並列実行")
            
            extracted_buffers[layout.name] = column_buffer
        
        return extracted_buffers

__all__ = ["GPUMemoryManagerV7ColumnWise", "V7BufferInfo"]