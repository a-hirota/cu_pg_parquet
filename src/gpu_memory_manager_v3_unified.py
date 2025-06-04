"""
GPU Memory Manager V3: 統合バッファ版
===================================

固定長列を統合バッファに配置し、メモリアクセス効率を最大化
- ColumnMetaベースの自動レイアウト生成
- ブロック単位のメモリコアレッシング最適化
- 可変長列は従来方式を維持
"""

from typing import List, Dict, Any, Tuple, NamedTuple
import numpy as np
import cupy as cp
from numba import cuda

from .type_map import *

class FixedColumnLayout(NamedTuple):
    """固定長列のレイアウト情報"""
    name: str
    column_index: int      # 元のテーブル列インデックス
    buffer_offset: int     # 統合バッファ内のオフセット
    element_size: int      # 要素サイズ（バイト）
    arrow_type_id: int     # Arrow型ID
    decimal_scale: int     # Decimal列のスケール（非Decimal列は0）

class UnifiedBufferInfo(NamedTuple):
    """統合バッファの情報"""
    buffer: cuda.cudadrv.devicearray.DeviceNDArray  # 統合バッファ
    row_stride: int                                  # 1行分のバイト数
    fixed_layouts: List[FixedColumnLayout]           # 固定長列レイアウト
    var_column_indices: List[int]                    # 可変長列インデックス

class GPUMemoryManagerV3Unified:
    """統合バッファ対応のGPUメモリマネージャー"""
    
    def __init__(self):
        self.device_buffers = {}
        self.unified_buffer_info = None
        
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
        return size_map.get(arrow_id, 4)  # デフォルト4バイト
    
    def analyze_columns(self, columns, rows: int) -> Tuple[List[FixedColumnLayout], List[int]]:
        """列を固定長と可変長に分類してレイアウト生成"""
        fixed_layouts = []
        var_column_indices = []
        current_offset = 0
        
        for col_idx, col in enumerate(columns):
            if col.is_variable:
                # 可変長列（UTF8, BINARY）
                var_column_indices.append(col_idx)
            else:
                # 固定長列
                element_size = self.get_element_size(col.arrow_id)
                
                # Decimal列のスケール取得
                decimal_scale = 0
                if col.arrow_id == DECIMAL128:
                    precision, scale = col.arrow_param or (38, 0)
                    # PostgreSQL NUMERIC default (precision=0) の処理
                    if precision == 0:
                        precision, scale = 38, 0
                    decimal_scale = scale
                
                layout = FixedColumnLayout(
                    name=col.name,
                    column_index=col_idx,
                    buffer_offset=current_offset,
                    element_size=element_size,
                    arrow_type_id=col.arrow_id,
                    decimal_scale=decimal_scale
                )
                fixed_layouts.append(layout)
                current_offset += element_size
        
        return fixed_layouts, var_column_indices
    
    def create_unified_buffer(self, fixed_layouts: List[FixedColumnLayout], rows: int) -> Tuple[cuda.cudadrv.devicearray.DeviceNDArray, int]:
        """統合バッファを作成"""
        if not fixed_layouts:
            # 固定長列がない場合は空バッファ
            return cuda.device_array(1, dtype=np.uint8), 0
        
        # 1行分のバイト数計算
        row_stride = sum(layout.element_size for layout in fixed_layouts)
        
        # 統合バッファ確保（行数 × 行ストライド）
        total_bytes = rows * row_stride
        unified_buffer = cuda.device_array(total_bytes, dtype=np.uint8)
        
        print(f"統合バッファ作成: {len(fixed_layouts)}列, 行ストライド={row_stride}バイト, 総サイズ={total_bytes}バイト")
        
        return unified_buffer, row_stride
    
    def create_variable_buffers(self, columns, var_column_indices: List[int], rows: int) -> Dict[str, Any]:
        """可変長列の個別バッファ作成（従来方式）"""
        var_buffers = {}
        
        for var_idx, col_idx in enumerate(var_column_indices):
            col = columns[col_idx]
            
            # 初期推定サイズ
            estimated_avg_length = 50  # 平均文字列長の推定
            estimated_total = rows * estimated_avg_length
            
            # データバッファ
            d_values = cuda.device_array(estimated_total, dtype=np.uint8)
            
            # NULLバッファ（統合NULL処理があるため使わないが互換性のため）
            d_nulls = cuda.device_array(rows, dtype=np.uint8)
            
            # オフセットバッファ（rows + 1）
            d_offsets = cuda.device_array(rows + 1, dtype=np.int32)
            
            var_buffers[col.name] = (d_values, d_nulls, d_offsets, estimated_total)
            print(f"可変長バッファ作成: {col.name}, 推定サイズ={estimated_total}バイト")
        
        return var_buffers
    
    def initialize_unified_buffers(self, columns, rows: int) -> UnifiedBufferInfo:
        """統合バッファシステムの初期化"""
        
        # 列分析
        fixed_layouts, var_column_indices = self.analyze_columns(columns, rows)
        
        print(f"\n=== 統合バッファ分析結果 ===")
        print(f"固定長列: {len(fixed_layouts)}列")
        for layout in fixed_layouts:
            print(f"  {layout.name}: offset={layout.buffer_offset}, size={layout.element_size}, type={layout.arrow_type_id}")
        print(f"可変長列: {len(var_column_indices)}列")
        
        # 統合バッファ作成
        unified_buffer, row_stride = self.create_unified_buffer(fixed_layouts, rows)
        
        # 可変長バッファ作成
        var_buffers = self.create_variable_buffers(columns, var_column_indices, rows)
        self.device_buffers.update(var_buffers)
        
        # 統合バッファ情報
        self.unified_buffer_info = UnifiedBufferInfo(
            buffer=unified_buffer,
            row_stride=row_stride,
            fixed_layouts=fixed_layouts,
            var_column_indices=var_column_indices
        )
        
        return self.unified_buffer_info
    
    def extract_fixed_column_arrays(self, unified_buffer_info: UnifiedBufferInfo, rows: int) -> Dict[str, cuda.cudadrv.devicearray.DeviceNDArray]:
        """統合バッファから個別列バッファを抽出"""
        extracted_buffers = {}
        
        unified_buffer = unified_buffer_info.buffer
        row_stride = unified_buffer_info.row_stride
        
        for layout in unified_buffer_info.fixed_layouts:
            # この列のデータを抽出
            column_buffer = cuda.device_array(rows * layout.element_size, dtype=np.uint8)
            
            # GPU上でデータコピー（ストライドアクセス）
            self._extract_column_data_gpu(
                unified_buffer, column_buffer,
                layout.buffer_offset, layout.element_size, row_stride, rows
            )
            
            extracted_buffers[layout.name] = column_buffer
        
        return extracted_buffers
    
    def _extract_column_data_gpu(self, unified_buffer, column_buffer, offset, element_size, row_stride, rows):
        """GPU上でストライドアクセスによる列データ抽出"""
        
        @cuda.jit
        def extract_kernel(unified_buf, col_buf, offset, elem_size, stride, num_rows):
            row = cuda.grid(1)
            if row >= num_rows:
                return
            
            src_offset = row * stride + offset
            dst_offset = row * elem_size
            
            for i in range(elem_size):
                col_buf[dst_offset + i] = unified_buf[src_offset + i]
        
        threads = 256
        blocks = (rows + threads - 1) // threads
        extract_kernel[blocks, threads](unified_buffer, column_buffer, offset, element_size, row_stride, rows)
        cuda.synchronize()
    
    def replace_varlen_data_buffer(self, column_name: str, new_size: int) -> cuda.cudadrv.devicearray.DeviceNDArray:
        """可変長列のデータバッファを再確保"""
        if column_name not in self.device_buffers:
            raise ValueError(f"Column {column_name} not found in device buffers")
        
        d_values_old, d_nulls, d_offsets, old_size = self.device_buffers[column_name]
        
        # 新しいデータバッファ確保
        d_values_new = cuda.device_array(new_size, dtype=np.uint8)
        
        # バッファ情報更新
        self.device_buffers[column_name] = (d_values_new, d_nulls, d_offsets, new_size)
        
        return d_values_new
    
    def get_buffer_info(self) -> UnifiedBufferInfo:
        """統合バッファ情報を取得"""
        return self.unified_buffer_info
    
    def get_variable_buffers(self) -> Dict[str, Any]:
        """可変長バッファを取得"""
        return {k: v for k, v in self.device_buffers.items()}

__all__ = ["GPUMemoryManagerV3Unified", "FixedColumnLayout", "UnifiedBufferInfo"]