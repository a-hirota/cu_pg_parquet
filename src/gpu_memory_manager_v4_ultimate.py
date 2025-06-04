"""
GPU Memory Manager V4: Ultimate統合版
===================================

Pass2完全廃止を実現する究極のメモリ管理システム:
1. 固定長列統合バッファ（V3継承）
2. 可変長文字列統合バッファ（新規）
3. 1回のカーネル処理用最適化
4. 並列Prefix-sum対応
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

class VariableColumnLayout(NamedTuple):
    """可変長列のレイアウト情報"""
    name: str
    column_index: int      # 元のテーブル列インデックス
    var_index: int         # 可変長列内のインデックス
    arrow_type_id: int     # Arrow型ID

class UltimateBufferInfo(NamedTuple):
    """Ultimate統合バッファの情報"""
    # 固定長バッファ
    fixed_buffer: cuda.cudadrv.devicearray.DeviceNDArray
    row_stride: int
    fixed_layouts: List[FixedColumnLayout]
    
    # 可変長バッファ
    var_data_buffer: cuda.cudadrv.devicearray.DeviceNDArray
    var_offset_buffers: Dict[str, cuda.cudadrv.devicearray.DeviceNDArray]
    var_layouts: List[VariableColumnLayout]
    
    # 一時作業領域
    var_lens_buffer: cuda.cudadrv.devicearray.DeviceNDArray

class GPUMemoryManagerV4Ultimate:
    """Ultimate統合バッファ対応のGPUメモリマネージャー"""
    
    def __init__(self):
        self.device_buffers = {}
        self.ultimate_buffer_info = None
        
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
    
    def analyze_columns(self, columns, rows: int) -> Tuple[List[FixedColumnLayout], List[VariableColumnLayout]]:
        """列を固定長と可変長に分類してレイアウト生成"""
        fixed_layouts = []
        var_layouts = []
        current_offset = 0
        var_index = 0
        
        for col_idx, col in enumerate(columns):
            if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY):
                # 可変長文字列列のみ対象
                layout = VariableColumnLayout(
                    name=col.name,
                    column_index=col_idx,
                    var_index=var_index,
                    arrow_type_id=col.arrow_id
                )
                var_layouts.append(layout)
                var_index += 1
            else:
                # 固定長列（Decimal含む）
                element_size = self.get_element_size(col.arrow_id)
                
                # Decimal列のスケール取得
                decimal_scale = 0
                if col.arrow_id == DECIMAL128:
                    precision, scale = col.arrow_param or (38, 0)
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
        
        return fixed_layouts, var_layouts
    
    def create_fixed_buffer(self, fixed_layouts: List[FixedColumnLayout], rows: int) -> Tuple[cuda.cudadrv.devicearray.DeviceNDArray, int]:
        """固定長統合バッファを作成"""
        if not fixed_layouts:
            return cuda.device_array(1, dtype=np.uint8), 0
        
        # 1行分のバイト数計算
        row_stride = sum(layout.element_size for layout in fixed_layouts)
        
        # 統合バッファ確保
        total_bytes = rows * row_stride
        fixed_buffer = cuda.device_array(total_bytes, dtype=np.uint8)
        
        print(f"固定長統合バッファ作成: {len(fixed_layouts)}列, 行ストライド={row_stride}バイト, 総サイズ={total_bytes}バイト")
        
        return fixed_buffer, row_stride
    
    def estimate_variable_total_size(self, var_layouts: List[VariableColumnLayout], rows: int) -> int:
        """可変長データの総サイズを推定"""
        if not var_layouts:
            return 1  # ダミー
        
        # 保守的な推定：列数 × 行数 × 平均長
        avg_string_length = 50  # 平均文字列長
        safety_factor = 1.5     # 安全係数
        
        total_estimated = len(var_layouts) * rows * avg_string_length * safety_factor
        return int(total_estimated)
    
    def create_variable_buffers(self, var_layouts: List[VariableColumnLayout], rows: int) -> Tuple[
        cuda.cudadrv.devicearray.DeviceNDArray,  # 統合データバッファ
        Dict[str, cuda.cudadrv.devicearray.DeviceNDArray],  # オフセットバッファ群
        cuda.cudadrv.devicearray.DeviceNDArray   # 長さ配列
    ]:
        """可変長統合バッファシステムを作成"""
        
        if not var_layouts:
            # 可変長列がない場合
            dummy_buffer = cuda.device_array(1, dtype=np.uint8)
            dummy_lens = cuda.device_array((1, 1), dtype=np.int32)
            return dummy_buffer, {}, dummy_lens
        
        # 統合データバッファ
        total_size = self.estimate_variable_total_size(var_layouts, rows)
        var_data_buffer = cuda.device_array(total_size, dtype=np.uint8)
        
        # 各列のオフセットバッファ
        var_offset_buffers = {}
        for layout in var_layouts:
            # オフセット配列（rows + 1）
            offset_buffer = cuda.device_array(rows + 1, dtype=np.int32)
            offset_buffer[0] = 0  # 初期化
            var_offset_buffers[layout.name] = offset_buffer
        
        # 長さ配列（各可変長列 × 行数）
        var_lens_buffer = cuda.device_array((len(var_layouts), rows), dtype=np.int32)
        
        print(f"可変長統合バッファ作成: {len(var_layouts)}列, 統合データサイズ={total_size}バイト")
        
        return var_data_buffer, var_offset_buffers, var_lens_buffer
    
    def initialize_ultimate_buffers(self, columns, rows: int) -> UltimateBufferInfo:
        """Ultimate統合バッファシステムの初期化"""
        
        # 列分析
        fixed_layouts, var_layouts = self.analyze_columns(columns, rows)
        
        print(f"\n=== Ultimate統合バッファ分析結果 ===")
        print(f"固定長列: {len(fixed_layouts)}列")
        for layout in fixed_layouts:
            print(f"  {layout.name}: offset={layout.buffer_offset}, size={layout.element_size}, type={layout.arrow_type_id}")
        print(f"可変長文字列列: {len(var_layouts)}列")
        for layout in var_layouts:
            print(f"  {layout.name}: var_idx={layout.var_index}, type={layout.arrow_type_id}")
        
        # 固定長統合バッファ作成
        fixed_buffer, row_stride = self.create_fixed_buffer(fixed_layouts, rows)
        
        # 可変長統合バッファ作成
        var_data_buffer, var_offset_buffers, var_lens_buffer = \
            self.create_variable_buffers(var_layouts, rows)
        
        # Ultimate統合バッファ情報
        self.ultimate_buffer_info = UltimateBufferInfo(
            fixed_buffer=fixed_buffer,
            row_stride=row_stride,
            fixed_layouts=fixed_layouts,
            var_data_buffer=var_data_buffer,
            var_offset_buffers=var_offset_buffers,
            var_layouts=var_layouts,
            var_lens_buffer=var_lens_buffer
        )
        
        return self.ultimate_buffer_info
    
    def extract_fixed_column_arrays(self, ultimate_info: UltimateBufferInfo, rows: int) -> Dict[str, cuda.cudadrv.devicearray.DeviceNDArray]:
        """固定長統合バッファから個別列バッファを抽出"""
        extracted_buffers = {}
        
        fixed_buffer = ultimate_info.fixed_buffer
        row_stride = ultimate_info.row_stride
        
        for layout in ultimate_info.fixed_layouts:
            # この列のデータを抽出
            column_buffer = cuda.device_array(rows * layout.element_size, dtype=np.uint8)
            
            # GPU上でデータコピー（ストライドアクセス）
            self._extract_column_data_gpu(
                fixed_buffer, column_buffer,
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
    
    def get_buffer_info(self) -> UltimateBufferInfo:
        """Ultimate統合バッファ情報を取得"""
        return self.ultimate_buffer_info

__all__ = ["GPUMemoryManagerV4Ultimate", "UltimateBufferInfo"]