"""
GPUメモリの割り当てと管理を担当するモジュール
"""

import numpy as np
import cupy as cp
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError
from typing import List, Dict, Any, Optional, Tuple

from .utils import ColumnInfo, get_column_type, get_column_length

class GPUMemoryManager:
    """GPU上のメモリリソースを管理するクラス"""
    
    def __init__(self):
        """初期化"""
        # CUDA初期化
        try:
            cuda.select_device(0)
            print("CUDA device initialized")
        except Exception as e:
            print(f"Failed to initialize CUDA device: {e}")
            raise
    
    def initialize_device_buffers(self, columns: List[ColumnInfo], chunk_size: int):
        """GPUバッファの初期化（累積オフセット方式）
        
        Args:
            columns: カラム情報のリスト
            chunk_size: チャンクサイズ（行数）
            
        Returns:
            各種GPUバッファオブジェクト
        """
        # カラム情報の収集
        col_types = []  # 0: integer, 1: numeric, 2: string
        col_lengths = []
        str_offsets = []  # 各文字列カラムのバッファ内オフセット
        total_str_buffer_size = 0  # 文字列バッファの合計サイズ
        num_int_cols = 0
        num_str_cols = 0
        
        for col in columns:
            col_type = get_column_type(col.type)
            col_types.append(col_type)
            
            if col_type <= 1:  # 数値型（integer or numeric）
                num_int_cols += 1
                col_lengths.append(get_column_length(col.type, col.length))
            else:  # 文字列型
                length = get_column_length(col.type, col.length)
                col_lengths.append(length)
                
                # このカラムの文字列バッファ開始位置を記録
                str_offsets.append(total_str_buffer_size)
                # 累積サイズに加算
                total_str_buffer_size += chunk_size * length
                num_str_cols += 1
        
        # バッファの確保
        try:
            # バッファサイズの計算
            int_buffer_size = chunk_size * num_int_cols
            str_null_pos_size = chunk_size * num_str_cols
            
            print(f"Allocating buffers: int={int_buffer_size}, str={total_str_buffer_size}, null={str_null_pos_size}")
            print(f"[DEBUG] String buffer details:")
            print(f"[DEBUG] - Total size: {total_str_buffer_size}")
            print(f"[DEBUG] - Number of rows: {chunk_size}")
            print(f"[DEBUG] - Number of string columns: {num_str_cols}")
            print(f"[DEBUG] - Column lengths: {[get_column_length(col.type, col.length) for col in columns if get_column_type(col.type) == 2]}")
            print(f"[DEBUG] - String column offsets: {str_offsets}")
            
            # バッファの確保と初期化
            int_buffer = None
            num_hi_output = None
            num_lo_output = None
            num_scale_output = None
            str_buffer = None
            str_null_pos = None
            d_str_offsets = None
            
            # メモリ割り当ての順序を調整
            if num_int_cols > 0:
                int_buffer = cuda.to_device(np.zeros(int_buffer_size, dtype=np.int32))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
                # numeric型用のバッファ
                num_hi_output = cuda.device_array(chunk_size * num_int_cols, dtype=np.int64)
                num_lo_output = cuda.device_array(chunk_size * num_int_cols, dtype=np.int64)
                num_scale_output = cuda.device_array(num_int_cols, dtype=np.int32)
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
            if num_str_cols > 0:
                # 文字列バッファを一括で確保
                str_buffer = cuda.to_device(np.zeros(total_str_buffer_size, dtype=np.uint8))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
                str_null_pos = cuda.to_device(np.zeros(str_null_pos_size, dtype=np.int32))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
                # 文字列オフセット情報をGPUへ転送
                d_str_offsets = cuda.to_device(np.array(str_offsets, dtype=np.int32))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
            
            return {
                "int_buffer": int_buffer, 
                "num_hi_output": num_hi_output, 
                "num_lo_output": num_lo_output, 
                "num_scale_output": num_scale_output,
                "str_buffer": str_buffer, 
                "str_null_pos": str_null_pos, 
                "d_str_offsets": d_str_offsets,
                "col_types": np.array(col_types, dtype=np.int32), 
                "col_lengths": np.array(col_lengths, dtype=np.int32)
            }
        except CudaAPIError as e:
            print(f"Failed to allocate GPU memory: {e}")
            self.cleanup_buffers(locals())
            raise
            
    def transfer_to_device(self, data, dtype=np.uint8):
        """データをGPUに転送"""
        try:
            d_data = cuda.to_device(np.array(data, dtype=dtype))
            cuda.synchronize()  # 転送完了を待つ
            return d_data
        except CudaAPIError as e:
            print(f"Failed to transfer data to GPU: {e}")
            raise
    
    def cleanup_buffers(self, buffers_dict):
        """GPUバッファのクリーンアップ
        
        Args:
            buffers_dict: バッファオブジェクトを含む辞書
        """
        # クリーンアップ対象のリスト
        cleanup_targets = [
            "int_buffer", "num_hi_output", "num_lo_output", "num_scale_output",
            "str_buffer", "str_null_pos", "d_str_offsets", "d_chunk", "d_offsets", "d_lengths",
            "d_col_types", "d_col_lengths"
        ]
        
        for target in cleanup_targets:
            if target in buffers_dict and buffers_dict[target] is not None:
                try:
                    del buffers_dict[target]
                except:
                    pass
        
        # 強制的に同期して解放を確実に
        cuda.synchronize()
