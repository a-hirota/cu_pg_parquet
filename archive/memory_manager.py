"""
GPUメモリの割り当てと管理を担当するモジュール
"""

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError
from typing import List, Dict, Any, Optional, Tuple

# CuPyを条件付きでインポート（インストールされていない場合はスキップ）
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    # CuPyがない場合はNoneとして扱う
    cp = None
    HAS_CUPY = False
    print("警告: CuPyがインストールされていません。一部の機能が制限されます。")

from .utils import ColumnInfo, get_column_type, get_column_length

def allocate_gpu_buffers(rows_in_chunk, num_columns, col_types, col_lengths):
    """GPUバッファを割り当てる
    
    Args:
        rows_in_chunk: チャンク内の行数
        num_columns: カラム数
        col_types: カラムタイプの配列
        col_lengths: カラム長の配列
        
    Returns:
        バッファ辞書
    """
    memory_manager = GPUMemoryManager()
    
    # カラム情報の作成
    columns = []
    for i in range(num_columns):
        col_type = "integer" if col_types[i] == 0 else "numeric" if col_types[i] == 1 else "character varying"
        columns.append(ColumnInfo(f"col_{i}", col_type, col_lengths[i]))
    
    # バッファ初期化
    return memory_manager.initialize_device_buffers(columns, rows_in_chunk)

class GPUMemoryManager:
    """GPU上のメモリリソースを管理するクラス"""
    
    def __init__(self):
        """初期化"""
        # CUDA初期化 - 既存のコンテキストを使用
        try:
            # デバイスを選択（既存のコンテキストを再利用）
            try:
                context = cuda.current_context()
                print("既存のCUDAコンテキストを使用")
            except:
                # コンテキストがない場合は新規に作成
                cuda.select_device(0)
                print("新規CUDAコンテキストを作成")
            print("CUDA device initialized")
            # GPUメモリ情報の表示
            self.print_gpu_memory_info()
        except Exception as e:
            print(f"Failed to initialize CUDA device: {e}")
            raise
            
    def print_gpu_memory_info(self):
        """GPUメモリ使用状況の表示"""
        try:
            mem_info = cuda.current_context().get_memory_info()
            free_memory = mem_info[0]
            total_memory = mem_info[1]
            used_memory = total_memory - free_memory
            percent_used = (used_memory / total_memory) * 100
            
            print(f"GPU Memory: {free_memory / (1024**2):.2f} MB free / {total_memory / (1024**2):.2f} MB total ({percent_used:.1f}% used)")
        except Exception as e:
            print(f"Unable to get GPU memory info: {e}")
    
    def get_available_gpu_memory(self):
        """利用可能なGPUメモリ量を取得（バイト単位）"""
        try:
            mem_info = cuda.current_context().get_memory_info()
            free_memory = mem_info[0]
            return free_memory
        except Exception as e:
            print(f"Unable to get GPU memory info: {e}")
            # フォールバック - 仮の値を返す
            return 1 * 1024**3  # 1GBをデフォルト値として返す
            
    def calculate_optimal_chunk_size(self, columns, total_rows):
        """GPUメモリ量に基づいて最適なチャンクサイズを計算
        
        Args:
            columns: カラム情報のリスト
            total_rows: 処理する総行数
            
        Returns:
            最適な行数
        """
        # 使用可能なGPUメモリを取得
        free_memory = self.get_available_gpu_memory()
        # 安全マージン（80%のみ使用）
        usable_memory = free_memory * 0.8
        
        # 1行あたりのメモリ使用量を計算（各カラムタイプ別）
        mem_per_row = 0
        for col in columns:
            from .utils import get_column_type, get_column_length
            if get_column_type(col.type) <= 1:  # 数値型
                mem_per_row += 8  # 8バイト（int64/float64）
            else:  # 文字列型
                mem_per_row += get_column_length(col.type, col.length)
        
        # オーバーヘッドを考慮して20%追加
        mem_per_row *= 1.2
        
        # 最大処理可能行数を計算
        max_rows = int(usable_memory / mem_per_row)
        
        # 最小行数と最大行数の設定
        min_rows = 1000  # 最低でも1000行は処理
        max_hard_limit = 65535  # ハードウェア制限（65535行）
        
        # 範囲内に収める
        max_rows = max(min_rows, min(max_rows, max_hard_limit, total_rows))
        
        print(f"メモリ計算: 利用可能={free_memory/1024**2:.2f}MB, 1行あたり={mem_per_row:.2f}B, 最適行数={max_rows}")
        
        return max_rows
    
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
            # numpy.int32値をリストとして表示する問題を防ぐ
            print(f"[DEBUG] - String column offsets: {[int(offset) for offset in str_offsets]}")
            
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
                int_buffer = cuda.to_device(np.zeros(int(int_buffer_size), dtype=np.int32))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
                # numeric型用のバッファ (タプルで形状を指定)
                num_hi_output = cuda.device_array((int(chunk_size * num_int_cols),), dtype=np.int64)
                num_lo_output = cuda.device_array((int(chunk_size * num_int_cols),), dtype=np.int64)
                num_scale_output = cuda.device_array((int(num_int_cols),), dtype=np.int32)
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
            if num_str_cols > 0:
                # 文字列バッファを一括で確保
                str_buffer = cuda.to_device(np.zeros(int(total_str_buffer_size), dtype=np.uint8))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
                str_null_pos = cuda.to_device(np.zeros(int(str_null_pos_size), dtype=np.int32))
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
        # クリーンアップ前のメモリ状態を記録
        try:
            before_free_memory = self.get_available_gpu_memory()
        except:
            before_free_memory = None
            
        # クリーンアップ対象のリスト
        cleanup_targets = [
            "int_buffer", "num_hi_output", "num_lo_output", "num_scale_output",
            "str_buffer", "str_null_pos", "d_str_offsets", "d_chunk", "d_offsets", "d_lengths",
            "d_col_types", "d_col_lengths"
        ]
        
        # 削除したオブジェクト数をカウント
        deleted_count = 0
        
        for target in cleanup_targets:
            if target in buffers_dict and buffers_dict[target] is not None:
                try:
                    del buffers_dict[target]
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {target}: {e}")
        
        # 強制的に同期して解放を確実に
        cuda.synchronize()
        
        # ガベージコレクションを明示的に実行
        try:
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error during garbage collection: {e}")
            
        # クリーンアップ後のメモリ状態を確認
        if before_free_memory is not None:
            try:
                after_free_memory = self.get_available_gpu_memory()
                freed_memory = after_free_memory - before_free_memory
                
                # メモリ解放が確認できた場合にのみ表示
                if freed_memory > 0:
                    print(f"クリーンアップにより {freed_memory / (1024**2):.2f} MB のGPUメモリを解放しました ({deleted_count} オブジェクト)")
            except:
                pass
