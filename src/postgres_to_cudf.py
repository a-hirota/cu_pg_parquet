"""
直接列抽出プロセッサー
====================

統合バッファを使わずに、入力データとパース結果から
直接cuDF列を作成する最適化実装
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import warnings
import numpy as np
import cupy as cp
import cudf
from numba import cuda
import pylibcudf as plc
import rmm
from numba import uint8
import os

from .types import (
    ColumnMeta, INT16, INT32, INT64, FLOAT32, FLOAT64, DECIMAL128,
    UTF8, BINARY, DATE32, TS64_US, BOOL, UNKNOWN
)
from .cuda_kernels.decimal_tables import (
    POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
)
from .cuda_kernels.gpu_config_utils import optimize_grid_size


# 128ビット演算のデバイス関数（モジュールレベル）
@cuda.jit(device=True, inline=True)
def _add128_fast(a_hi, a_lo, b_hi, b_lo):
    """高速128ビット加算（data_decoder.pyから移植）"""
    res_lo = a_lo + b_lo
    carry = 1 if res_lo < a_lo else 0
    res_hi = a_hi + b_hi + carry
    return res_hi, res_lo

@cuda.jit(device=True, inline=True)
def _mul128_u64_fast(a_hi, a_lo, b):
    """高速128ビット × 64ビット乗算（data_decoder.pyから移植）"""
    mask32 = 0xFFFFFFFF
    
    a0 = a_lo & mask32
    a1 = a_lo >> 32
    a2 = a_hi & mask32
    a3 = a_hi >> 32
    
    b0 = b & mask32
    b1 = b >> 32
    
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    p20 = a2 * b0
    p21 = a2 * b1
    p30 = a3 * b0
    
    c0 = p00 >> 32
    r0 = p00 & mask32
    
    temp1 = p01 + p10 + c0
    c1 = temp1 >> 32
    r1 = temp1 & mask32
    
    temp2 = p11 + p20 + c1
    c2 = temp2 >> 32
    r2 = temp2 & mask32
    
    temp3 = p21 + p30 + c2
    r3 = temp3 & mask32
    
    res_lo = (r1 << 32) | r0
    res_hi = (r3 << 32) | r2
    
    return res_hi, res_lo


class DirectColumnExtractor:
    """入力データから直接列を抽出するプロセッサー"""
    
    def __init__(self):
        """初期化"""
        self.d_pow10_table_lo = None
        self.d_pow10_table_hi = None
        self.device_props = self._get_device_properties()
    
    def _get_device_properties(self) -> dict:
        """GPUデバイスプロパティを取得"""
        try:
            device = cuda.get_current_device()
            # CuPyを使用してGPUメモリサイズを取得
            meminfo = cp.cuda.runtime.memGetInfo()
            gpu_memory_size = meminfo[1]  # total memory
            
            return {
                'compute_capability': device.compute_capability,
                'name': device.name.decode('utf-8') if hasattr(device.name, 'decode') else str(device.name),
                'multiprocessor_count': device.MULTIPROCESSOR_COUNT,
                'warp_size': device.WARP_SIZE,
                'max_threads_per_block': device.MAX_THREADS_PER_BLOCK,
                'max_block_dim_x': device.MAX_BLOCK_DIM_X,
                'max_block_dim_y': device.MAX_BLOCK_DIM_Y,
                'max_block_dim_z': device.MAX_BLOCK_DIM_Z,
                'max_grid_dim_x': device.MAX_GRID_DIM_X,
                'max_grid_dim_y': device.MAX_GRID_DIM_Y,
                'max_grid_dim_z': device.MAX_GRID_DIM_Z,
                'gpu_memory_size': gpu_memory_size
            }
        except Exception as e:
            warnings.warn(f"GPUプロパティ取得失敗: {e}")
            return {
                'compute_capability': (7, 0),
                'name': 'Unknown GPU',
                'multiprocessor_count': 80,
                'warp_size': 32,
                'max_threads_per_block': 1024,
                'max_block_dim_x': 1024,
                'max_block_dim_y': 1024,
                'max_block_dim_z': 64,
                'max_grid_dim_x': 2147483647,
                'max_grid_dim_y': 65535,
                'max_grid_dim_z': 65535,
                'gpu_memory_size': 0
            }
    
    def _ensure_decimal_tables(self):
        """Decimal処理用テーブルの遅延初期化"""
        if self.d_pow10_table_lo is None:
            self.d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
            self.d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)
    
    def extract_columns_direct(
        self,
        raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        row_positions_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # 追加
        field_offsets_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        field_lengths_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        columns: List[ColumnMeta],
        string_buffers: Optional[Dict[str, Any]] = None,
        thread_ids_dev: Optional[cuda.cudadrv.devicearray.DeviceNDArray] = None,
        thread_start_positions_dev: Optional[cuda.cudadrv.devicearray.DeviceNDArray] = None,
        thread_end_positions_dev: Optional[cuda.cudadrv.devicearray.DeviceNDArray] = None
    ) -> cudf.DataFrame:
        """
        入力データから直接列を抽出してcuDF DataFrameを作成
        
        統合バッファを経由せず、メモリ効率的に処理
        """
        rows, ncols = field_offsets_dev.shape
        cudf_series_dict = {}
        
        # デバッグ：処理する列の概要を出力
        if os.environ.get('GPUPGPARSER_TEST_MODE') == '1':
            chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
            prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
            print(f"\n{prefix} === 列抽出処理開始 ===")
            print(f"{prefix} 総列数: {len(columns)}")
            fixed_cols = [c for c in columns if not c.is_variable]
            string_cols = [c for c in columns if c.is_variable and (c.arrow_id == UTF8 or c.arrow_id == BINARY)]
            print(f"{prefix} 固定長列: {len(fixed_cols)}個")
            print(f"{prefix} 文字列列: {len(string_cols)}個")
            if fixed_cols:
                first_fixed = fixed_cols[0]
                type_name = first_fixed.arrow_id.name if hasattr(first_fixed.arrow_id, 'name') else f"Type_{first_fixed.arrow_id}"
                print(f"{prefix} 最初の固定長列: {first_fixed.name} ({type_name})")
            if string_cols:
                print(f"{prefix} 最初の文字列列: {string_cols[0].name}")
            print(f"{prefix} =================\n")
        
        for col_idx, col in enumerate(columns):
            try:
                if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY):
                    # 文字列列は既存の最適化済み処理を使用
                    if string_buffers and col.name in string_buffers:
                        series = self._create_string_series_from_buffer(
                            col, rows, string_buffers[col.name]
                        )
                    else:
                        series = cudf.Series([None] * rows, dtype='string')
                else:
                    # 固定長列：入力データから直接抽出
                    series = self._extract_fixed_column_direct(
                        raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev,
                        col, col_idx, rows, columns
                    )
                
                cudf_series_dict[col.name] = series
                
            except Exception as e:
                warnings.warn(f"列 {col.name} の直接抽出でエラー: {e}")
                cudf_series_dict[col.name] = cudf.Series([None] * rows)
        
        # テストモード時にデバッグ列を追加
        if thread_ids_dev is not None:
            try:
                # thread_idsをCuPy配列として解釈
                thread_ids_cupy = cp.asarray(thread_ids_dev)
                cudf_series_dict['_thread_id'] = cudf.Series(thread_ids_cupy)
            except Exception as e:
                warnings.warn(f"thread_ids列の追加でエラー: {e}")
        
        # row_positionsも追加（row_positions_devから）
        if row_positions_dev is not None:
            try:
                row_positions_cupy = cp.asarray(row_positions_dev)
                cudf_series_dict['_row_position'] = cudf.Series(row_positions_cupy)
            except Exception as e:
                warnings.warn(f"row_positions列の追加でエラー: {e}")
        
        # thread_start_positionsを追加
        if thread_start_positions_dev is not None:
            try:
                thread_start_cupy = cp.asarray(thread_start_positions_dev)
                cudf_series_dict['_thread_start_pos'] = cudf.Series(thread_start_cupy)
            except Exception as e:
                warnings.warn(f"thread_start_positions列の追加でエラー: {e}")
        
        # thread_end_positionsを追加
        if thread_end_positions_dev is not None:
            try:
                thread_end_cupy = cp.asarray(thread_end_positions_dev)
                cudf_series_dict['_thread_end_pos'] = cudf.Series(thread_end_cupy)
            except Exception as e:
                warnings.warn(f"thread_end_positions列の追加でエラー: {e}")
        
        return cudf.DataFrame(cudf_series_dict)
    
    def _extract_fixed_column_direct(
        self,
        raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        row_positions_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # 追加
        field_offsets_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        field_lengths_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        col: ColumnMeta,
        col_idx: int,
        rows: int,
        columns: List[ColumnMeta] = None  # Decimal列判定用
    ) -> cudf.Series:
        """固定長列を入力データから直接抽出"""
        
        # 列のサイズを決定
        if col.arrow_id == DECIMAL128:
            col_size = 16
            self._ensure_decimal_tables()
            # PostgreSQLメタデータからscaleを取得
            if col.arrow_param and isinstance(col.arrow_param, tuple):
                precision, decimal_scale = col.arrow_param
            else:
                # メタデータがない場合のデフォルト
                precision, decimal_scale = 38, 0
        elif col.arrow_id == INT32 or col.arrow_id == FLOAT32 or col.arrow_id == DATE32:
            col_size = 4
        elif col.arrow_id == INT64 or col.arrow_id == FLOAT64 or col.arrow_id == TS64_US:
            col_size = 8
        elif col.arrow_id == INT16:
            col_size = 2
        elif col.arrow_id == BOOL:
            col_size = 1
        else:
            # 未対応の型
            return cudf.Series([None] * rows)
        
        # RMM DeviceBufferを使用（ゼロコピー実現）
        column_buffer_rmm = rmm.DeviceBuffer(size=rows * col_size)
        
        # CuPy配列としてビュー作成（ゼロコピー）
        column_cupy = cp.ndarray(
            shape=(rows * col_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(column_buffer_rmm.ptr, column_buffer_rmm.size, column_buffer_rmm),
                0
            )
        )
        
        # Numbaカーネル用に変換
        column_buffer = cuda.as_cuda_array(column_cupy)
        
        # NULL配列もRMM DeviceBuffer使用
        null_mask_rmm = rmm.DeviceBuffer(size=rows)
        null_mask_cupy = cp.ndarray(
            shape=(rows,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(null_mask_rmm.ptr, null_mask_rmm.size, null_mask_rmm),
                0
            )
        )
        null_mask = cuda.as_cuda_array(null_mask_cupy)
        
        # 直接抽出カーネルを実行（2次元グリッド対応）
        threads_per_block = 256
        
        # 2次元グリッドサイズ計算
        max_blocks_x = self.device_props.get('max_grid_dim_x', 2147483647)
        max_blocks_y = self.device_props.get('max_grid_dim_y', 65535)
        
        # まずは1行/1スレッドで計算
        total_threads_needed = rows
        blocks_x = min((total_threads_needed + threads_per_block - 1) // threads_per_block, max_blocks_x)
        blocks_y = (total_threads_needed + blocks_x * threads_per_block - 1) // (blocks_x * threads_per_block)
        
        # Y次元の制限を超える場合は各スレッドが複数行処理
        rows_per_thread = 1
        if blocks_y > max_blocks_y:
            # 最大グリッドサイズでの総スレッド数
            max_total_threads = max_blocks_x * max_blocks_y * threads_per_block
            rows_per_thread = (rows + max_total_threads - 1) // max_total_threads
            blocks_y = min(blocks_y, max_blocks_y)
        
        # グリッド設定
        grid = (blocks_x, blocks_y)
        
        # カーネル実行情報を出力（最初の固定長列のみ）
        if os.environ.get('GPUPGPARSER_TEST_MODE') == '1' and col_idx == 0:
            chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
            prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
            print(f"\n{prefix} ============================================================")
            print(f"{prefix} Kernel Execution Info: _extract_fixed_direct (First Column)")
            print(f"{prefix} ============================================================")
            print(f"{prefix} Column: {col.name} (index: {col_idx})")
            # arrow_idが整数の場合と列挙型の場合の両方に対応
            data_type_name = col.arrow_id.name if hasattr(col.arrow_id, 'name') else f"Type_{col.arrow_id}"
            print(f"{prefix} Data Type: {data_type_name}")
            print(f"{prefix} Grid Dimensions: {grid}")
            print(f"{prefix} Block Dimensions: {threads_per_block}")
            print(f"{prefix} Total Blocks: {blocks_x * blocks_y:,}")
            print(f"{prefix} Total Threads: {blocks_x * blocks_y * threads_per_block:,}")
            print(f"{prefix} Rows per Thread: {rows_per_thread}")
            print(f"{prefix} Total Rows: {rows:,}")
            print(f"{prefix} ============================================================\n")
        
        if col.arrow_id == DECIMAL128:
            # カーネル実行情報を出力（最初のDecimal列のみ）
            if os.environ.get('GPUPGPARSER_TEST_MODE') == '1' and columns:
                # Decimal列のカウントが必要（最初のDecimal列のみ出力）
                is_first_decimal = True
                for i in range(col_idx):
                    if i < len(columns) and columns[i].arrow_id == DECIMAL128:
                        is_first_decimal = False
                        break
                
                if is_first_decimal:
                    chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
                    prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
                    print(f"\n{prefix} ============================================================")
                    print(f"{prefix} Kernel Execution Info: _extract_decimal_direct (First Decimal)")
                    print(f"{prefix} ============================================================")
                    print(f"{prefix} Column: {col.name} (index: {col_idx})")
                    print(f"{prefix} Data Type: DECIMAL128")
                    print(f"{prefix} Precision/Scale: {precision}/{decimal_scale}")
                    print(f"{prefix} Grid Dimensions: {grid}")
                    print(f"{prefix} Block Dimensions: {threads_per_block}")
                    print(f"{prefix} Total Blocks: {blocks_x * blocks_y:,}")
                    print(f"{prefix} Total Threads: {blocks_x * blocks_y * threads_per_block:,}")
                    print(f"{prefix} Rows per Thread: {rows_per_thread}")
                    print(f"{prefix} Total Rows: {rows:,}")
                    print(f"{prefix} ============================================================\n")
            
            # Decimal専用カーネル
            self._extract_decimal_direct[grid, threads_per_block](
                raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev,
                col_idx, column_buffer, null_mask,
                self.d_pow10_table_lo, self.d_pow10_table_hi,
                decimal_scale, rows, rows_per_thread
            )
        else:
            # その他の固定長型用カーネル
            # arrow_idの値を整数として渡す
            arrow_type_id = int(col.arrow_id)
            self._extract_fixed_direct[grid, threads_per_block](
                raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev,
                col_idx, arrow_type_id, col_size, 
                column_buffer, null_mask, rows, rows_per_thread
            )
        
        cuda.synchronize()
        
        # cuDF Series作成（RMM DeviceBufferを直接使用）
        return self._create_series_from_rmm_buffer(
            column_buffer_rmm, null_mask_rmm, col, rows
        )
    
    @staticmethod
    @cuda.jit
    def _extract_fixed_direct(
        raw_data, row_positions, field_offsets, field_lengths,
        col_idx, arrow_type, col_size,
        output_buffer, null_mask, rows, rows_per_thread
    ):
        """固定長データの直接抽出カーネル（2次元グリッド対応）"""
        # 2次元グリッドからスレッドIDを計算
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + \
              cuda.blockIdx.y * cuda.gridDim.x * cuda.blockDim.x
        
        # 各スレッドが処理する行範囲
        start_row = tid * rows_per_thread
        end_row = min(start_row + rows_per_thread, rows)
        
        # 複数行を処理
        for row in range(start_row, end_row):
            # フィールド情報取得（相対オフセットから絶対位置を計算）
            row_start = row_positions[row]
            relative_offset = field_offsets[row, col_idx]
            src_offset = row_start + relative_offset  # uint64として計算
            field_length = field_lengths[row, col_idx]
            
            # NULL判定
            is_null = (field_length == -1)
            null_mask[row] = 0 if is_null else 1
            
            # 出力位置
            dst_offset = row * col_size
            
            if is_null or src_offset == 0:
                # NULLの場合はゼロで埋める
                for i in range(col_size):
                    if dst_offset + i < output_buffer.size:
                        output_buffer[dst_offset + i] = 0
            else:
                # データをコピー（エンディアン変換）
                # INT32=1, FLOAT32=3, DATE32=9
                if arrow_type == 1 or arrow_type == 3 or arrow_type == 9:
                    # 4バイト型：ビッグエンディアン→リトルエンディアン
                    if src_offset + 4 <= raw_data.size and dst_offset + 4 <= output_buffer.size:
                        output_buffer[dst_offset] = raw_data[src_offset + 3]
                        output_buffer[dst_offset + 1] = raw_data[src_offset + 2]
                        output_buffer[dst_offset + 2] = raw_data[src_offset + 1]
                        output_buffer[dst_offset + 3] = raw_data[src_offset]
                
                # INT64=2, FLOAT64=4, TS64_US=10
                elif arrow_type == 2 or arrow_type == 4 or arrow_type == 10:
                    # 8バイト型：ビッグエンディアン→リトルエンディアン
                    if src_offset + 8 <= raw_data.size and dst_offset + 8 <= output_buffer.size:
                        for i in range(8):
                            output_buffer[dst_offset + i] = raw_data[src_offset + 7 - i]
                
                # INT16=11
                elif arrow_type == 11:
                    # 2バイト型：ビッグエンディアン→リトルエンディアン
                    if src_offset + 2 <= raw_data.size and dst_offset + 2 <= output_buffer.size:
                        output_buffer[dst_offset] = raw_data[src_offset + 1]
                        output_buffer[dst_offset + 1] = raw_data[src_offset]
                
                # BOOL=8
                elif arrow_type == 8:
                    # 1バイト型：そのままコピー
                    if src_offset < raw_data.size and dst_offset < output_buffer.size:
                        output_buffer[dst_offset] = raw_data[src_offset]
    
    @staticmethod
    @cuda.jit
    def _extract_decimal_direct(
        raw_data, row_positions, field_offsets, field_lengths,
        col_idx, output_buffer, null_mask,
        d_pow10_table_lo, d_pow10_table_hi,
        target_scale, rows, rows_per_thread
    ):
        """Decimal128データの直接抽出カーネル（2次元グリッド対応）"""
        # 2次元グリッドからスレッドIDを計算
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + \
              cuda.blockIdx.y * cuda.gridDim.x * cuda.blockDim.x
        
        # 各スレッドが処理する行範囲
        start_row = tid * rows_per_thread
        end_row = min(start_row + rows_per_thread, rows)
        
        # 複数行を処理
        for row in range(start_row, end_row):
            # フィールド情報取得（相対オフセットから絶対位置を計算）
            row_start = row_positions[row]
            relative_offset = field_offsets[row, col_idx]
            src_offset = row_start + relative_offset  # uint64として計算
            field_length = field_lengths[row, col_idx]
            
            # NULL判定
            is_null = (field_length == -1)
            null_mask[row] = 0 if is_null else 1
            
            # 出力位置（16バイト）
            dst_offset = row * 16
            
            if is_null or src_offset == 0:
                # NULLの場合はゼロで埋める
                for i in range(16):
                    if dst_offset + i < output_buffer.size:
                        output_buffer[dst_offset + i] = 0
            else:
                # PostgreSQL NUMERICからDecimal128への変換
                # （data_decoder.pyのparse_decimal_from_rawロジックを使用）
                
                # NUMERICヘッダ読み取り（8バイト）
                if src_offset + 8 > raw_data.size:
                    for i in range(16):
                        if dst_offset + i < output_buffer.size:
                            output_buffer[dst_offset + i] = 0
                    continue
                
                nd = (raw_data[src_offset] << 8) | raw_data[src_offset + 1]
                weight = (raw_data[src_offset + 2] << 8) | raw_data[src_offset + 3]
                sign = (raw_data[src_offset + 4] << 8) | raw_data[src_offset + 5]
                dscale = (raw_data[src_offset + 6] << 8) | raw_data[src_offset + 7]
                
                # NaN処理
                if sign == 0xC000:
                    for i in range(16):
                        if dst_offset + i < output_buffer.size:
                            output_buffer[dst_offset + i] = 0
                    continue
                
                # 桁数制限
                if nd > 9:
                    for i in range(16):
                        if dst_offset + i < output_buffer.size:
                            output_buffer[dst_offset + i] = 0
                    continue
                
                current_offset = src_offset + 8
                
                # 基数10000から128ビット整数への変換（data_decoder.pyの実装を使用）
                val_hi = 0
                val_lo = 0
                
                # 基数1e8最適化実装
                i = 0
                while i < nd:
                    if i + 1 < nd and current_offset + 4 <= raw_data.size:
                        # 2桁まとめて処理（基数10000 * 10000 = 1e8）
                        digit1 = (raw_data[current_offset] << 8) | raw_data[current_offset + 1]
                        digit2 = (raw_data[current_offset + 2] << 8) | raw_data[current_offset + 3]
                        combined_digit = digit1 * 10000 + digit2
                        
                        val_hi, val_lo = _mul128_u64_fast(val_hi, val_lo, 100000000)  # 1e8
                        val_hi, val_lo = _add128_fast(val_hi, val_lo, 0, combined_digit)
                        current_offset += 4
                        i += 2
                    elif current_offset + 2 <= raw_data.size:
                        # 残り1桁の処理
                        digit = (raw_data[current_offset] << 8) | raw_data[current_offset + 1]
                        val_hi, val_lo = _mul128_u64_fast(val_hi, val_lo, 10000)
                        val_hi, val_lo = _add128_fast(val_hi, val_lo, 0, digit)
                        current_offset += 2
                        i += 1
                    else:
                        break
                
                # weightによる調整
                # weightは基数10000での最上位桁の位置を示す
                # weight=0: 1～9999、weight=1: 10000～99990000
                # 実際の値 = Σ(digits[i] * 10000^(weight-i))
                # 
                # 最下位桁の位置を計算：weight - (nd - 1)
                # これを10000^n倍する必要がある
                if weight > 0:
                    # 10000^weightだけ乗算が必要
                    # ただし、すでに桁を順番に処理しているので、
                    # 最下位桁の位置分だけ追加で乗算
                    lowest_digit_weight = weight - (nd - 1)
                    for _ in range(lowest_digit_weight):
                        val_hi, val_lo = _mul128_u64_fast(val_hi, val_lo, 10000)
                
                # スケール調整
                # PostgreSQLのdscaleをtarget_scaleに合わせる
                if dscale != target_scale:
                    scale_diff = target_scale - dscale
                    if scale_diff > 0:
                        # 小数点を右に移動（10^scale_diff倍）
                        for _ in range(scale_diff):
                            val_hi, val_lo = _mul128_u64_fast(val_hi, val_lo, 10)
                    # scale_diff < 0の場合（除算）は複雑なため、現時点では省略
                
                # 符号適用
                if sign == 0x4000:  # 負数
                    # 2の補数
                    val_lo = ~val_lo + 1
                    val_hi = ~val_hi
                    if val_lo == 0:
                        val_hi += 1
                
                # リトルエンディアンで書き込み
                for i in range(8):
                    if dst_offset + i < output_buffer.size:
                        output_buffer[dst_offset + i] = (val_lo >> (i * 8)) & 0xFF
                for i in range(8):
                    if dst_offset + 8 + i < output_buffer.size:
                        output_buffer[dst_offset + 8 + i] = (val_hi >> (i * 8)) & 0xFF
    
    def _create_series_from_rmm_buffer(
        self,
        column_buffer_rmm: rmm.DeviceBuffer,
        null_mask_rmm: rmm.DeviceBuffer,
        col: ColumnMeta,
        rows: int
    ) -> cudf.Series:
        """RMM DeviceBufferからcuDF Seriesを作成（ゼロコピー）"""
        
        try:
            # pylibcudfを使用してゼロコピーでSeries作成
            import pylibcudf as plc
            
            if col.arrow_id == DECIMAL128:
                # PostgreSQLメタデータからscaleを取得
                if col.arrow_param and isinstance(col.arrow_param, tuple):
                    precision, decimal_scale = col.arrow_param
                else:
                    precision, decimal_scale = 38, 0
                dt = plc.types.DataType(plc.types.TypeId.DECIMAL128, -decimal_scale)
            elif col.arrow_id == INT32:
                dt = plc.types.DataType(plc.types.TypeId.INT32)
            elif col.arrow_id == INT64:
                dt = plc.types.DataType(plc.types.TypeId.INT64)
            elif col.arrow_id == FLOAT32:
                dt = plc.types.DataType(plc.types.TypeId.FLOAT32)
            elif col.arrow_id == FLOAT64:
                dt = plc.types.DataType(plc.types.TypeId.FLOAT64)
            elif col.arrow_id == DATE32:
                dt = plc.types.DataType(plc.types.TypeId.DATE32)
            elif col.arrow_id == BOOL:
                dt = plc.types.DataType(plc.types.TypeId.BOOL8)
            elif col.arrow_id == INT16:
                dt = plc.types.DataType(plc.types.TypeId.INT16)
            elif col.arrow_id == TS64_US:
                dt = plc.types.DataType(plc.types.TypeId.TIMESTAMP_MICROSECONDS)
            else:
                # フォールバック
                raise ValueError(f"未対応の型: {col.arrow_id}")
            
            # RMM DeviceBufferのgpumemoryview（ゼロコピー）
            data_mv = plc.gpumemoryview(column_buffer_rmm)
            
            # NULLマスクの処理
            # TODO: null_mask_rmmをcuDFのビットマスクに変換
            
            # Column作成
            col_cpp = plc.column.Column(
                dt,          # data_type
                rows,        # size
                data_mv,     # data buffer
                None,        # null_mask (TODO: 実装)
                0,           # null_count
                0,           # offset
                []           # children
            )
            
            return cudf.Series.from_pylibcudf(col_cpp)
            
        except Exception as e:
            warnings.warn(f"RMM Series作成失敗: {e}")
            # フォールバック
            return self._create_series_from_buffer_fallback(column_buffer_rmm, col, rows)
    
    def _create_series_from_buffer_fallback(
        self,
        column_buffer_rmm: rmm.DeviceBuffer,
        col: ColumnMeta,
        rows: int
    ) -> cudf.Series:
        """フォールバック: RMM DeviceBufferからCuPy経由でSeries作成"""
        try:
            # CuPy配列として解釈
            if col.arrow_id == FLOAT32:
                col_size = 4
                dtype = cp.float32
            elif col.arrow_id == FLOAT64:
                col_size = 8
                dtype = cp.float64
            elif col.arrow_id == INT32 or col.arrow_id == DATE32:
                col_size = 4
                dtype = cp.int32
            elif col.arrow_id == INT64 or col.arrow_id == TS64_US:
                col_size = 8
                dtype = cp.int64
            elif col.arrow_id == INT16:
                col_size = 2
                dtype = cp.int16
            elif col.arrow_id == BOOL:
                col_size = 1
                dtype = cp.uint8
            elif col.arrow_id == DECIMAL128:
                # Decimalは特殊処理
                return self._create_decimal_series_from_rmm(column_buffer_rmm, col, rows)
            else:
                return cudf.Series([None] * rows)
            
            data_cupy = cp.ndarray(
                shape=(rows,),
                dtype=dtype,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(column_buffer_rmm.ptr, rows * col_size, column_buffer_rmm),
                    0
                )
            )
            
            # DATE32の特殊処理
            if col.arrow_id == DATE32:
                return cudf.Series(data_cupy, dtype='datetime64[D]')
            elif col.arrow_id == TS64_US:
                return cudf.Series(data_cupy, dtype='datetime64[us]')
            else:
                return cudf.Series(data_cupy)
                
        except Exception as e:
            warnings.warn(f"フォールバック失敗: {e}")
            return cudf.Series([None] * rows)
    
    def _create_decimal_series_from_rmm(
        self,
        column_buffer_rmm: rmm.DeviceBuffer,
        col: ColumnMeta,
        rows: int
    ) -> cudf.Series:
        """Decimal128列の特殊処理"""
        try:
            # 下位64bitのみをint64として取得
            int64_cupy = cp.ndarray(
                shape=(rows,),
                dtype=cp.int64,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(column_buffer_rmm.ptr, rows * 8, column_buffer_rmm),
                    0
                )
            )
            
            # Decimal128に変換（PostgreSQLメタデータ使用）
            if col.arrow_param and isinstance(col.arrow_param, tuple):
                precision, decimal_scale = col.arrow_param
            else:
                precision, decimal_scale = 38, 0
            decimal_dtype = cudf.Decimal128Dtype(precision=precision, scale=decimal_scale)
            return cudf.Series(int64_cupy).astype(decimal_dtype)
            
        except Exception as e:
            warnings.warn(f"Decimal変換失敗: {e}")
            return cudf.Series([None] * rows)
    
    def _create_series_from_buffer(
        self,
        column_buffer: cuda.cudadrv.devicearray.DeviceNDArray,
        null_mask: cuda.cudadrv.devicearray.DeviceNDArray,
        col: ColumnMeta,
        rows: int
    ) -> cudf.Series:
        """レガシー: Numba配列からcuDF Seriesを作成"""
        # 互換性のため保持
        warnings.warn("レガシーメソッドが呼ばれました")
        return cudf.Series([None] * rows)
    
    def _create_string_series_from_buffer(
        self,
        col: ColumnMeta,
        rows: int,
        buffer_info: Dict[str, Any]
    ) -> cudf.Series:
        """文字列バッファからSeries作成（既存の最適化版を使用）"""
        
        if buffer_info['data'] is None or buffer_info['offsets'] is None:
            return cudf.Series([None] * rows, dtype='string')
        
        try:
            data_buffer = buffer_info['data']
            offsets_buffer = buffer_info['offsets']
            
            # CuPy配列として解釈
            data_ptr = data_buffer.__cuda_array_interface__['data'][0]
            offsets_ptr = offsets_buffer.__cuda_array_interface__['data'][0]
            
            data_cupy = cp.asarray(cp.ndarray(
                shape=(buffer_info['actual_size'],),
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(data_ptr, buffer_info['actual_size'], data_buffer),
                    0
                )
            ))
            
            offsets_cupy = cp.asarray(cp.ndarray(
                shape=(rows + 1,),
                dtype=cp.int32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(offsets_ptr, (rows + 1) * 4, offsets_buffer),
                    0
                )
            ))
            
            # cuDFの最適化された文字列作成（真のゼロコピー）
            try:
                import pylibcudf as plc
                
                # RMM DeviceBufferが直接渡されている場合
                if isinstance(data_buffer, rmm.DeviceBuffer) and isinstance(offsets_buffer, rmm.DeviceBuffer):
                    # 基本ベンチマークと同じ子カラム構造を使用
                    # オフセット子カラムの作成
                    offsets_mv = plc.gpumemoryview(offsets_buffer)
                    offsets_col = plc.column.Column(
                        plc.types.DataType(plc.types.TypeId.INT32),
                        rows + 1,  # オフセット配列のサイズ
                        offsets_mv,
                        None,  # mask
                        0,     # null_count
                        0,     # offset
                        []     # children
                    )
                    
                    # STRING親カラムの作成
                    chars_mv = plc.gpumemoryview(data_buffer)
                    parent = plc.column.Column(
                        plc.types.DataType(plc.types.TypeId.STRING),
                        rows,           # 文字列の本数
                        chars_mv,       # chars buffer
                        None,           # mask
                        0,              # null_count
                        0,              # offset
                        [offsets_col]   # offset column
                    )
                    
                    # Series作成
                    return cudf.Series.from_pylibcudf(parent)
                else:
                    # Numba配列の場合（フォールバック）
                    # CuPy配列として解釈して直接使用（ホスト転送なし）
                    data_ptr = data_buffer.__cuda_array_interface__['data'][0]
                    offsets_ptr = offsets_buffer.__cuda_array_interface__['data'][0]
                    
                    # gpumemoryview作成（ゼロコピー）
                    # TODO: Numba配列からのgpumemoryview作成を最適化
                    data_cupy = cp.asarray(cp.ndarray(
                        shape=(buffer_info['actual_size'],),
                        dtype=cp.uint8,
                        memptr=cp.cuda.MemoryPointer(
                            cp.cuda.UnownedMemory(data_ptr, buffer_info['actual_size'], data_buffer),
                            0
                        )
                    ))
                    
                    offsets_cupy = cp.asarray(cp.ndarray(
                        shape=(rows + 1,),
                        dtype=cp.int32,
                        memptr=cp.cuda.MemoryPointer(
                            cp.cuda.UnownedMemory(offsets_ptr, (rows + 1) * 4, offsets_buffer),
                            0
                        )
                    ))
                    
                    # RMM DeviceBufferへのゼロコピー変換を試みる
                    # TODO: より効率的な方法を検討
                    chars_mv = plc.gpumemoryview(data_cupy.__cuda_array_interface__)
                    offsets_mv = plc.gpumemoryview(offsets_cupy.__cuda_array_interface__)
                    offsets_count = rows + 1
                
                # 子カラム作成
                offsets_col = plc.column.Column(
                    plc.types.DataType(plc.types.TypeId.INT32),
                    offsets_count,
                    offsets_mv,
                    None,  # mask
                    0,     # null_count
                    0,     # offset
                    []     # children
                )
                
                # STRING Column構築
                parent = plc.column.Column(
                    plc.types.DataType(plc.types.TypeId.STRING),
                    rows,                    # 文字列の本数
                    chars_mv,                # chars buffer
                    None,                    # mask
                    0,                       # null_count
                    0,                       # offset
                    [offsets_col]            # offset column のみ
                )
                
                # Python Series化
                return cudf.Series.from_pylibcudf(parent)
                
            except Exception as e2:
                warnings.warn(f"ゼロコピー文字列変換失敗、フォールバック使用: {e2}")
                # フォールバック：GPU→CPU→GPU転送版
                try:
                    # オフセット配列をバイト配列として変換
                    offsets_host = offsets_cupy.get()
                    offsets_bytes = offsets_host.tobytes()
                    offsets_buf = rmm.DeviceBuffer.to_device(offsets_bytes)
                    
                    # データ配列をバイト配列として変換
                    data_host = data_cupy.get()
                    chars_buf = rmm.DeviceBuffer.to_device(data_host.tobytes())
                    
                    # 子カラム作成（offsets only）
                    offsets_mv = plc.gpumemoryview(offsets_buf)
                    offsets_col = plc.column.Column(
                        plc.types.DataType(plc.types.TypeId.INT32),
                        len(offsets_host),
                        offsets_mv,
                        None,  # mask
                        0,     # null_count
                        0,     # offset
                        []     # children
                    )
                    
                    # STRING Column構築
                    chars_mv = plc.gpumemoryview(chars_buf)
                    parent = plc.column.Column(
                        plc.types.DataType(plc.types.TypeId.STRING),
                        rows,                    # 文字列の本数
                        chars_mv,                # chars buffer
                        None,                    # mask
                        0,                       # null_count
                        0,                       # offset
                        [offsets_col]            # offset column のみ
                    )
                    
                    # Python Series化
                    return cudf.Series.from_pylibcudf(parent)
                except Exception as e3:
                    warnings.warn(f"フォールバックも失敗: {e3}")
                    return cudf.Series([None] * rows, dtype='string')
            
        except Exception as e:
            warnings.warn(f"文字列Series作成失敗: {e}")
            return cudf.Series([None] * rows, dtype='string')
    
    def create_string_buffers(
        self,
        columns: List[ColumnMeta],
        rows: int,
        raw_dev,
        row_positions_dev,  # 追加
        field_offsets_dev,
        field_lengths_dev
    ) -> Dict[str, Any]:
        """
        文字列バッファ作成（RMM DeviceBuffer版 - ゼロコピー実現）
        
        RMM DeviceBufferを使用してpylibcudfとの完全な互換性を実現
        """
        
        string_buffers = {}
        var_columns = [col for col in columns if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY)]
        
        if not var_columns:
            return string_buffers
        
        for col_idx, col in enumerate(var_columns):
            # 対応する列インデックスを検索
            actual_col_idx = None
            for i, c in enumerate(columns):
                if c.name == col.name:
                    actual_col_idx = i
                    break
            
            if actual_col_idx is None:
                continue
            
            try:
                # === 1. RMM DeviceBufferで長さ配列を作成 ===
                lengths_buffer = rmm.DeviceBuffer(size=rows * 4)  # int32は4バイト
                
                # CuPy配列としてビュー作成（ゼロコピー）
                lengths_cupy = cp.ndarray(
                    shape=(rows,),
                    dtype=cp.int32,
                    memptr=cp.cuda.MemoryPointer(
                        cp.cuda.UnownedMemory(lengths_buffer.ptr, lengths_buffer.size, lengths_buffer),
                        0
                    )
                )
                
                # Numbaカーネル用に変換
                d_lengths_numba = cuda.as_cuda_array(lengths_cupy)
                
                # 2次元グリッドサイズの計算
                threads_per_block = 256
                max_blocks_x = self.device_props.get('max_grid_dim_x', 2147483647)
                max_blocks_y = self.device_props.get('max_grid_dim_y', 65535)
                
                # まずは1行/1スレッドで計算
                total_threads_needed = rows
                blocks_x = min((total_threads_needed + threads_per_block - 1) // threads_per_block, max_blocks_x)
                blocks_y = (total_threads_needed + blocks_x * threads_per_block - 1) // (blocks_x * threads_per_block)
                
                # Y次元の制限を超える場合は各スレッドが複数行処理
                rows_per_thread = 1
                if blocks_y > max_blocks_y:
                    # 最大グリッドサイズでの総スレッド数
                    max_total_threads = max_blocks_x * max_blocks_y * threads_per_block
                    rows_per_thread = (rows + max_total_threads - 1) // max_total_threads
                    blocks_y = min(blocks_y, max_blocks_y)
                
                # グリッド設定
                grid = (blocks_x, blocks_y)
                
                # カーネル実行情報を出力（最初の文字列列のみ）
                if os.environ.get('GPUPGPARSER_TEST_MODE') == '1' and col_idx == 0:
                    chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
                    prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
                    print(f"\n{prefix} ============================================================")
                    print(f"{prefix} Kernel Execution Info: copy_string_data_direct (First String)")
                    print(f"{prefix} ============================================================")
                    print(f"{prefix} Column: {col.name} (index: {actual_col_idx})")
                    print(f"{prefix} Data Type: STRING")
                    print(f"{prefix} Grid Dimensions: {grid}")
                    print(f"{prefix} Block Dimensions: {threads_per_block}")
                    print(f"{prefix} Total Blocks: {blocks_x * blocks_y:,}")
                    print(f"{prefix} Total Threads: {blocks_x * blocks_y * threads_per_block:,}")
                    print(f"{prefix} Rows per Thread: {rows_per_thread}")
                    print(f"{prefix} Total Rows: {rows:,}")
                    print(f"{prefix} ============================================================\n")
                
                @cuda.jit
                def extract_lengths_coalesced(field_lengths, col_idx, lengths_out, num_rows):
                    """ワープ効率を考慮した長さ抽出"""
                    row = cuda.grid(1)
                    if row < num_rows:
                        lengths_out[row] = field_lengths[row, col_idx]
                
                extract_lengths_coalesced[grid, threads_per_block](
                    field_lengths_dev, actual_col_idx, d_lengths_numba, rows
                )
                cuda.synchronize()
                
                # === 2. GPU上でのオフセット計算（CuPy使用） ===
                # CuPyのcumsum使用（既にCuPy配列なので変換不要）
                offsets_cumsum = cp.cumsum(lengths_cupy, dtype=cp.int32)
                
                # RMM DeviceBufferでオフセット配列作成
                offsets_buffer = rmm.DeviceBuffer(size=(rows + 1) * 4)  # int32
                offsets_cupy = cp.ndarray(
                    shape=(rows + 1,),
                    dtype=cp.int32,
                    memptr=cp.cuda.MemoryPointer(
                        cp.cuda.UnownedMemory(offsets_buffer.ptr, offsets_buffer.size, offsets_buffer),
                        0
                    )
                )
                offsets_cupy[0] = 0
                offsets_cupy[1:] = offsets_cumsum
                
                # 総データサイズを取得
                total_size = int(offsets_cupy[-1])
                
                if total_size == 0:
                    string_buffers[col.name] = {'data': None, 'offsets': None, 'actual_size': 0}
                    continue
                
                # === 3. RMM DeviceBufferでデータバッファ作成 ===
                data_buffer = rmm.DeviceBuffer(size=total_size)
                
                # CuPy配列としてビュー作成（ゼロコピー）
                data_cupy = cp.ndarray(
                    shape=(total_size,),
                    dtype=cp.uint8,
                    memptr=cp.cuda.MemoryPointer(
                        cp.cuda.UnownedMemory(data_buffer.ptr, data_buffer.size, data_buffer),
                        0
                    )
                )
                
                # Numbaカーネル用に変換
                d_data_numba = cuda.as_cuda_array(data_cupy)
                d_offsets_numba = cuda.as_cuda_array(offsets_cupy)
                
                @cuda.jit
                def copy_string_data_direct(
                    raw_data, row_positions, field_offsets, field_lengths,
                    col_idx, data_out, offsets, num_rows, rows_per_thread
                ):
                    """
                    シンプルな直接コピーカーネル（2次元グリッド対応）
                    デバッグ用に最もシンプルな実装
                    """
                    # 2次元グリッドからスレッドIDを計算
                    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + \
                          cuda.blockIdx.y * cuda.gridDim.x * cuda.blockDim.x
                    
                    # 各スレッドが処理する行範囲
                    start_row = tid * rows_per_thread
                    end_row = min(start_row + rows_per_thread, num_rows)
                    
                    # 複数行を処理
                    for row in range(start_row, end_row):
                        # 相対オフセットから絶対位置を計算
                        row_start = row_positions[row]
                        relative_offset = field_offsets[row, col_idx]
                        field_offset = row_start + relative_offset  # uint64として計算
                        field_length = field_lengths[row, col_idx]
                        output_offset = offsets[row]
                        
                        # NULL チェック
                        if field_length <= 0:
                            continue
                        
                        # シンプルな直接コピー
                        for i in range(field_length):
                            src_idx = field_offset + i
                            dst_idx = output_offset + i
                            
                            # 境界チェック
                            if (src_idx < raw_data.size and 
                                dst_idx < data_out.size):
                                data_out[dst_idx] = raw_data[src_idx]
                
                copy_string_data_direct[grid, threads_per_block](
                    raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev,
                    actual_col_idx, d_data_numba, d_offsets_numba, rows, rows_per_thread
                )
                cuda.synchronize()
                
                # RMM DeviceBufferをそのまま返す（pylibcudf互換）
                string_buffers[col.name] = {
                    'data': data_buffer,      # RMM DeviceBuffer
                    'offsets': offsets_buffer, # RMM DeviceBuffer
                    'actual_size': total_size
                }
                
            except Exception as e:
                warnings.warn(f"文字列バッファ作成エラー ({col.name}): {e}")
                string_buffers[col.name] = {'data': None, 'offsets': None, 'actual_size': 0}
        
        return string_buffers


__all__ = ["DirectColumnExtractor"]