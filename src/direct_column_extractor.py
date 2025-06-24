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

from .types import (
    ColumnMeta, INT16, INT32, INT64, FLOAT32, FLOAT64, DECIMAL128,
    UTF8, BINARY, DATE32, TS64_US, BOOL, UNKNOWN
)
from .cuda_kernels.decimal_tables import (
    POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
)


class DirectColumnExtractor:
    """入力データから直接列を抽出するプロセッサー"""
    
    def __init__(self):
        """初期化"""
        self.d_pow10_table_lo = None
        self.d_pow10_table_hi = None
    
    def _ensure_decimal_tables(self):
        """Decimal処理用テーブルの遅延初期化"""
        if self.d_pow10_table_lo is None:
            self.d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
            self.d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)
    
    def extract_columns_direct(
        self,
        raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        field_offsets_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        field_lengths_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        columns: List[ColumnMeta],
        string_buffers: Optional[Dict[str, Any]] = None
    ) -> cudf.DataFrame:
        """
        入力データから直接列を抽出してcuDF DataFrameを作成
        
        統合バッファを経由せず、メモリ効率的に処理
        """
        rows, ncols = field_offsets_dev.shape
        cudf_series_dict = {}
        
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
                        raw_dev, field_offsets_dev, field_lengths_dev,
                        col, col_idx, rows
                    )
                
                cudf_series_dict[col.name] = series
                
            except Exception as e:
                warnings.warn(f"列 {col.name} の直接抽出でエラー: {e}")
                cudf_series_dict[col.name] = cudf.Series([None] * rows)
        
        return cudf.DataFrame(cudf_series_dict)
    
    def _extract_fixed_column_direct(
        self,
        raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        field_offsets_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        field_lengths_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        col: ColumnMeta,
        col_idx: int,
        rows: int
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
        
        # 直接抽出カーネルを実行
        threads = 256
        blocks = (rows + threads - 1) // threads
        
        if col.arrow_id == DECIMAL128:
            # Decimal専用カーネル
            self._extract_decimal_direct[blocks, threads](
                raw_dev, field_offsets_dev, field_lengths_dev,
                col_idx, column_buffer, null_mask,
                self.d_pow10_table_lo, self.d_pow10_table_hi,
                decimal_scale, rows
            )
        else:
            # その他の固定長型用カーネル
            # arrow_idの値を整数として渡す
            arrow_type_id = int(col.arrow_id)
            self._extract_fixed_direct[blocks, threads](
                raw_dev, field_offsets_dev, field_lengths_dev,
                col_idx, arrow_type_id, col_size, 
                column_buffer, null_mask, rows
            )
        
        cuda.synchronize()
        
        # cuDF Series作成（RMM DeviceBufferを直接使用）
        return self._create_series_from_rmm_buffer(
            column_buffer_rmm, null_mask_rmm, col, rows
        )
    
    @staticmethod
    @cuda.jit
    def _extract_fixed_direct(
        raw_data, field_offsets, field_lengths,
        col_idx, arrow_type, col_size,
        output_buffer, null_mask, rows
    ):
        """固定長データの直接抽出カーネル"""
        row = cuda.grid(1)
        if row >= rows:
            return
        
        # フィールド情報取得
        src_offset = field_offsets[row, col_idx]
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
        raw_data, field_offsets, field_lengths,
        col_idx, output_buffer, null_mask,
        d_pow10_table_lo, d_pow10_table_hi,
        target_scale, rows
    ):
        """Decimal128データの直接抽出カーネル"""
        row = cuda.grid(1)
        if row >= rows:
            return
        
        # フィールド情報取得
        src_offset = field_offsets[row, col_idx]
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
                return
            
            nd = (raw_data[src_offset] << 8) | raw_data[src_offset + 1]
            weight = (raw_data[src_offset + 2] << 8) | raw_data[src_offset + 3]
            sign = (raw_data[src_offset + 4] << 8) | raw_data[src_offset + 5]
            dscale = (raw_data[src_offset + 6] << 8) | raw_data[src_offset + 7]
            
            # NaN処理
            if sign == 0xC000:
                for i in range(16):
                    if dst_offset + i < output_buffer.size:
                        output_buffer[dst_offset + i] = 0
                return
            
            # 桁数制限
            if nd > 9:
                for i in range(16):
                    if dst_offset + i < output_buffer.size:
                        output_buffer[dst_offset + i] = 0
                return
            
            current_offset = src_offset + 8
            
            # 基数10000から128ビット整数への変換
            val_hi = 0
            val_lo = 0
            
            # 基数10000桁読み取り
            for digit_idx in range(nd):
                if current_offset + 2 > raw_data.size:
                    break
                
                digit = (raw_data[current_offset] << 8) | raw_data[current_offset + 1]
                current_offset += 2
                
                # val = val * 10000 + digit
                # 簡略化実装（完全な128ビット演算は複雑なため）
                val_lo = val_lo * 10000 + digit
                # TODO: 完全な128ビット演算実装
            
            # スケール調整（簡略化）
            # TODO: 完全なスケール調整実装
            
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


__all__ = ["DirectColumnExtractor"]