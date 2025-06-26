"""Rust実装のArrow文字列ビルダーをPythonから使用するラッパー"""
from typing import List, Optional, Tuple
import numpy as np
import cupy as cp
from numba import cuda
import cudf

try:
    import gpupgparser_rust
except ImportError:
    raise ImportError(
        "gpupgparser_rust モジュールが見つかりません。"
        "cd rust && maturin develop を実行してください。"
    )


class RustStringBuilder:
    """
    Rust実装の高速文字列ビルダー
    
    既存のNumba実装の代替として使用可能
    """
    
    def __init__(self):
        self._buffer = None
        self._offsets = [0]
        self._data = bytearray()
    
    def add_string(self, s: bytes) -> None:
        """文字列を追加"""
        self._data.extend(s)
        self._offsets.append(len(self._data))
    
    def add_strings_batch(self, strings: List[bytes]) -> None:
        """複数の文字列を一括追加"""
        for s in strings:
            self.add_string(s)
    
    def build_gpu_buffers(self) -> Tuple[cuda.cudadrv.devicearray.DeviceNDArray, 
                                         cuda.cudadrv.devicearray.DeviceNDArray]:
        """
        GPUバッファを構築してNumba DeviceNDArrayとして返す
        
        Returns:
            (data_gpu, offsets_gpu) のタプル
        """
        # NumPy配列に変換
        offsets_np = np.array(self._offsets, dtype=np.int32)
        data_np = np.frombuffer(self._data, dtype=np.uint8)
        
        # GPUに転送
        offsets_gpu = cuda.to_device(offsets_np)
        data_gpu = cuda.to_device(data_np)
        
        return data_gpu, offsets_gpu
    
    def build_cudf_series(self) -> cudf.Series:
        """
        cuDF Seriesを直接構築
        
        Returns:
            文字列のcuDF Series
        """
        data_gpu, offsets_gpu = self.build_gpu_buffers()
        
        # 既存のbuild_cudf_from_buf.pyの_create_string_series_zero_copyと
        # 同じロジックを使用
        from ..build_cudf_from_buf import CuDFZeroCopyProcessor
        
        processor = CuDFZeroCopyProcessor()
        
        # ダミーのbuffer_info作成
        buffer_info_col = {
            'data': data_gpu,
            'offsets': offsets_gpu,
            'actual_size': len(self._data)
        }
        
        # ダミーのColumnMeta
        from ..types import ColumnMeta, UTF8
        col = ColumnMeta(name="string_col", pg_oid=25, elem_size=-1, 
                        arrow_id=UTF8, elem_size_net=0)
        
        rows = len(self._offsets) - 1
        return processor._create_string_series_zero_copy(
            col, rows, None, {'string_col': buffer_info_col}
        )
    
    @staticmethod
    def from_postgres_binary(
        binary_data: bytes, 
        column_index: int = 0
    ) -> 'RustStringBuilder':
        """
        PostgreSQL COPY BINARYデータから直接構築
        
        Args:
            binary_data: COPY BINARYデータ
            column_index: 抽出するカラムインデックス
            
        Returns:
            構築されたRustStringBuilder
        """
        # TODO: Rust側で実装して高速化
        builder = RustStringBuilder()
        
        # 簡易パーサー実装（後でRust実装に置き換え）
        # ヘッダースキップ
        pos = 11 + 4 + 4  # signature + flags + header extension
        
        while pos < len(binary_data) - 2:
            if binary_data[pos:pos+2] == b'\xff\xff':
                break
                
            # フィールド数読み取り
            field_count = int.from_bytes(binary_data[pos:pos+2], 'big')
            pos += 2
            
            # 指定されたカラムまでスキップ
            for i in range(column_index):
                field_len = int.from_bytes(binary_data[pos:pos+4], 'big', signed=True)
                pos += 4
                if field_len > 0:
                    pos += field_len
            
            # 対象カラムを読み取り
            field_len = int.from_bytes(binary_data[pos:pos+4], 'big', signed=True)
            pos += 4
            
            if field_len == -1:
                # NULL
                builder.add_string(b'')
            else:
                field_data = binary_data[pos:pos+field_len]
                builder.add_string(field_data)
                pos += field_len
            
            # 残りのフィールドをスキップ
            for i in range(column_index + 1, field_count):
                field_len = int.from_bytes(binary_data[pos:pos+4], 'big', signed=True)
                pos += 4
                if field_len > 0:
                    pos += field_len
        
        return builder