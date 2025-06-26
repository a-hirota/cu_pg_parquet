"""PostgreSQL COPY BINARY → GPU 転送 (Rust実装使用)"""
import os
from typing import Dict, Any, Optional
import numpy as np
import cupy as cp
from numba import cuda
import pylibcudf as plc
import cudf
import rmm

try:
    import gpupgparser_rust
except ImportError:
    raise ImportError(
        "gpupgparser_rust モジュールが見つかりません。"
        "cd rust && maturin develop を実行してください。"
    )


class PostgresGPUReader:
    """PostgreSQLからGPUへの直接データ転送"""
    
    def __init__(self, dsn: Optional[str] = None):
        """
        初期化
        
        Args:
            dsn: PostgreSQL接続文字列。Noneの場合は環境変数から取得
        """
        if dsn is None:
            dsn = os.environ.get('GPUPASER_PG_DSN')
            if not dsn:
                raise ValueError("DSNが指定されていません。環境変数GPUPASER_PG_DSNを設定してください。")
        self.dsn = dsn
    
    def fetch_string_column_to_gpu(
        self, 
        query: str, 
        column_index: int = 0
    ) -> Dict[str, Any]:
        """
        PostgreSQLから文字列カラムをGPUに直接転送
        
        Args:
            query: COPY (SELECT ...) TO STDOUT WITH BINARY形式のクエリ
            column_index: 取得するカラムのインデックス
            
        Returns:
            GPUバッファ情報を含む辞書
        """
        return gpupgparser_rust.fetch_postgres_copy_binary_to_gpu(
            self.dsn, query, column_index
        )
    
    def create_cudf_series_from_gpu_buffers(
        self, 
        gpu_buffers: Dict[str, Any]
    ) -> cudf.Series:
        """
        GPUバッファからcuDF Seriesを作成（ゼロコピー）
        
        Args:
            gpu_buffers: fetch_string_column_to_gpuの戻り値
            
        Returns:
            cuDF Series
        """
        data_ptr = gpu_buffers['data_ptr']
        data_size = gpu_buffers['data_size']
        offsets_ptr = gpu_buffers['offsets_ptr']
        offsets_size = gpu_buffers['offsets_size']
        row_count = gpu_buffers['row_count']
        
        # CuPy配列としてGPUメモリをラップ（所有権は移転しない）
        data_cupy = cp.ndarray(
            shape=(data_size,),
            dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(data_ptr, data_size, None),
                0
            )
        )
        
        offsets_cupy = cp.ndarray(
            shape=(row_count + 1,),
            dtype=cp.int32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(offsets_ptr, offsets_size, None),
                0
            )
        )
        
        # pylibcudfを使用してゼロコピー変換
        return self._create_string_series_pylibcudf(
            data_cupy, offsets_cupy, row_count
        )
    
    def _create_string_series_pylibcudf(
        self,
        data_cupy: cp.ndarray,
        offsets_cupy: cp.ndarray,
        row_count: int
    ) -> cudf.Series:
        """pylibcudfを使用した文字列Series作成"""
        # RMM DeviceBufferに変換
        offsets_bytes = offsets_cupy.tobytes()
        offsets_buf = rmm.DeviceBuffer.to_device(offsets_bytes)
        
        data_bytes = data_cupy.tobytes()
        chars_buf = rmm.DeviceBuffer.to_device(data_bytes)
        
        # pylibcudf Column構築
        offsets_mv = plc.gpumemoryview(offsets_buf)
        offsets_col = plc.column.Column(
            plc.types.DataType(plc.types.TypeId.INT32),
            len(offsets_cupy),
            offsets_mv,
            None,  # mask
            0,     # null_count
            0,     # offset
            []     # children
        )
        
        chars_mv = plc.gpumemoryview(chars_buf)
        string_col = plc.column.Column(
            plc.types.DataType(plc.types.TypeId.STRING),
            row_count,
            chars_mv,
            None,
            0,
            0,
            [offsets_col]
        )
        
        return cudf.Series.from_pylibcudf(string_col)
    
    def transfer_binary_to_gpu(self, binary_data: bytes) -> Dict[str, Any]:
        """
        バイナリデータをGPUに転送（Numba連携用）
        
        Args:
            binary_data: PostgreSQL COPY BINARYデータ
            
        Returns:
            GPUポインタ情報
        """
        return gpupgparser_rust.transfer_to_gpu_numba(binary_data)