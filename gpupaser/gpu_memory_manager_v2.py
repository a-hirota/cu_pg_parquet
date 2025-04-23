"""
GPU メモリ確保ロジック (Arrow ColumnMeta ベース, v2)

サポート範囲
-------------
* 固定長型 : INT{16,32,64}, FLOAT{32,64}, DECIMAL128, BOOL, DATE32, TS64_US
* 可変長型 : UTF8, BINARY
  - NUMERIC は今回は UTF8 として扱う (可変長文字列)
* Arrow 型 ID や各型長は ``type_map.ColumnMeta`` / ``arrow_utils`` で管理

API
-----
GPUMemoryManagerV2.initialize_device_buffers(columns, rows_in_batch) -> dict
    * 各カラム名 → device array (cuda.devicearray.DeviceNDArray) を返却
    * 補助メタ : ``type_ids`` / ``elem_sizes`` / ``param1`` / ``param2`` も返す
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np

from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError

from .type_map import (
    ColumnMeta,
    UTF8,
    BINARY,
    DECIMAL128,
)
from .arrow_utils import (
    arrow_elem_size,
    build_gpu_meta_arrays,
)


# ----------------------------------------------------------------------
#  GPU メモリマネージャ
# ----------------------------------------------------------------------
class GPUMemoryManagerV2:
    """
    Arrow ColumnMeta をもとに GPU バッファを確保する軽量クラス
    """

    def __init__(self):
        try:
            # 既存コンテキストがあれば流用
            try:
                cuda.current_context()
                print("[GPUMemoryManagerV2] existing CUDA context")
            except cuda.cudadrv.error.CudaSupportError:
                cuda.select_device(0)
                print("[GPUMemoryManagerV2] new CUDA context created")
            self.print_gpu_memory_info()
        except Exception as e:
            raise RuntimeError(f"CUDA init failed: {e}") from e

    # ------------------------
    # public
    # ------------------------
    def initialize_device_buffers(
        self,
        columns: List[ColumnMeta],
        rows: int,
    ) -> Dict[str, Any]:
        """
        ColumnMeta に従い各列のバッファを確保し GPU へ配置

        Returns
        -------
        dict
          {
            '<colname>': device_array / (offsets, values, nulls) tuple for varlen,
            'type_ids': np.ndarray[int32],
            'elem_sizes': np.ndarray[int32],
            'param1': np.ndarray[int32],
            'param2': np.ndarray[int32],
          }
        """
        type_ids, elem_sizes, param1, param2 = build_gpu_meta_arrays(columns)

        buffers: Dict[str, Any] = {}

        # 固定長 & 可変長の確保
        for meta in columns:
            aid = meta.arrow_id
            if aid in (UTF8, BINARY):
                # 可変長列: 現時点では単純に rows * max_len の連続バッファ
                # 長さは param1 if set else 256
                max_len = meta.arrow_param or 256
                total_bytes = rows * max_len

                try:
                    d_values = cuda.device_array(total_bytes, dtype=np.uint8)
                    d_nulls = cuda.device_array(rows, dtype=np.uint8)
                except CudaAPIError as e:
                    self._cleanup_partial(buffers)
                    raise RuntimeError(f"GPU alloc failed (varlen {meta.name}): {e}") from e

                buffers[meta.name] = (d_values, d_nulls, max_len)

            else:
                esize = arrow_elem_size(aid)
                if esize == 0:
                    # DECIMAL128 を固定長 16 byte として扱う
                    if aid == DECIMAL128:
                        esize = 16
                    else:
                        raise ValueError(f"Unsupported fixed type size for aid={aid}")

                try:
                    d_values = cuda.device_array(rows, dtype=self._dtype_for_size(esize))
                    d_nulls = cuda.device_array(rows, dtype=np.uint8)
                except CudaAPIError as e:
                    self._cleanup_partial(buffers)
                    raise RuntimeError(f"GPU alloc failed (fixed {meta.name}): {e}") from e

                buffers[meta.name] = (d_values, d_nulls)

        # メタ配列はホスト側 numpy で保持 (caller が必要に応じて GPU 転送)
        buffers["type_ids"] = type_ids
        buffers["elem_sizes"] = elem_sizes
        buffers["param1"] = param1
        buffers["param2"] = param2

        return buffers

    # ------------------------
    # helpers
    # ------------------------
    @staticmethod
    def _dtype_for_size(esize: int):
        if esize == 1:
            return np.uint8
        if esize == 2:
            return np.int16
        if esize == 4:
            return np.int32
        if esize == 8:
            return np.int64
        if esize == 16:
            # Decimal128: store as two int64 fields packed -> use np.int64[rows,2]?
            # simplify: raw bytes
            return np.uint8
        raise ValueError(f"Unsupported element size {esize}")

    @staticmethod
    def _cleanup_partial(bufs: Dict[str, Any]):
        bufs.clear()
        cuda.synchronize()

    @staticmethod
    def print_gpu_memory_info():
        try:
            free_b, total_b = cuda.current_context().get_memory_info()
            print(
                f"[GPU MEM] free={free_b/1024**2:.1f}MB / total={total_b/1024**2:.1f}MB "
                f"({(total_b-free_b)/total_b*100:.1f}% used)"
            )
        except Exception:
            pass


__all__ = ["GPUMemoryManagerV2"]
