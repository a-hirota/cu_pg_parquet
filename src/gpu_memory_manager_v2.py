"""GPU メモリ確保 (Arrow ColumnMeta ベース, v2)"""

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
                    # Initial allocation with max_len (will be replaced later)
                    d_values = cuda.device_array(total_bytes, dtype=np.uint8)
                    d_nulls = cuda.device_array(rows, dtype=np.uint8)
                    # Allocate offset buffer (rows + 1 for Arrow standard)
                    d_offsets = cuda.device_array(rows + 1, dtype=np.int32)
                except CudaAPIError as e:
                    self._cleanup_partial(buffers)
                    raise RuntimeError(f"GPU alloc failed (varlen {meta.name}): {e}") from e

                # Store all three buffers + initial max_len
                buffers[meta.name] = (d_values, d_nulls, d_offsets, max_len)

            else:
                esize = arrow_elem_size(aid)
                if esize == 0:
                    # DECIMAL128 を固定長 16 byte として扱う
                    if aid == DECIMAL128:
                        esize = 16
                    else:
                        raise ValueError(f"Unsupported fixed type size for aid={aid}")

                # 固定長列もバイト配列として確保（stride計算のブレを防止）
                if meta.pg_oid in (20, 21, 23):  # int8, int2, int4 
                    alloc_size = meta.elem_size
                    total_bytes = rows * alloc_size
                else:
                    alloc_size = esize
                    total_bytes = rows * esize
                try:
                    d_values = cuda.device_array(total_bytes, dtype=np.uint8)
                    d_nulls = cuda.device_array(rows, dtype=np.uint8)
                except CudaAPIError as e:
                    self._cleanup_partial(buffers)
                    raise RuntimeError(f"GPU alloc failed (fixed {meta.name}): {e}") from e

                # 可変長列と同じく3要素タプル（stride=実際の確保サイズ）として統一
                # このstrideはpass2_scatter_fixed中での行アドレス計算に使用される
                # For fixed-length, store (values, nulls, stride)
                buffers[meta.name] = (d_values, d_nulls, alloc_size)

        # メタ配列はホスト側 numpy で保持 (caller が必要に応じて GPU 転送)
        buffers["type_ids"] = type_ids
        buffers["elem_sizes"] = elem_sizes
        buffers["param1"] = param1
        buffers["param2"] = param2

        # Keep track of allocated buffers for potential cleanup/replacement
        self._allocated_buffers = buffers

        return buffers

    def replace_varlen_data_buffer(self, column_name: str, new_size: int):
        """
        指定された可変長列のデータバッファを指定サイズで再確保し、
        内部のバッファ辞書を更新する。
        """
        if column_name not in self._allocated_buffers:
            raise ValueError(f"Column '{column_name}' not found in allocated buffers.")

        current_tuple = self._allocated_buffers[column_name]
        if len(current_tuple) != 4: # Should be (d_values, d_nulls, d_offsets, max_len)
             raise TypeError(f"Buffer entry for '{column_name}' is not a variable-length tuple.")

        # old_data_buffer = current_tuple[0] # No need to explicitly free with Numba's context management?

        try:
            print(f"[GPUMemoryManagerV2] Reallocating data buffer for '{column_name}' to size {new_size}")
            new_data_buffer = cuda.device_array(max(1, new_size), dtype=np.uint8) # Ensure size >= 1
        except CudaAPIError as e:
            # Attempt cleanup before raising
            self._cleanup_partial(self._allocated_buffers)
            raise RuntimeError(f"GPU re-allocation failed for varlen data buffer '{column_name}': {e}") from e

        # Update the buffer dictionary with the new data buffer
        self._allocated_buffers[column_name] = (new_data_buffer, current_tuple[1], current_tuple[2], current_tuple[3])
        print(f"[GPUMemoryManagerV2] Reallocation successful for '{column_name}'.")
        # Return the new buffer for convenience, although the internal dict is updated
        return new_data_buffer


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

    def _cleanup_partial(self, bufs: Dict[str, Any]):
        """Clean up allocated device arrays stored in the dictionary."""
        print("[GPUMemoryManagerV2] Cleaning up partially allocated buffers...")
        # Numba handles context and memory freeing, just clear the dict
        bufs.clear()
        # No explicit free needed for DeviceNDArray objects when they go out of scope
        # or the context is destroyed.
        # cuda.synchronize() # Synchronization might be needed depending on usage context

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
