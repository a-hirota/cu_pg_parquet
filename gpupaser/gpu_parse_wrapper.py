"""
gpu_parse_wrapper.py
====================
`cuda_kernels.pg_parser_kernels.parse_binary_format_kernel_one_row`
を呼び出し、行 × 列の `field_offsets` / `field_lengths` Device 配列を返す
ヘルパ。

テスト・プロトタイプ用途を想定しており、実運用では chunk サイズ・行数に
合わせたメモリ再利用などを追加実装する。
"""

from __future__ import annotations

import numpy as np
from numba import cuda
from .cuda_kernels.pg_parser_kernels import parse_binary_format_kernel_one_row


def parse_binary_chunk_gpu(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # uint8[:]
    rows: int,
    ncols: int,
    threads_per_block: int = 256,
):
    """
    Parameters
    ----------
    raw_dev : DeviceNDArray(uint8)
        COPY BINARY チャンク（GPU メモリ上）
    rows : int
        行数 (LIMIT で既知、または最大想定行数)
    ncols : int
        カラム数 (RowDescription で取得済み)
    threads_per_block : int, optional
        CUDA スレッド数

    Returns
    -------
    field_offsets_dev : DeviceNDArray(int32)[rows, ncols]
    field_lengths_dev : DeviceNDArray(int32)[rows, ncols]
    """
    blocks = (rows + threads_per_block - 1) // threads_per_block

    field_offsets_dev = cuda.device_array((rows, ncols), dtype=np.int32)
    field_lengths_dev = cuda.device_array((rows, ncols), dtype=np.int32)

    # 実カーネルは 1 行 1 スレッドなので blockDim.x = threads_per_block
    dummy = cuda.device_array(1, dtype=np.int32)  # unused
    parse_binary_format_kernel_one_row[blocks, threads_per_block](
        raw_dev,
        field_offsets_dev,
        field_lengths_dev,
        ncols,
        11,            # header_size (固定 COPY バイナリヘッダ 11B)
        dummy,         # row_start_positions (未使用, None)
    )
    cuda.synchronize()

    return field_offsets_dev, field_lengths_dev


__all__ = ["parse_binary_chunk_gpu"]
