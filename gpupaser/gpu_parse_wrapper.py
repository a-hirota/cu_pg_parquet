"""
gpu_parse_wrapper.py
====================
CPU シリアルパーサーによるフォールバック実装。
GPU カーネル (`pg_parser_kernels.py`) のオフセット/長さ計算に
問題があるため、一時的に CPU でパースします。
"""

from __future__ import annotations
import numpy as np
from numba import cuda
import math
from .cuda_kernels.pg_parser_kernels import parse_binary_format_kernel_one_row # GPU Kernel import uncommented

def detect_pg_header_size(raw_data: np.ndarray) -> int:
    """
    COPY BINARY フォーマットのヘッダーサイズを検出
    """
    header_size = 11
    if len(raw_data) < header_size:
        return header_size
    if raw_data[0] != 80 or raw_data[1] != 71:  # 'P','G'
        return header_size
    # 拡張ヘッダー長
    if len(raw_data) >= header_size + 4:
        ext_len = int.from_bytes(raw_data[header_size:header_size+4], 'big', signed=False)
        if ext_len > 0 and len(raw_data) >= header_size + 4 + ext_len:
            header_size += 4 + ext_len
    return header_size


def parse_binary_chunk_gpu(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
    rows: int,
    ncols: int,
    threads_per_block: int = 256, # Parameter kept for interface compatibility
    header_size: int = None,
    row_start_positions: cuda.cudadrv.devicearray.DeviceNDArray | None = None,
) -> tuple[cuda.cudadrv.devicearray.DeviceNDArray, cuda.cudadrv.devicearray.DeviceNDArray]:
    """
    GPU カーネルを使用して COPY BINARY データをパースし、
    フィールドのオフセットと長さを計算します。

    Args:
        raw_dev: GPU上の生バイナリデータチャンク
        rows: 期待される行数
        ncols: 期待される列数
        threads_per_block: カーネル実行時のブロックあたりスレッド数
        header_size: バイナリヘッダーのサイズ (Noneの場合は自動検出)
        row_start_positions: 各行の開始位置を示すGPU配列 (Noneの場合はカーネル内で計算)

    Returns:
        (field_offsets_dev, field_lengths_dev):
            - field_offsets_dev: shape=(rows, ncols), dtype=int32 のオフセット配列 (GPU上)
            - field_lengths_dev: shape=(rows, ncols), dtype=int32 の長さ配列 (GPU上, NULL=-1)
    """
    print("[GPU PARSER] Starting GPU offset/length parsing...") # DEBUG

    if header_size is None:
        # ヘッダーサイズ検出のために一時的にホストにコピー (最適化の余地あり)
        # Note: This copy adds overhead. If header_size is known beforehand, pass it.
        raw_host_for_header = raw_dev[:100].copy_to_host() # Copy only a small part
        header_size = detect_pg_header_size(raw_host_for_header)
        print(f"[GPU PARSER] Detected header size: {header_size}") # DEBUG

    # Allocate output arrays on the GPU
    field_offsets_dev = cuda.device_array((rows, ncols), dtype=np.int32)
    field_lengths_dev = cuda.device_array((rows, ncols), dtype=np.int32)

    # Calculate grid and block dimensions
    # We use a 1D grid where each thread processes one row
    blocks_per_grid = math.ceil(rows / threads_per_block)

    print(f"[GPU PARSER] Launching kernel: grid=({blocks_per_grid},), block=({threads_per_block},)") # DEBUG
    print(f"[GPU PARSER] Args: rows={rows}, ncols={ncols}, header_size={header_size}, row_starts provided={row_start_positions is not None}") # DEBUG

    # Launch the GPU kernel
    parse_binary_format_kernel_one_row[blocks_per_grid, threads_per_block](
        raw_dev,
        field_offsets_dev,
        field_lengths_dev,
        ncols,
        header_size,
        row_start_positions # Pass None if not provided
    )

    # Synchronize to ensure kernel completion before returning results
    cuda.synchronize()
    print("[GPU PARSER] GPU kernel finished.") # DEBUG

    return field_offsets_dev, field_lengths_dev


__all__ = ["parse_binary_chunk_gpu", "detect_pg_header_size"]
