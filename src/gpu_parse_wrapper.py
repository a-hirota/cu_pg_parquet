"""
gpu_parse_wrapper.py
====================
CPU シリアルパーサーによるフォールバック実装。
GPU カーネル (`pg_parser_kernels.py`) のオフセット/長さ計算に
問題があるため、一時的に CPU でパースします。
"""

from __future__ import annotations
import os # Import os module
import numpy as np
import cupy as cp # Import cupy
from numba import cuda
import math
# Import new kernels and remove the old one
from .cuda_kernels.pg_parser_kernels import (
    count_rows_gpu,
    calculate_row_lengths_gpu,
    parse_fields_from_offsets_gpu
)

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
    # rows: int, # rows is now calculated internally
    ncols: int,
    threads_per_block: int = 256,
    header_size: int | None = None,
    # row_start_positions: cuda.cudadrv.devicearray.DeviceNDArray | None = None, # Removed
) -> tuple[cuda.cudadrv.devicearray.DeviceNDArray, cuda.cudadrv.devicearray.DeviceNDArray]:
    """
    GPU カーネルを使用して COPY BINARY データをパースし、
    フィールドのオフセットと長さを計算します (Plan C ベース)。

    Args:
        raw_dev: GPU上の生バイナリデータチャンク
        ncols: 期待される列数
        threads_per_block: カーネル実行時のブロックあたりスレッド数
        header_size: バイナリヘッダーのサイズ (Noneの場合は自動検出)

    Returns:
        (field_offsets_dev, field_lengths_dev):
            - field_offsets_dev: shape=(rows, ncols), dtype=int32 のオフセット配列 (GPU上)
            - field_lengths_dev: shape=(rows, ncols), dtype=int32 の長さ配列 (GPU上, NULL=-1)
    """
    print("[GPU PARSER v2] Starting GPU parsing (Plan C)...")

    if header_size is None:
        raw_host_for_header = raw_dev[:128].copy_to_host() # Copy slightly more for safety
        header_size = detect_pg_header_size(raw_host_for_header)
        print(f"[GPU PARSER v2] Detected header size: {header_size}")

    # 1. Count rows using GPU kernel
    print("[GPU PARSER v2] Counting rows (parallel)...")
    row_cnt_dev = cuda.device_array(1, dtype=np.int32)
    row_cnt_dev[0] = 0 # Initialize counter

    # Calculate launch configuration for parallel scan
    data_bytes = int(raw_dev.size - header_size)
    if data_bytes <= 0:
        rows = 0
    else:
        bytes_per_thread = 4096  # Heuristic: Each thread scans approx 4KB
        n_threads_needed = (data_bytes + bytes_per_thread - 1) // bytes_per_thread
        threads = threads_per_block # Use the provided threads_per_block
        # Ensure at least 1 thread is launched if data exists
        n_threads_needed = max(1, n_threads_needed)
        # Calculate blocks needed, respecting max blocks from env var
        max_blocks_env = int(os.getenv("GPUPASER_MAX_BLOCKS", "2048")) # Get max blocks from env
        blocks = min(
            (n_threads_needed + threads - 1) // threads,
            max_blocks_env
        )
        # Ensure at least 1 block is launched
        blocks = max(1, blocks)

        print(f"[GPU PARSER v2] Launch count_rows_gpu<<<{blocks}, {threads}>>>")
        count_rows_gpu[blocks, threads](raw_dev, header_size, row_cnt_dev)
        cuda.synchronize()
        rows = int(row_cnt_dev.copy_to_host()[0])

    print(f"[GPU PARSER v2] Row count: {rows}")

    if rows <= 0:
        print("[GPU PARSER v2] No rows found or error during count.")
        # Return empty arrays matching the expected shape but with 0 rows
        field_offsets_dev = cuda.device_array((0, ncols), dtype=np.int32)
        field_lengths_dev = cuda.device_array((0, ncols), dtype=np.int32)
        return field_offsets_dev, field_lengths_dev

    # 2. Calculate row lengths using GPU kernel
    print("[GPU PARSER v2] Calculating row lengths...")
    row_lengths_dev = cuda.device_array(rows, dtype=np.int64)
    blocks_per_grid_len = math.ceil(rows / threads_per_block)
    calculate_row_lengths_gpu[blocks_per_grid_len, threads_per_block](
        raw_dev, rows, header_size, row_lengths_dev
    )
    cuda.synchronize()
    # TODO: Add check for errors indicated by row_lengths_dev (e.g., -1)

    # 3. Calculate row offsets using CuPy's prefix sum (cumsum)
    print("[GPU PARSER v2] Calculating row offsets (Prefix Sum)...")
    row_lengths_cp = cp.asarray(row_lengths_dev) # Zero-copy view
    # Exclusive scan: offset[i] = sum(lengths[0]...lengths[i-1])
    # We need to prepend header_size to the cumulative sum.
    row_offsets_cp = cp.zeros(rows + 1, dtype=cp.int64) # Need rows+1 for exclusive scan result
    cp.cumsum(row_lengths_cp, out=row_offsets_cp[1:])
    row_offsets_cp += header_size # Add header size to all offsets
    # Final offsets needed for the kernel (size=rows)
    row_offsets_dev = cuda.as_cuda_array(row_offsets_cp[:-1]) # Exclude the last element (total sum)

    # 4. Allocate output arrays for field info
    field_offsets_dev = cuda.device_array((rows, ncols), dtype=np.int32)
    field_lengths_dev = cuda.device_array((rows, ncols), dtype=np.int32)

    # 5. Parse fields using the calculated row offsets
    print(f"[GPU PARSER v2] Parsing fields... grid=({blocks_per_grid_len},), block=({threads_per_block},)")
    parse_fields_from_offsets_gpu[blocks_per_grid_len, threads_per_block](
        raw_dev,
        ncols,
        rows,
        row_offsets_dev, # Pass calculated offsets
        field_offsets_dev,
        field_lengths_dev
    )
    cuda.synchronize()
    print("[GPU PARSER v2] GPU parsing finished.")

    return field_offsets_dev, field_lengths_dev


__all__ = ["parse_binary_chunk_gpu", "detect_pg_header_size"] # Keep detect_pg_header_size if still used externally
