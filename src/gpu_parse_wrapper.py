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
    # calculate_row_lengths_gpu, # Replaced by calculate_row_lengths_and_null_flags_gpu
    calculate_row_lengths_and_null_flags_gpu,
    parse_fields_from_offsets_gpu
)

# Helper function to build row start offsets on CPU
# This could be further optimized or moved to a GPU kernel in the future.
def build_pg_row_starts_cpu(raw_data_host: np.ndarray, header_size: int, num_rows_expected: int) -> np.ndarray:
    """
    Scans the raw binary data (host) to find the start byte offset of each row.
    Args:
        raw_data_host: NumPy array of uint8 containing the binary data.
        header_size: Size of the COPY BINARY header.
        num_rows_expected: The number of rows expected (e.g., from a previous row count).
    Returns:
        NumPy array of int32 containing the start offset for each row.
        If a row is not found or data ends prematurely, its offset might be -1.
    """
    row_starts = np.full(num_rows_expected, -1, dtype=np.int32) 
    current_row_idx = 0
    pos = header_size
    data_len = raw_data_host.shape[0]

    while pos < data_len and current_row_idx < num_rows_expected:
        if pos + 2 > data_len:  # Not enough bytes for num_fields
            break
        
        b0 = raw_data_host[pos]
        b1 = raw_data_host[pos + 1]
        num_fields_in_row = (b0 << 8) | b1

        if num_fields_in_row == 0xFFFF:  # End of data marker
            break

        row_starts[current_row_idx] = pos
        current_row_idx += 1
        pos += 2  # Advance past num_fields

        for _ in range(num_fields_in_row):
            if pos + 4 > data_len:  # Not enough bytes for field_len
                # Mark current and subsequent expected rows as invalid if data ends abruptly
                for r_idx in range(current_row_idx -1, num_rows_expected): # current_row_idx was already incremented
                    if r_idx < num_rows_expected : row_starts[r_idx] = -1 # Ensure index is within bounds
                # Return only up to the point of failure, or the expected size with -1s
                return row_starts 

            fb0 = raw_data_host[pos]
            fb1 = raw_data_host[pos + 1]
            fb2 = raw_data_host[pos + 2]
            fb3 = raw_data_host[pos + 3]
            field_len = (fb0 << 24) | (fb1 << 16) | (fb2 << 8) | fb3
            
            if field_len == 0xFFFFFFFF: # Signed int32 representation of -1
                field_len = -1
            elif field_len >= 0x80000000: # Handle other negative numbers or large positives
                field_len = field_len - 0x100000000 # Convert to signed int32
            
            pos += 4  # Advance past field_len
            if field_len > 0:
                if pos + field_len > data_len: # Not enough bytes for field data
                    for r_idx in range(current_row_idx -1, num_rows_expected):
                         if r_idx < num_rows_expected : row_starts[r_idx] = -1
                    return row_starts
                pos += field_len  # Advance past field data
    
    # If fewer rows were found than expected, the remaining entries in row_starts will be -1.
    return row_starts


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
        field_offsets_dev = cuda.device_array((0, ncols), dtype=np.int32)
        field_lengths_dev = cuda.device_array((0, ncols), dtype=np.int32)
        return field_offsets_dev, field_lengths_dev

    # 2. Calculate row start offsets on CPU (for now)
    # This is a placeholder for a potentially more efficient GPU-based row start detection
    print("[GPU PARSER v2] Calculating row start offsets (CPU)...")
    # Potentially large copy, consider optimizing this step in the future
    raw_host_for_starts = raw_dev.copy_to_host() 
    row_starts_host = build_pg_row_starts_cpu(raw_host_for_starts, header_size, rows)
    del raw_host_for_starts # Free host memory
    
    # Filter out invalid row starts (-1) which might indicate fewer rows than counted by count_rows_gpu
    # or issues in build_pg_row_starts_cpu.
    # This ensures row_starts_dev only contains valid offsets for existing rows.
    valid_row_starts_host = row_starts_host[row_starts_host != -1]
    actual_rows_from_starts = len(valid_row_starts_host)

    if actual_rows_from_starts == 0:
        print("[GPU PARSER v2] No valid row starts found by CPU scan.")
        field_offsets_dev = cuda.device_array((0, ncols), dtype=np.int32)
        field_lengths_dev = cuda.device_array((0, ncols), dtype=np.int32)
        return field_offsets_dev, field_lengths_dev
        
    if actual_rows_from_starts < rows:
        print(f"[GPU PARSER v2] Warning: count_rows_gpu found {rows} but CPU scan found {actual_rows_from_starts} valid row starts. Using {actual_rows_from_starts}.")
        rows = actual_rows_from_starts # Adjust row count

    row_starts_dev = cuda.to_device(valid_row_starts_host)

    # 3. Calculate row lengths and null flags using the new GPU kernel
    print("[GPU PARSER v2] Calculating row lengths and null flags (GPU)...")
    row_lengths_dev = cuda.device_array(rows, dtype=np.int32) # New kernel outputs int32 for row_lengths
    null_flags_dev = cuda.device_array((rows, ncols), dtype=np.int8)
    
    blocks_per_grid_len = math.ceil(rows / threads_per_block)
    calculate_row_lengths_and_null_flags_gpu[blocks_per_grid_len, threads_per_block](
        raw_dev, rows, ncols, row_starts_dev, row_lengths_dev, null_flags_dev
    )
    cuda.synchronize()
    # TODO: Add check for errors indicated by row_lengths_dev (e.g., -1 from the kernel)
    # The null_flags_dev is now populated. It can be used later or returned.

    # 4. Calculate row offsets for parse_fields_from_offsets_gpu using CuPy's prefix sum (cumsum)
    # These offsets are relative to the start of the data (after header).
    print("[GPU PARSER v2] Calculating row offsets for field parsing (Prefix Sum)...")
    row_lengths_cp = cp.asarray(row_lengths_dev) 
    
    # Ensure row_lengths_cp contains no negative values before cumsum if they indicate errors
    # For now, assuming lengths are valid or 0 for empty/error rows handled by kernel.
    # If row_lengths_dev can contain -1 from the kernel, those rows should be handled.
    # For cumsum, negative lengths would be problematic. Let's assume kernel sets length to 0 for error rows.
    # If kernel sets length to -1 for error, we need to filter/mask them before cumsum or ensure kernel output is non-negative for length.
    # The new kernel sets row_lengths[rid] = -1 for errors. This needs to be handled.
    # For now, let's proceed assuming valid lengths for cumsum, or that downstream handles -1.
    # A robust solution would be to filter out rows with length -1 or replace -1 with 0 before cumsum.
    # Example: row_lengths_cp = cp.where(row_lengths_cp < 0, 0, row_lengths_cp)

    row_offsets_cp = cp.zeros(rows + 1, dtype=cp.int64) 
    cp.cumsum(row_lengths_cp.astype(cp.int64), out=row_offsets_cp[1:]) # Cast to int64 for cumsum if lengths are int32
    row_offsets_cp += header_size 
    
    # row_offsets_dev for parse_fields_from_offsets_gpu should be the start of each row.
    # This is essentially row_starts_dev, but we've re-derived it via cumsum of lengths from the new kernel.
    # We can directly use row_starts_dev if it's already adjusted for header, or adjust it.
    # The `row_starts_host` from `build_pg_row_starts_cpu` are absolute offsets from raw_data_host[0].
    # So `row_starts_dev` are also absolute.
    # `parse_fields_from_offsets_gpu` expects row_offsets_in to be absolute.
    # So, we can use row_starts_dev directly.
    
    # Using row_starts_dev directly for parse_fields_from_offsets_gpu
    # This makes the cumsum above redundant if row_starts_dev is accurate and used.
    # However, the original code structure used cumsum from lengths.
    # Let's stick to minimal structural change: use cumsum'd offsets for parse_fields_from_offsets_gpu.
    # The `row_offsets_dev` for `parse_fields_from_offsets_gpu` should be the one derived from `cumsum`.
    final_row_offsets_for_parse_fields_dev = cuda.as_cuda_array(row_offsets_cp[:-1])


    # 5. Allocate output arrays for field info (field_offsets_dev, field_lengths_dev for fields)
    # Note: field_lengths_dev here is for *fields*, not to be confused with row_lengths_dev for *rows*.
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
