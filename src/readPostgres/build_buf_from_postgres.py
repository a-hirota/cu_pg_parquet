"""GPU Parse Wrapper: PostgreSQL COPY BINARY parser"""

from __future__ import annotations
import os
import math
import cupy as cp
import numpy as np
from numba import cuda

# Debug flags
GPUPGPARSER_DEBUG_KERNELS_WRAPPER = os.environ.get("GPUPGPARSER_DEBUG_KERNELS", "0").lower() in ("1", "true")
DEBUG_ARRAY_SIZE_WRAPPER = 1024

# GPU kernels
from .cuda_kernels.pg_parser_kernels import (
    count_rows_gpu,
    calculate_row_lengths_and_null_flags_gpu,
    parse_fields_from_offsets_gpu,
    find_row_start_offsets_gpu,
)

def build_pg_row_starts_cpu(
    raw_data_host: np.ndarray, header_size: int, num_rows_expected: int
) -> np.ndarray:
    """Return byte offsets for each row (-1 if missing)"""
    row_starts = np.full(num_rows_expected, -1, np.int32)
    pos, cur_row, n = header_size, 0, raw_data_host.size

    while pos < n and cur_row < num_rows_expected:
        if pos + 2 > n:
            break
        num_fields = (raw_data_host[pos] << 8) | raw_data_host[pos + 1]
        if num_fields == 0xFFFF:
            break
        row_starts[cur_row] = pos
        cur_row += 1
        pos += 2  # num_fields
        for _ in range(num_fields):
            if pos + 4 > n:
                row_starts[cur_row - 1 :] = -1
                return row_starts
            fld_len = int.from_bytes(raw_data_host[pos : pos + 4], "big", signed=True)
            pos += 4
            if fld_len > 0:
                if pos + fld_len > n:
                    row_starts[cur_row - 1 :] = -1
                    return row_starts
                pos += fld_len
    return row_starts

def detect_pg_header_size(raw_data: np.ndarray) -> int:
    """Detect COPY BINARY header size"""
    base = 11
    if raw_data.size < base:
        return base

    sig = b"PGCOPY\n\377\r\n\0"
    if not np.array_equal(raw_data[:11], np.frombuffer(sig, np.uint8)):
        return base

    size = base + 4  # flags
    if raw_data.size < size + 4:
        return size
    ext_len = int.from_bytes(raw_data[size : size + 4], "big")
    size += 4 + ext_len if raw_data.size >= size + 4 + ext_len else 0
    return size

def parse_binary_chunk_gpu(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
    ncols: int,
    threads_per_block: int = 256,
    header_size: int | None = None,
):
    """Parse COPY BINARY on GPU"""

    if header_size is None:
        header_size = detect_pg_header_size(raw_dev[:128].copy_to_host())

    # Row count
    data_bytes = int(raw_dev.size - header_size)
    if data_bytes <= 0:
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)

    bytes_per_thread = 4096
    n_threads = (data_bytes + bytes_per_thread - 1) // bytes_per_thread
    threads = threads_per_block
    blocks = max(1, min((n_threads + threads - 1) // threads, int(os.getenv("GPUPASER_MAX_BLOCKS", "2048"))))

    row_cnt_dev = cuda.device_array(1, np.int32)
    row_cnt_dev[0] = 0

    # Debug arrays
    from .cuda_kernels.pg_parser_kernels import DEBUG_ARRAY_SIZE as KERNEL_DEBUG_ARRAY_SIZE

    if GPUPGPARSER_DEBUG_KERNELS_WRAPPER and KERNEL_DEBUG_ARRAY_SIZE > 0:
        dbg_arr_count = cuda.device_array(KERNEL_DEBUG_ARRAY_SIZE * 5, np.int32)
        dbg_idx_count = cuda.device_array(1, np.int32)
        dbg_idx_count[0] = 0
    else:
        dbg_arr_count = cuda.device_array(1 * 5, np.int32)
        dbg_idx_count = cuda.device_array(1, np.int32)
        dbg_idx_count[0] = 0
        
    count_rows_gpu[blocks, threads](raw_dev, header_size, row_cnt_dev, dbg_arr_count, dbg_idx_count)
    cuda.synchronize()

    rows = int(row_cnt_dev.copy_to_host()[0])
    if rows <= 0:
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)

    # Row starts
    row_starts_tmp = cuda.device_array(rows, np.int32)
    row_count_actual_dev = cuda.device_array(1, np.int32)
    row_count_actual_dev[0] = 0
    
    if GPUPGPARSER_DEBUG_KERNELS_WRAPPER and KERNEL_DEBUG_ARRAY_SIZE > 0:
        dbg_arr_find = cuda.device_array(KERNEL_DEBUG_ARRAY_SIZE * 5, np.int32)
        dbg_idx_find = cuda.device_array(1, np.int32)
        dbg_idx_find[0] = 0
    else:
        dbg_arr_find = cuda.device_array(1 * 5, np.int32)
        dbg_idx_find = cuda.device_array(1, np.int32)
        dbg_idx_find[0] = 0

    find_row_start_offsets_gpu[blocks, threads](
        raw_dev, header_size, row_starts_tmp, row_count_actual_dev, dbg_arr_find, dbg_idx_find
    )
    cuda.synchronize()

    actual_rows = int(row_count_actual_dev.copy_to_host()[0])
    if actual_rows == 0:
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
    
    row_starts_dev = cuda.device_array(actual_rows, np.int32)
    row_starts_dev[:] = row_starts_tmp[:actual_rows]
    rows = actual_rows

    # Lengths & nulls
    row_lengths_dev = cuda.device_array(rows, np.int32)
    null_flags_dev = cuda.device_array((rows, ncols), np.int8)
    blocks_len = math.ceil(rows / threads_per_block)
    calculate_row_lengths_and_null_flags_gpu[blocks_len, threads_per_block](
        raw_dev, rows, ncols, row_starts_dev, row_lengths_dev, null_flags_dev
    )
    cuda.synchronize()

    # Field parse
    field_offsets_dev = cuda.device_array((rows, ncols), np.int32)
    field_lengths_dev = cuda.device_array((rows, ncols), np.int32)
    parse_fields_from_offsets_gpu[blocks_len, threads_per_block](
        raw_dev, ncols, rows, row_starts_dev, field_offsets_dev, field_lengths_dev
    )
    cuda.synchronize()

    return field_offsets_dev, field_lengths_dev

__all__ = ["parse_binary_chunk_gpu", "detect_pg_header_size"]