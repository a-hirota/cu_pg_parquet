"""
gpu_decoder_v2.py
=================
GPUMemoryManagerV2 を用いて

  1. pass‑1 (len/null 収集)
  2. prefix‑sum で offsets / total_bytes
  3. pass‑2 (scatter‑copy, NUMERIC→UTF8 文字列化)
  4. Arrow RecordBatch を生成

までを 1 関数 `decode_chunk()` で実行するラッパ。

前提
-----
* PostgreSQL COPY BINARY 解析済みの
    - `field_offsets` (rows, ncols) int32 DeviceNDArray
    - `field_lengths` (rows, ncols) int32 DeviceNDArray
  を受け取る。  ( `pg_parser_kernels.parse_binary_format_kernel_one_row` の結果 )

* ColumnMeta 配列は `meta_fetch.fetch_column_meta()` で取得済みとする。

本モジュール自体はテスト用スタブ実装の位置付け。
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import cupy as cp
import pyarrow as pa
try:
    import pyarrow.cuda as pacuda
except ImportError:
    pacuda = None

from numba import cuda

from .type_map import (
    ColumnMeta,
    UTF8,
    BINARY,
)
from .gpu_memory_manager_v2 import GPUMemoryManagerV2

from .cuda_kernels.arrow_gpu_pass1 import pass1_len_null
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.numeric_utils import int64_to_decimal_ascii  # noqa: F401  (import for Numba registration)


# ----------------------------------------------------------------------
def _build_var_indices(columns: List[ColumnMeta]) -> np.ndarray:
    """
    可変長列 → インデックス配列 (-1 = fixed)
    """
    var_idx = -1
    idxs = np.full(len(columns), -1, dtype=np.int32)
    for i, m in enumerate(columns):
        if m.is_variable and m.pg_oid != 1700:
            var_idx += 1
            idxs[i] = var_idx
    return idxs


# ----------------------------------------------------------------------
def decode_chunk(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,  # uint8[:]
    field_offsets_dev,  # int32[:, :]
    field_lengths_dev,  # int32[:, :]
    columns: List[ColumnMeta],
) -> pa.RecordBatch:
    """
    GPU メモリ上の COPY バイナリ解析結果を 2‑pass で Arrow RecordBatch へ変換
    """
    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    # ----------------------------------
    # 1. GPU バッファ確保
    # ----------------------------------
    gmm = GPUMemoryManagerV2()
    bufs: Dict[str, Any] = gmm.initialize_device_buffers(columns, rows)

    d_nulls = {}
    varlen_meta = []  # (col_idx, var_idx, name)

    # 固定長列 : (d_values, d_nulls)
    # 可変長列 : (d_values, d_nulls, max_len)  ※初期は max_len で確保
    for cidx, col in enumerate(columns):
        if col.is_variable and col.pg_oid != 1700:
            var_idx = bufs["param1"][cidx]  # 将来 max_len だがここでは使わない
            d_values, d_null, _ = bufs[col.name]
            d_nulls[col.name] = d_null
            varlen_meta.append((cidx, len(varlen_meta), col.name))
        else:
            triple = bufs[col.name]
            d_values = triple[0]
            d_null = triple[1]
            d_nulls[col.name] = d_null

    # ----------------------------------
    # 2. pass‑1 len/null
    # ----------------------------------
    var_indices_host = _build_var_indices(columns)
    d_var_indices = cuda.to_device(var_indices_host)

    # 可変長列 len 配列 (n_var, rows)
    n_var = len(varlen_meta)
    d_var_lens = cuda.device_array((n_var, rows), dtype=np.int32)
    # 全 NULL 初期化
    d_var_lens[:] = 0  # device-side memset

    # nulls (ncols, rows)
    d_nulls_all = cuda.device_array((ncols, rows), dtype=np.uint8)
    d_nulls_all[:] = 0

    threads = 256
    blocks = (rows + threads - 1) // threads
    pass1_len_null[blocks, threads](
        field_lengths_dev, d_var_indices, d_var_lens, d_nulls_all
    )
    cuda.synchronize()

    # ----------------------------------
    # 3. prefix‑sum offsets
    # ----------------------------------
    d_offsets = cuda.device_array((n_var, rows + 1), dtype=np.int32)
    total_bytes = []
    for v in range(n_var):
        cp_len = cp.asarray(d_var_lens[v])
        cp_off = cp.concatenate([cp.zeros(1, dtype=cp.int32), cp.cumsum(cp_len, dtype=cp.int32)])
        d_offsets[v] = cp_off
        total_bytes.append(int(cp_off[-1].get()))

    # 可変長 values バッファを調整 (max_len ではなく実サイズ)
    values_dev = []
    for (cidx, v_idx, name), tbytes in zip(varlen_meta, total_bytes):
        # NUMERIC(1700) は ASCII 文字列化で最大 24B 程度になるため余裕を持たせる
        col = columns[cidx]
        if col.pg_oid == 1700:
            tbytes = max(tbytes, rows * 24)
        # 再確保して置換
        new_buf = cuda.device_array(tbytes, dtype=np.uint8)
        old_tuple = bufs[name]
        bufs[name] = (new_buf, old_tuple[1], old_tuple[2] if len(old_tuple) > 2 else 0)
        values_dev.append(new_buf)

    # ----------------------------------
    # 4. pass‑2 scatter‑copy per var‑col
    # ----------------------------------
    for (cidx, v_idx, name) in varlen_meta:
        offsets_v = d_offsets[v_idx]
        field_off_v = field_offsets_dev[:, cidx]
        field_len_v = field_lengths_dev[:, cidx]
        numeric_mode = 1 if columns[cidx].pg_oid == 1700 else 0
        pass2_scatter_varlen[blocks, threads](
            raw_dev,
            field_off_v,
            field_len_v,
            offsets_v,
            values_dev[v_idx],
            numeric_mode,
        )
    cuda.synchronize()

    # ----------------------------------
    # 5. Arrow RecordBatch 組立
    # ----------------------------------
    arrays = []
    for cidx, col in enumerate(columns):
        if pacuda:
            # CUDA Buffer から zero-copy
            if col.is_variable:
                v_idx = var_indices_host[cidx]
                values_buf = pacuda.CudaBufferBuf(values_dev[v_idx])
                offsets_buf = pacuda.CudaBufferBuf(d_offsets[v_idx])
                null_buf = pacuda.CudaBufferBuf(d_nulls_all[cidx])
                if col.pg_oid == 1700:  # NUMERIC
                    triple = bufs[col.name]
                    d_values = triple[0]
                    d_null = triple[1]
                    data_buf = pacuda.CudaBufferBuf(d_values)
                    null_buf = pacuda.CudaBufferBuf(d_nulls_all[cidx])
                    if col.arrow_param is None:
                        precision, scale = 38, 0  # デフォルト値
                    else:
                        precision, scale = col.arrow_param
                    pa_type = pa.decimal128(precision, scale)
                    arrays.append(
                        pa.Array.from_buffers(pa_type, rows, [null_buf, data_buf])
                    )
                else:
                    pa_arr = pa.BinaryArray.from_buffers(
                        pa.binary(),
                        rows,
                        [null_buf, offsets_buf, values_buf],
                    )
                    if col.arrow_id == UTF8:
                        pa_arr = pa_arr.cast(pa.string())
                    arrays.append(pa_arr)
            else:
                triple = bufs[col.name]
                d_values = triple[0]
                d_null = triple[1]
                data_buf = pacuda.CudaBufferBuf(d_values)
                null_buf = pacuda.CudaBufferBuf(d_nulls_all[cidx])
                pa_type = pa.int32() if col.elem_size == 4 else pa.int64()
                arrays.append(
                    pa.Array.from_buffers(pa_type, rows, [null_buf, data_buf])
                )
        else:
            # pacuda がなければホストへコピーして通常の Buffer 使用
            if col.is_variable:
                v_idx = var_indices_host[cidx]
                # デバイスから転送
                values_np = values_dev[v_idx].copy_to_host()
                offsets_np = d_offsets[v_idx].copy_to_host()
                null_np = d_nulls_all[cidx].copy_to_host()
                
                # PyArrow バッファへ
                values_buf = pa.py_buffer(values_np)
                offsets_buf = pa.py_buffer(offsets_np)
                null_buf = pa.py_buffer(null_np)
                
                if col.pg_oid == 1700:  # NUMERIC
                    triple = bufs[col.name]
                    d_values = triple[0]
                    d_null = triple[1]
                    values_np = d_values.copy_to_host()
                    null_np = d_nulls_all[cidx].copy_to_host()
                    
                    values_buf = pa.py_buffer(values_np)
                    null_buf = pa.py_buffer(null_np)
                    
                    if col.arrow_param is None:
                        precision, scale = 38, 0  # デフォルト値
                    else:
                        precision, scale = col.arrow_param
                    pa_type = pa.decimal128(precision, scale)
                    arrays.append(
                        pa.Array.from_buffers(pa_type, rows, [null_buf, values_buf])
                    )
                else:
                    pa_arr = pa.BinaryArray.from_buffers(
                        pa.binary(),
                        rows,
                        [null_buf, offsets_buf, values_buf],
                    )
                    if col.arrow_id == UTF8:
                        pa_arr = pa_arr.cast(pa.string())
                    arrays.append(pa_arr)
            else:
                triple = bufs[col.name]
                d_values = triple[0]
                d_null = triple[1]
                values_np = d_values.copy_to_host()
                null_np = d_nulls_all[cidx].copy_to_host()
                
                values_buf = pa.py_buffer(values_np)
                null_buf = pa.py_buffer(null_np)
                
                pa_type = pa.int32() if col.elem_size == 4 else pa.int64()
                arrays.append(
                    pa.Array.from_buffers(pa_type, rows, [null_buf, values_buf])
                )

    batch = pa.RecordBatch.from_arrays(arrays, [c.name for c in columns])
    return batch


__all__ = ["decode_chunk"]
