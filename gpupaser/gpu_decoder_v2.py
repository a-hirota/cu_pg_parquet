"""
gpu_decoder_v2.py
=================
GPUMemoryManagerV2 を用いて

  1. pass‑1 (len/null 収集) - CPUで実行
  2. prefix‑sum で offsets / total_bytes - GPU(CuPy)で実行
  3. pass‑2 (scatter‑copy, NUMERIC→UTF8 文字列化) - GPUカーネルで実行
  4. Arrow RecordBatch を生成 - CPUで実行

までを 1 関数 `decode_chunk()` で実行するラッパ。

前提
-----
* PostgreSQL COPY BINARY 解析済みの
    - `field_offsets` (rows, ncols) int32 DeviceNDArray
    - `field_lengths` (rows, ncols) int32 DeviceNDArray
  を受け取る。 (現在は `gpu_parse_wrapper.py` の CPU 実装の結果)

* ColumnMeta 配列は `meta_fetch.fetch_column_meta()` で取得済みとする。
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import cupy as cp
import pyarrow as pa

try:
    import pyarrow.cuda as pacuda  # noqa: F401
except ImportError:  # pragma: no cover
    pacuda = None  # type: ignore

from numba import cuda

from .type_map import ColumnMeta, UTF8, BINARY  # noqa: F401 (BINARY is reserved for future use)
from .gpu_memory_manager_v2 import GPUMemoryManagerV2
from .cuda_kernels.arrow_gpu_pass2 import pass2_scatter_varlen
from .cuda_kernels.arrow_gpu_pass2_fixed import pass2_scatter_fixed
from .cuda_kernels.numeric_utils import int64_to_decimal_ascii  # noqa: F401  (Numba registration)

# ----------------------------------------------------------------------

def _build_var_indices(columns: List[ColumnMeta]) -> np.ndarray:
    """可変長列 → インデックス配列 (-1 = fixed)"""
    var_idx = -1
    idxs = np.full(len(columns), -1, dtype=np.int32)
    for i, meta in enumerate(columns):
        if meta.is_variable:
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
    """GPU メモリ上の COPY BINARY 解析結果を 2‑pass で Arrow RecordBatch へ変換"""

    rows, ncols = field_lengths_dev.shape
    if rows == 0:
        raise ValueError("rows == 0")

    # ----------------------------------
    # 1. GPU バッファ確保 (Arrow出力用)
    # ----------------------------------
    gmm = GPUMemoryManagerV2()
    bufs: Dict[str, Any] = gmm.initialize_device_buffers(columns, rows)

    # varlen_meta の準備 (Pass 2 で使用)
    varlen_meta = [(cidx, len([v for v in columns[:cidx] if v.is_variable]), col.name)
                   for cidx, col in enumerate(columns) if col.is_variable]

    # ----------------------------------
    # 2. pass‑1 len/null (CPUで実行)
    # ----------------------------------
    field_lengths_host = field_lengths_dev.copy_to_host()
    var_indices_host = _build_var_indices(columns)
    n_var = len(varlen_meta)

    # host バッファ
    host_nulls_all = np.ones((rows, ncols), dtype=np.uint8)
    host_var_lens = np.zeros((n_var, rows), dtype=np.int32)

    for r in range(rows):
        for c in range(ncols):
            flen = field_lengths_host[r, c]
            if flen == -1:
                host_nulls_all[r, c] = 0  # NULL
            else:
                v_idx = var_indices_host[c]
                if v_idx != -1:
                    host_var_lens[v_idx, r] = flen

    d_var_lens = cuda.to_device(host_var_lens)

    # ----------------------------------
    # 3. prefix‑sum offsets (GPU - CuPy)
    # ----------------------------------
    d_offsets = cuda.device_array((n_var, rows + 1), dtype=np.int32)
    total_bytes: List[int] = []

    for v in range(n_var):
        cp_len = cp.asarray(d_var_lens[v])
        cp_off = cp.concatenate([cp.zeros(1, dtype=np.int32), cp.cumsum(cp_len, dtype=np.int32)])
        d_offsets[v] = cp_off
        total_bytes.append(int(cp_off[-1].get()))

    # 可変長 values バッファを調整 (max_len ではなく実サイズ)
    values_dev = []
    for (cidx, v_idx, name), tbytes in zip(varlen_meta, total_bytes):
        col = columns[cidx]
        if col.pg_oid == 1700:  # NUMERIC → 十分なバッファを確保
            tbytes = max(tbytes, rows * 24)
        new_buf = cuda.device_array(max(1, tbytes), dtype=np.uint8)
        old_tuple = bufs[name]
        bufs[name] = (new_buf, old_tuple[1], old_tuple[2] if len(old_tuple) > 2 else 0)
        values_dev.append(new_buf)

    # ----------------------------------
    # 4. pass‑2 scatter‑copy (GPU Kernel)
    # ----------------------------------
    threads = 256
    blocks = (rows + threads - 1) // threads

    for cidx, v_idx, _ in varlen_meta:
        pass2_scatter_varlen[blocks, threads](
            raw_dev,
            field_offsets_dev[:, cidx],
            field_lengths_dev[:, cidx],
            d_offsets[v_idx],
            values_dev[v_idx],
            1 if columns[cidx].pg_oid == 1700 else 0,
        )
    cuda.synchronize()

    for cidx, col in enumerate(columns):
        if not col.is_variable:
            d_vals, _d_nulls, stride = bufs[col.name]
            pass2_scatter_fixed[blocks, threads](
                raw_dev,
                field_offsets_dev[:, cidx],
                col.elem_size,
                d_vals,
                stride,
            )
    cuda.synchronize()

    # ----------------------------------
    # 5. Arrow RecordBatch 組立 (CPU)
    # ----------------------------------
    arrays: List[pa.Array] = []
    for cidx, col in enumerate(columns):
        mask = host_nulls_all[:, cidx] == 0  # True == NULL

        if col.is_variable:
            v_idx = var_indices_host[cidx]
            values_np = values_dev[v_idx].copy_to_host()
            offsets_np = d_offsets[v_idx].copy_to_host()
            list_data = [None if mask[r] else bytes(values_np[offsets_np[r]: offsets_np[r + 1]])
                         for r in range(rows)]
            pa_type = pa.string() if col.arrow_id == UTF8 else pa.binary()
            arrays.append(pa.array(list_data, type=pa_type))
        else:
            d_vals, _d_nulls, stride = bufs[col.name]
            values_np = d_vals.copy_to_host()
            np_dtype_map = {21: np.int16, 23: np.int32, 20: np.int64, 700: np.float32,
                            701: np.float64, 16: np.bool_, 1082: 'datetime64[D]',
                            1114: 'datetime64[us]', 1184: 'datetime64[us]'}
            if col.pg_oid not in np_dtype_map:
                arrays.append(pa.nulls(rows))
                continue

            np_dtype = np_dtype_map[col.pg_oid]
            if stride != np.dtype(np_dtype).itemsize:
                # gather
                gathered = np.empty(rows, dtype=np_dtype)
                for r in range(rows):
                    if not mask[r]:
                        start = r * stride
                        gathered[r] = np.frombuffer(values_np[start: start + stride], dtype=np_dtype)[0]
                values_np = gathered
            else:
                values_np = values_np.view(np_dtype)

            pa_type_map = {21: pa.int16(), 23: pa.int32(), 20: pa.int64(), 700: pa.float32(),
                            701: pa.float64(), 16: pa.bool_(), 1082: pa.date32(),
                            1114: pa.timestamp('us'), 1184: pa.timestamp('us')}
            arrays.append(pa.array(values_np, type=pa_type_map[col.pg_oid], mask=mask))

    batch = pa.RecordBatch.from_arrays(arrays, names=[c.name for c in columns])
    return batch


__all__ = ["decode_chunk"]
