"""
psql_copy_stream.py
===================
psycopg3 + CUDA pinned memory で COPY BINARY をチャンク受信し
GPU へ転送する最速ストリーム実装例。

・COPY (SELECT ...) TO STDOUT (FORMAT BINARY) を実行
・pinned_array へ read_into で直接書き込み
・cuda.device_array へ転送し、既存 GPU パイプラインへ流す
"""

import psycopg
import numpy as np
from numba import cuda

def copy_binary_to_gpu_chunks(
    dsn: str,
    sql: str,
    chunk_bytes: int = 32 << 20,  # 32MB
    process_chunk=None,           # Callable[[cuda.devicearray], None]
):
    """
    COPY BINARY を pinned buffer でチャンク受信し GPU へ転送

    Parameters
    ----------
    dsn : str
        PostgreSQL DSN
    sql : str
        SELECT ... のクエリ
    chunk_bytes : int
        1チャンクのバイト数
    process_chunk : callable
        (gpu_dev_array, nbytes) を受けて処理するコールバック
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor().copy(f"COPY ({sql}) TO STDOUT (FORMAT binary)") as cpy:
            pinned = cuda.pinned_array(chunk_bytes, dtype=np.uint8)
            gpu_raw = cuda.device_array(chunk_bytes, dtype=np.uint8)
            mv_host = memoryview(pinned)
            while True:
                n = cpy.read_into(mv_host)
                if n == 0:
                    break
                # GPU 転送
                gpu_raw[:n].set(pinned[:n])
                if process_chunk:
                    process_chunk(gpu_raw, n)
