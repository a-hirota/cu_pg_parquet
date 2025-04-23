"""
End‑to‑end test: PostgreSQL → COPY BINARY → GPU 2‑pass → Arrow RecordBatch

環境変数
--------
GPUPASER_PG_DSN  : psycopg2.connect 互換 DSN 文字列
PG_TABLE_PREFIX  : lineorder テーブルがスキーマ名付きの場合に指定 (optional)

テスト内容
----------
lineorder テーブルの先頭 100 行を
1. CPU 側で普通に SELECT → pyarrow.Table (expected)
2. COPY BINARY で取得 → GPU パイプライン decode_chunk
3. RecordBatch → Table にし expected とデータ一致を assert
"""

import os
import pytest
import numpy as np
import psycopg
import pyarrow as pa
# CUDA backendは必須ではない

from numba import cuda

from .meta_fetch import fetch_column_meta
from .gpu_parse_wrapper import parse_binary_chunk_gpu
from .gpu_decoder_v2 import decode_chunk


ROWS = 100
TABLE_NAME = "lineorder"


@pytest.mark.skipif(
    "GPUPASER_PG_DSN" not in os.environ,
    reason="GPUPASER_PG_DSN not set"
)
def test_e2e_postgres_arrow():
    dsn = os.environ["GPUPASER_PG_DSN"]
    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    conn = psycopg.connect(dsn)
    try:
        # -------------------------------
        # Expected: SELECT ... LIMIT ROWS
        # -------------------------------
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {tbl} LIMIT {ROWS}")
        colnames = [d.name for d in cur.description]
        rows_py = cur.fetchall()
        expected_tb = pa.Table.from_pylist(
            [dict(zip(colnames, r)) for r in rows_py]
        )
        cur.close()

        # -------------------------------
        # ColumnMeta
        # -------------------------------
        columns = fetch_column_meta(conn, f"SELECT * FROM {tbl}")

        # -------------------------------
        # COPY BINARY chunk
        # -------------------------------
        copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {ROWS}) TO STDOUT (FORMAT binary)"
        with conn.cursor().copy(copy_sql) as cpy:
            buf = bytearray()
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
    finally:
        conn.close()
    raw_dev = cuda.to_device(raw_host)

    rows = ROWS
    ncols = len(columns)

    # GPU parse offsets/lengths
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev, rows, ncols
    )

    # Decode
    batch = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    result_tb = pa.Table.from_batches([batch])

    # Compare
    assert result_tb.equals(expected_tb, check_metadata=False)
    conn.close()
