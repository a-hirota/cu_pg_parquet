"""
pytest unit test for gpu_decoder_v2 + gpu_parse_wrapper

* loads COPY BINARY samples under expected_meta/ if present
  (lineorder.bin, customer.bin, date1.bin) – not committed yet ->
  skips if file missing
* generates ColumnMeta from expected_meta/*.json
* runs full GPU pipeline and asserts RecordBatch row count matches
  metadata json length
"""

import json
import os
import pytest
import numpy as np
from numba import cuda

import pyarrow as pa
import pyarrow.json as paj

from gpu_parse_wrapper import parse_binary_chunk_gpu
from gpu_decoder_v2 import decode_chunk
from meta_fetch import ColumnMeta  # type: ignore


DATA_CASES = [
    ("lineorder", "test/expected_meta/lineorder.json", "input/lineorder.bin"),
    ("customer", "test/expected_meta/customer.json", "input/customer.bin"),
    ("date1", "test/expected_meta/date1.json", "input/date1.bin"),
]


def load_column_meta(meta_path):
    with open(meta_path) as f:
        meta_j = json.load(f)
    cols = []
    for col in meta_j["columns"]:
        cols.append(
            ColumnMeta(
                name=col["name"],
                pg_oid=col["pg_oid"],
                pg_typmod=col.get("pg_typmod", 0),
                arrow_id=col["arrow_id"],
                elem_size=col["elem_size"],
            )
        )
    return cols, meta_j["rows"]


@pytest.mark.parametrize("name,meta_path,bin_path", DATA_CASES)
def test_gpu_decode_case(name, meta_path, bin_path):
    if not os.path.exists(meta_path) or not os.path.exists(bin_path):
        pytest.skip(f"sample {name} missing")

    cols, exp_rows = load_column_meta(meta_path)

    with open(bin_path, "rb") as f:
        raw_host = np.frombuffer(f.read(), dtype=np.uint8)

    # Device copy
    raw_dev = cuda.to_device(raw_host)

    rows = exp_rows
    ncols = len(cols)

    # GPU parse
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev, rows, ncols
    )

    batch = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, cols)

    assert batch.num_rows == rows
    assert len(batch.columns) == ncols
