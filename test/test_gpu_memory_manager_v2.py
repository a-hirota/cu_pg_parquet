"""
簡易テスト: ColumnMeta 取得 → GPUMemoryManagerV2 で GPU バッファ確保

必要:
    - CUDA / Numba 環境
    - psycopg2
実行例:
    python test_gpu_memory_manager_v2.py --dsn "dbname=postgres user=postgres host=localhost" --rows 1000
"""

from __future__ import annotations

import argparse
import sys

import psycopg2

from meta_fetch import fetch_column_meta
from gpu_memory_manager_v2 import GPUMemoryManagerV2


TABLES = ["lineorder", "customer", "date1"]


def run_test(conn, rows: int):
    mgr = GPUMemoryManagerV2()

    for tbl in TABLES:
        print(f"== Table: {tbl} ==")
        metas = fetch_column_meta(conn, f"SELECT * FROM {tbl}")
        print(f"  Columns: {[m.name for m in metas]}")
        bufs = mgr.initialize_device_buffers(metas, rows)
        for name, val in bufs.items():
            if name.startswith("_"):
                continue
            if isinstance(val, tuple):
                print(f"    {name}: varlen buffers shapes {[v.shape for v in val[:-1]]}, max_len={val[-1]}")
            elif hasattr(val, 'shape'):
                print(f"    {name}: shape {val.shape}, dtype {val.dtype}")
        print("  -> OK\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=postgres user=postgres host=localhost")
    ap.add_argument("--rows", type=int, default=1000)
    args = ap.parse_args()

    try:
        conn = psycopg2.connect(args.dsn)
    except Exception as e:
        print(f"psycopg2 connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    run_test(conn, args.rows)
    conn.close()


if __name__ == "__main__":
    main()
