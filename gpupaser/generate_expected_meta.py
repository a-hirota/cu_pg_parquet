"""
指定された PostgreSQL テーブルの ColumnMeta を取得し、
JSON 形式で snapshot ファイルに保存するユーティリティ。

使い方:
    python generate_expected_meta.py --tables lineorder customer date1 \\
        --dsn "dbname=postgres user=postgres host=localhost"

保存先:
    expected_meta/{table}.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import psycopg2
from psycopg2.extensions import connection as _Conn

from .meta_fetch import fetch_column_meta
from .type_map import ColumnMeta


def meta_to_dict(meta: ColumnMeta) -> Dict[str, Any]:
    """ColumnMeta → JSON 変換しやすい dict"""
    return {
        "name": meta.name,
        "pg_oid": meta.pg_oid,
        "pg_typmod": meta.pg_typmod,
        "arrow_id": meta.arrow_id,
        "elem_size": meta.elem_size,
        "arrow_param": meta.arrow_param,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsn",
        default="dbname=postgres user=postgres host=localhost",
        help='psycopg2.connect に渡す DSN 文字列'
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        required=True,
        help="メタ情報を取得するテーブル名 (複数可)"
    )
    args = parser.parse_args()

    Path("expected_meta").mkdir(exist_ok=True)

    conn: _Conn = psycopg2.connect(args.dsn)

    for tbl in args.tables:
        print(f"Fetching meta for table: {tbl}")
        sql = f"SELECT * FROM {tbl}"
        metas: List[ColumnMeta] = fetch_column_meta(conn, sql)

        data = [meta_to_dict(m) for m in metas]
        out_path = Path("expected_meta") / f"{tbl}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  -> saved {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
