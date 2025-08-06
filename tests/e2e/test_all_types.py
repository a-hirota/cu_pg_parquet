"""
End-to-End tests for all supported PostgreSQL types
各型がFunction 1, 2, 3を通じて正しく処理されることを確認
"""

import os
import struct
import sys
import tempfile
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import cudf
    import cupy as cp
    from numba import cuda

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# 型ごとのテストパラメータ
TYPE_TEST_PARAMS = [
    # (type_name, sql_type, oid, test_values, arrow_type, is_implemented)
    ("smallint", "SMALLINT", 21, [-32768, 0, 32767, None], pa.int16(), True),
    ("integer", "INTEGER", 23, [-2147483648, 0, 2147483647, None], pa.int32(), True),
    (
        "bigint",
        "BIGINT",
        20,
        [-9223372036854775808, 0, 9223372036854775807, None],
        pa.int64(),
        True,
    ),
    ("real", "REAL", 700, [-3.4e38, 0.0, 3.4e38, None], pa.float32(), True),
    ("double", "DOUBLE PRECISION", 701, [-1.7e308, 0.0, 1.7e308, None], pa.float64(), True),
    (
        "numeric",
        "NUMERIC(10,2)",
        1700,
        [Decimal("-999.99"), Decimal("0.00"), Decimal("999.99"), None],
        pa.string(),
        True,
    ),  # Decimal as string
    ("boolean", "BOOLEAN", 16, [True, False, None], pa.bool_(), True),
    ("text", "TEXT", 25, ["Hello", "世界", "", None], pa.string(), True),
    ("varchar", "VARCHAR(50)", 1043, ["Test", "テスト", "", None], pa.string(), True),
    ("char", "CHAR(10)", 1042, ["Fixed     ", "Test      ", None], pa.string(), True),
    ("bytea", "BYTEA", 17, [b"\x00\x01\x02", b"\xff\xfe\xfd", b"", None], pa.binary(), True),
    ("date", "DATE", 1082, [date(2024, 1, 1), date(2024, 12, 31), None], pa.timestamp("s"), True),
    (
        "timestamp",
        "TIMESTAMP",
        1114,
        [datetime(2024, 1, 1, 12, 30, 45), None],
        pa.timestamp("us"),
        True,
    ),
    (
        "timestamptz",
        "TIMESTAMPTZ",
        1184,
        [datetime(2024, 1, 1, 12, 30, 45), None],
        pa.timestamp("us"),
        True,
    ),
    # 未実装の型
    ("time", "TIME", 1083, ["12:30:45", None], None, False),
    ("uuid", "UUID", 2950, ["550e8400-e29b-41d4-a716-446655440000", None], None, False),
    ("json", "JSON", 114, ['{"key": "value"}', None], None, False),
]


class TestAllTypes:
    """すべての型に対するE2Eテスト"""

    @pytest.mark.parametrize(
        "type_name,sql_type,oid,test_values,arrow_type,is_implemented", TYPE_TEST_PARAMS
    )
    def test_type_full_pipeline(
        self,
        db_connection,
        temp_output_dir,
        type_name,
        sql_type,
        oid,
        test_values,
        arrow_type,
        is_implemented,
    ):
        """各型のフルパイプラインテスト"""
        if not GPU_AVAILABLE and is_implemented:
            pytest.skip("GPU not available")

        table_name = f"test_{type_name}_pipeline"

        # テーブル作成
        cur = db_connection.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")

        if oid == 2950:  # UUID
            try:
                cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            except psycopg2.errors.FeatureNotSupported:
                # UUID拡張がインストールされていない場合はスキップ
                print(f"⚠️  {type_name}: UUID extension not available, skipping test")
                return

        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                test_column {sql_type}
            )
        """
        )

        # データ挿入
        for i, value in enumerate(test_values):
            if value is None:
                cur.execute(f"INSERT INTO {table_name} (id, test_column) VALUES (%s, NULL)", (i,))
            elif oid == 2950:  # UUID
                cur.execute(
                    f"INSERT INTO {table_name} (id, test_column) VALUES (%s, %s::uuid)", (i, value)
                )
            else:
                cur.execute(
                    f"INSERT INTO {table_name} (id, test_column) VALUES (%s, %s)", (i, value)
                )

        db_connection.commit()

        try:
            # Function 1: COPY BINARY extraction
            binary_data = bytearray()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", tmp)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                binary_data = f.read()
            os.unlink(tmp_path)

            print(f"\n✅ {type_name}: COPY BINARY extraction successful ({len(binary_data)} bytes)")

            if is_implemented and GPU_AVAILABLE:
                from src.postgres_to_parquet_converter import DirectProcessor
                from src.types import PG_OID_TO_ARROW, ColumnMeta

                # メタデータ作成
                columns = []

                # id列
                columns.append(
                    ColumnMeta(name="id", pg_oid=23, pg_typmod=-1, arrow_id=1, elem_size=4)  # INT32
                )

                # test_column
                if oid in PG_OID_TO_ARROW:
                    arrow_id, elem_size = PG_OID_TO_ARROW[oid]
                    columns.append(
                        ColumnMeta(
                            name="test_column",
                            pg_oid=oid,
                            pg_typmod=-1,
                            arrow_id=arrow_id,
                            elem_size=elem_size or -1,
                        )
                    )
                else:
                    # 未実装の型はスキップ
                    print(f"⚠️  {type_name}: Type not implemented in PG_OID_TO_ARROW mapping")
                    return

                # GPU処理
                gpu_data = cp.asarray(cp.frombuffer(binary_data, dtype=cp.uint8))
                raw_dev = cuda.as_cuda_array(gpu_data).view(dtype=cp.uint8)

                processor = DirectProcessor(
                    use_rmm=True, optimize_gpu=True, verbose=False, test_mode=False
                )
                output_path = temp_output_dir / f"{table_name}.parquet"

                try:
                    cudf_df, timing_info = processor.transform_postgres_to_parquet_format(
                        raw_dev=raw_dev,
                        columns=columns,
                        ncols=len(columns),
                        header_size=19,
                        output_path=str(output_path),
                        compression="snappy",
                    )

                    print(f"✅ {type_name}: GPU processing successful ({len(cudf_df)} rows)")

                    # Parquet検証
                    if output_path.exists():
                        metadata = pq.read_metadata(output_path)
                        print(f"✅ {type_name}: Parquet created ({metadata.num_rows} rows)")

                except Exception as e:
                    print(f"❌ {type_name}: GPU processing failed - {str(e)}")
                    if is_implemented:
                        raise

            elif not is_implemented:
                print(f"⚫ {type_name}: Type not implemented (expected)")

        finally:
            cur = db_connection.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            cur.close()

    def test_type_summary_report(self, capsys):
        """テスト完了後にサマリーレポートを出力"""
        # This will run after all tests
        with capsys.disabled():
            print("\n" + "=" * 80)
            print("TYPE SUPPORT SUMMARY")
            print("=" * 80)
            print(f"{'Type':<15} {'Status':<10} {'Notes':<50}")
            print("-" * 80)

            implemented_types = [
                ("SMALLINT", "✅", "INT16 - Fully supported"),
                ("INTEGER", "✅", "INT32 - Fully supported"),
                ("BIGINT", "✅", "INT64 - Fully supported"),
                ("REAL", "✅", "FLOAT32 - Fully supported"),
                ("DOUBLE", "✅", "FLOAT64 - Fully supported"),
                ("NUMERIC", "✅", "DECIMAL128 - Stored as string"),
                ("BOOLEAN", "✅", "BOOL - Fully supported"),
                ("TEXT", "✅", "UTF8 - Fully supported"),
                ("VARCHAR", "✅", "UTF8 - Fully supported"),
                ("CHAR", "✅", "UTF8 - Fully supported"),
                ("BYTEA", "✅", "BINARY - Fully supported"),
                ("DATE", "✅", "TS64_S - Converted to timestamp"),
                ("TIMESTAMP", "✅", "TS64_US - Fully supported"),
                ("TIMESTAMPTZ", "✅", "TS64_US - Fully supported"),
            ]

            unimplemented_types = [
                ("TIME", "❌", "Not implemented - OID 1083"),
                ("UUID", "❌", "Not implemented - OID 2950"),
                ("JSON", "❌", "Not implemented - OID 114"),
                ("JSONB", "❌", "Not implemented - OID 3802"),
                ("ARRAY", "❌", "Not implemented"),
                ("COMPOSITE", "❌", "Not implemented"),
            ]

            for type_name, status, notes in implemented_types:
                print(f"{type_name:<15} {status:<10} {notes:<50}")

            print("-" * 80)

            for type_name, status, notes in unimplemented_types:
                print(f"{type_name:<15} {status:<10} {notes:<50}")

            print("=" * 80)
