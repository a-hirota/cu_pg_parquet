"""
Type Test Matrix - すべての型×機能のテスト状態を管理
"""

import json
import os
import struct
import sys
import tempfile
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import psycopg2
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cudf
    import cupy as cp
    from numba import cuda

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# テスト対象の型定義
POSTGRES_TYPES = {
    # OID: (type_name, sql_type, test_values, expected_arrow_id)
    21: ("SMALLINT", "SMALLINT", [-32768, 0, 32767, None], 0),  # INT16
    23: ("INTEGER", "INTEGER", [-2147483648, 0, 2147483647, None], 1),  # INT32
    20: ("BIGINT", "BIGINT", [-9223372036854775808, 0, 9223372036854775807, None], 2),  # INT64
    700: ("REAL", "REAL", [-3.4e38, 0.0, 3.4e38, None], 3),  # FLOAT32
    701: ("DOUBLE", "DOUBLE PRECISION", [-1.7e308, 0.0, 1.7e308, None], 4),  # FLOAT64
    1700: (
        "NUMERIC",
        "NUMERIC(10,2)",
        [Decimal("-999.99"), Decimal("0.00"), Decimal("999.99"), None],
        5,
    ),  # DECIMAL128
    16: ("BOOLEAN", "BOOLEAN", [True, False, None], 10),  # BOOL
    25: ("TEXT", "TEXT", ["Hello", "世界", "", None], 6),  # UTF8
    1043: ("VARCHAR", "VARCHAR(50)", ["Test", "テスト", "", None], 6),  # UTF8
    1042: ("CHAR", "CHAR(10)", ["Fixed     ", "Test      ", None], 6),  # UTF8
    17: ("BYTEA", "BYTEA", [b"\x00\x01\x02", b"\xff\xfe\xfd", b"", None], 7),  # BINARY
    1082: ("DATE", "DATE", [date(2024, 1, 1), date(2024, 12, 31), None], 8),  # TS64_S
    1114: ("TIMESTAMP", "TIMESTAMP", [datetime(2024, 1, 1, 12, 30, 45), None], 9),  # TS64_US
    1184: ("TIMESTAMPTZ", "TIMESTAMPTZ", [datetime(2024, 1, 1, 12, 30, 45), None], 9),  # TS64_US
    # 未実装の型
    1083: ("TIME", "TIME", [time(12, 30, 45), None], -1),  # 未実装
    2950: ("UUID", "UUID", ["550e8400-e29b-41d4-a716-446655440000", None], -1),  # 未実装
    114: ("JSON", "JSON", ['{"key": "value"}', None], -1),  # 未実装
    3802: ("JSONB", "JSONB", ['{"key": "value"}', None], -1),  # 未実装
}


class TypeTestResult:
    """型ごとのテスト結果を記録"""

    def __init__(self, type_name, oid):
        self.type_name = type_name
        self.oid = oid
        self.results = {
            "function1_extract": None,  # PostgreSQL COPY BINARY extraction
            "function2_gpu_parse": None,  # GPU parsing
            "function3_arrow_cudf": None,  # Arrow to cuDF conversion
            "full_pipeline": None,  # End-to-end pipeline
        }

    def set_result(self, function, passed, error=None):
        self.results[function] = {"passed": passed, "error": error}

    def get_status_symbol(self, function):
        if self.results[function] is None:
            return "⚫"  # Not tested
        elif self.results[function]["passed"]:
            return "✅"  # Passed
        else:
            return "❌"  # Failed


# Global test results
TEST_RESULTS = {}


def create_test_table_for_type(conn, table_name, oid, type_info):
    """特定の型用のテストテーブルを作成"""
    type_name, sql_type, test_values, _ = type_info

    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    # UUID型の場合は拡張を有効化
    if oid == 2950:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    cur.execute(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            test_column {sql_type}
        )
    """
    )

    # テストデータを挿入
    for i, value in enumerate(test_values):
        if value is None:
            cur.execute(f"INSERT INTO {table_name} (id, test_column) VALUES (%s, NULL)", (i,))
        elif oid == 2950:  # UUID
            cur.execute(
                f"INSERT INTO {table_name} (id, test_column) VALUES (%s, %s::uuid)", (i, value)
            )
        else:
            cur.execute(f"INSERT INTO {table_name} (id, test_column) VALUES (%s, %s)", (i, value))

    conn.commit()
    cur.close()
    return len(test_values)


@pytest.mark.parametrize("oid,type_info", POSTGRES_TYPES.items())
class TestTypeMatrix:
    """すべての型に対して各機能をテスト"""

    def test_function1_copy_binary_extraction(self, db_connection, oid, type_info):
        """Function 1: PostgreSQL COPY BINARY extraction"""
        type_name, sql_type, test_values, expected_arrow_id = type_info

        if oid not in TEST_RESULTS:
            TEST_RESULTS[oid] = TypeTestResult(type_name, oid)

        table_name = f"test_{type_name.lower()}_extraction"

        try:
            num_rows = create_test_table_for_type(db_connection, table_name, oid, type_info)

            # COPY BINARY extraction
            cur = db_connection.cursor()
            binary_data = bytearray()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", tmp)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                binary_data = f.read()
            os.unlink(tmp_path)

            # 基本的な検証
            assert len(binary_data) > 19  # ヘッダーより大きい
            assert binary_data[:11] == b"PGCOPY\n\xff\r\n\0"

            TEST_RESULTS[oid].set_result("function1_extract", True)

        except Exception as e:
            TEST_RESULTS[oid].set_result("function1_extract", False, str(e))
            if expected_arrow_id >= 0:  # 実装済みの型なら失敗
                raise
        finally:
            cur = db_connection.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            cur.close()

    def test_function2_gpu_parsing(self, db_connection, oid, type_info):
        """Function 2: GPU parsing"""
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available")

        type_name, sql_type, test_values, expected_arrow_id = type_info

        if oid not in TEST_RESULTS:
            TEST_RESULTS[oid] = TypeTestResult(type_name, oid)

        table_name = f"test_{type_name.lower()}_gpu"

        try:
            from src.cuda_kernels.postgres_binary_parser import (
                parse_postgres_raw_binary_to_column_arrows,
            )
            from src.types import ColumnMeta

            num_rows = create_test_table_for_type(db_connection, table_name, oid, type_info)

            # メタデータ作成
            columns = [
                ColumnMeta(name="id", pg_oid=23, pg_typmod=-1, arrow_id=1, elem_size=4),
                ColumnMeta(
                    name="test_column",
                    pg_oid=oid,
                    pg_typmod=-1,
                    arrow_id=expected_arrow_id if expected_arrow_id >= 0 else 6,  # 未実装はUTF8
                    elem_size=-1 if oid in [1700, 25, 1042, 1043, 17] else 4,  # 可変長型
                ),
            ]

            # バイナリデータ取得
            cur = db_connection.cursor()
            binary_data = bytearray()
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", tmp)
                tmp_path = tmp.name

            with open(tmp_path, "rb") as f:
                binary_data = f.read()
            os.unlink(tmp_path)

            # GPU転送とパース
            gpu_data = cp.asarray(cp.frombuffer(binary_data, dtype=cp.uint8))
            raw_dev = cuda.as_cuda_array(gpu_data).view(dtype=cp.uint8)

            result = parse_postgres_raw_binary_to_column_arrows(
                raw_dev, columns, header_size=19, debug=False, test_mode=True
            )

            # 結果確認（tuple of arraysが返される）
            assert result is not None
            assert len(result) >= 3  # row_positions, field_offsets, field_lengths

            TEST_RESULTS[oid].set_result("function2_gpu_parse", True)

        except Exception as e:
            TEST_RESULTS[oid].set_result("function2_gpu_parse", False, str(e))
            if expected_arrow_id >= 0:  # 実装済みの型なら失敗
                raise
        finally:
            cur = db_connection.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            cur.close()

    def test_function3_arrow_to_cudf(self, oid, type_info):
        """Function 3: Arrow to cuDF conversion"""
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available")

        type_name, sql_type, test_values, expected_arrow_id = type_info

        if oid not in TEST_RESULTS:
            TEST_RESULTS[oid] = TypeTestResult(type_name, oid)

        try:
            import pyarrow as pa

            # 型ごとにArrowアレイを作成
            if expected_arrow_id == 0:  # INT16
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.int16())
            elif expected_arrow_id == 1:  # INT32
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.int32())
            elif expected_arrow_id == 2:  # INT64
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.int64())
            elif expected_arrow_id == 3:  # FLOAT32
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.float32())
            elif expected_arrow_id == 4:  # FLOAT64
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.float64())
            elif expected_arrow_id == 5:  # DECIMAL128
                arrow_array = pa.array(
                    [str(v) for v in test_values if v is not None], type=pa.string()
                )
            elif expected_arrow_id == 6:  # UTF8
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.string())
            elif expected_arrow_id == 7:  # BINARY
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.binary())
            elif expected_arrow_id == 8:  # TS64_S (DATE)
                arrow_array = pa.array(
                    [v for v in test_values if v is not None], type=pa.timestamp("s")
                )
            elif expected_arrow_id == 9:  # TS64_US
                arrow_array = pa.array(
                    [v for v in test_values if v is not None], type=pa.timestamp("us")
                )
            elif expected_arrow_id == 10:  # BOOL
                arrow_array = pa.array([v for v in test_values if v is not None], type=pa.bool_())
            else:
                # 未実装の型
                TEST_RESULTS[oid].set_result("function3_arrow_cudf", False, "Unsupported type")
                return

            # Arrow table作成
            table = pa.table({"test_column": arrow_array})

            # cuDF変換
            gdf = cudf.DataFrame.from_arrow(table)
            assert len(gdf) == len([v for v in test_values if v is not None])

            TEST_RESULTS[oid].set_result("function3_arrow_cudf", True)

        except Exception as e:
            TEST_RESULTS[oid].set_result("function3_arrow_cudf", False, str(e))
            if expected_arrow_id >= 0:  # 実装済みの型なら失敗
                raise


def pytest_sessionfinish(session, exitstatus):
    """テスト終了時にレポートを出力"""
    print("\n\n" + "=" * 80)
    print("TYPE TEST MATRIX REPORT")
    print("=" * 80)
    print(
        f"{'Type':<15} {'OID':<6} {'Extract':<10} {'GPU Parse':<10} {'Arrow→cuDF':<12} {'Pipeline':<10}"
    )
    print("-" * 80)

    for oid in sorted(TEST_RESULTS.keys()):
        result = TEST_RESULTS[oid]
        print(
            f"{result.type_name:<15} {oid:<6} "
            f"{result.get_status_symbol('function1_extract'):<10} "
            f"{result.get_status_symbol('function2_gpu_parse'):<10} "
            f"{result.get_status_symbol('function3_arrow_cudf'):<12} "
            f"{result.get_status_symbol('full_pipeline'):<10}"
        )

    print("\n" + "=" * 80)
    print("Legend: ✅ = Passed, ❌ = Failed, ⚫ = Not tested")
    print("=" * 80)

    # 詳細なエラー情報
    print("\nFAILED TESTS DETAILS:")
    print("-" * 80)
    for oid, result in TEST_RESULTS.items():
        for func, res in result.results.items():
            if res and not res["passed"]:
                print(f"{result.type_name} ({oid}) - {func}: {res['error']}")
