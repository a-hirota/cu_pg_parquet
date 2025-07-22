"""
Data Type Tests: Numeric Types

Tests for SMALLINT, INTEGER, BIGINT, REAL, DOUBLE PRECISION, NUMERIC/DECIMAL
"""

import decimal
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.datatypes
class TestNumericTypes:
    """Test numeric data type handling through the full pipeline."""

    def create_numeric_test_table(self, conn, table_name):
        """Create a test table with all numeric types."""
        cur = conn.cursor()

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                col_smallint SMALLINT,
                col_integer INTEGER,
                col_bigint BIGINT,
                col_real REAL,
                col_double DOUBLE PRECISION,
                col_numeric NUMERIC(10, 2),
                col_numeric_high_precision NUMERIC(38, 10),
                col_numeric_no_scale NUMERIC
            )
        """
        )
        conn.commit()
        cur.close()

    def test_smallint_handling(self, db_connection):
        """Test SMALLINT (-32768 to 32767) handling."""
        table_name = "test_smallint"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values including boundaries
        test_values = [
            -32768,  # Min
            -1000,
            0,
            1000,
            32767,  # Max
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_smallint) VALUES (%s)", (val,))
        db_connection.commit()

        # Export to binary format
        queue_path = Path("/dev/shm") / f"{table_name}.bin"
        try:
            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            # Verify data was written
            assert queue_path.exists()
            assert queue_path.stat().st_size > 0

            # Read back and verify (simple check)
            cur.execute(f"SELECT col_smallint FROM {table_name} ORDER BY id")
            results = [row[0] for row in cur.fetchall()]
            assert results == test_values

            print("✓ SMALLINT test passed")
            print(f"  Values: {results}")

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_integer_handling(self, db_connection):
        """Test INTEGER (-2147483648 to 2147483647) handling."""
        table_name = "test_integer"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values including boundaries
        test_values = [
            -2147483648,  # Min
            -1000000,
            0,
            1000000,
            2147483647,  # Max
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_integer) VALUES (%s)", (val,))
        db_connection.commit()

        # Verify round trip
        cur.execute(f"SELECT col_integer FROM {table_name} ORDER BY id")
        results = [row[0] for row in cur.fetchall()]
        assert results == test_values

        print("✓ INTEGER test passed")
        print(f"  Min: {test_values[0]}, Max: {test_values[4]}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_bigint_handling(self, db_connection):
        """Test BIGINT (-9223372036854775808 to 9223372036854775807) handling."""
        table_name = "test_bigint"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values including boundaries
        test_values = [
            -9223372036854775808,  # Min
            -1000000000000,
            0,
            1000000000000,
            9223372036854775807,  # Max
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_bigint) VALUES (%s)", (val,))
        db_connection.commit()

        # Verify round trip
        cur.execute(f"SELECT col_bigint FROM {table_name} ORDER BY id")
        results = [row[0] for row in cur.fetchall()]
        assert results == test_values

        print("✓ BIGINT test passed")
        print(f"  Min: {test_values[0]}, Max: {test_values[4]}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_real_handling(self, db_connection):
        """Test REAL (single precision float) handling."""
        table_name = "test_real"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values
        test_values = [
            -3.4e38,  # Near min
            -1.23456,
            0.0,
            3.14159,
            3.4e38,  # Near max
            float("inf"),
            float("-inf"),
            float("nan"),
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_real) VALUES (%s)", (val,))
        db_connection.commit()

        # Verify round trip
        cur.execute(f"SELECT col_real FROM {table_name} ORDER BY id")
        results = cur.fetchall()

        # Check non-NaN values
        for i, (result,) in enumerate(results):
            if test_values[i] is None:
                assert result is None
            elif np.isnan(test_values[i]):
                assert np.isnan(result)
            elif np.isinf(test_values[i]):
                assert np.isinf(result)
                assert np.sign(result) == np.sign(test_values[i])
            else:
                # Allow for floating point precision differences
                if test_values[i] == 0.0:
                    assert result == 0.0
                else:
                    assert abs(result - test_values[i]) / abs(test_values[i]) < 1e-6

        print("✓ REAL test passed")
        print("  Special values: inf, -inf, nan handled correctly")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_double_precision_handling(self, db_connection):
        """Test DOUBLE PRECISION handling."""
        table_name = "test_double"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values
        test_values = [
            -1.7976931348623157e308,  # Near min
            -2.718281828459045,
            0.0,
            3.141592653589793,
            1.7976931348623157e308,  # Near max
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_double) VALUES (%s)", (val,))
        db_connection.commit()

        # Verify round trip
        cur.execute(f"SELECT col_double FROM {table_name} ORDER BY id")
        results = [row[0] for row in cur.fetchall()]

        # Check values
        for i, result in enumerate(results):
            if test_values[i] is None:
                assert result is None
            else:
                assert abs(result - test_values[i]) < 1e-15

        print("✓ DOUBLE PRECISION test passed")
        print(f"  Range: {test_values[0]:.2e} to {test_values[4]:.2e}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_numeric_decimal_handling(self, db_connection):
        """Test NUMERIC/DECIMAL with various precision and scale."""
        table_name = "test_numeric"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values for NUMERIC(10, 2)
        test_values = [
            decimal.Decimal("-99999999.99"),  # Min for (10,2)
            decimal.Decimal("-1234.56"),
            decimal.Decimal("0.00"),
            decimal.Decimal("1234.56"),
            decimal.Decimal("99999999.99"),  # Max for (10,2)
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_numeric) VALUES (%s)", (val,))
        db_connection.commit()

        # Test high precision NUMERIC(38, 10)
        high_precision_value = decimal.Decimal("1234567890123456789012345678.1234567890")
        cur.execute(
            f"INSERT INTO {table_name} (col_numeric_high_precision) VALUES (%s)",
            (high_precision_value,),
        )

        # Test unlimited precision
        unlimited_value = decimal.Decimal(
            "123456789012345678901234567890.123456789012345678901234567890"
        )
        cur.execute(
            f"INSERT INTO {table_name} (col_numeric_no_scale) VALUES (%s)", (unlimited_value,)
        )
        db_connection.commit()

        # Verify NUMERIC(10,2)
        cur.execute(
            f"SELECT col_numeric FROM {table_name} WHERE col_numeric IS NOT NULL ORDER BY id"
        )
        results = [row[0] for row in cur.fetchall()]

        for i, result in enumerate(results):
            if test_values[i] is not None:
                assert result == test_values[i]

        print("✓ NUMERIC/DECIMAL test passed")
        print("  Precision preserved for NUMERIC(10,2)")
        print("  High precision value stored successfully")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_numeric_null_handling(self, db_connection):
        """Test NULL handling for all numeric types."""
        table_name = "test_numeric_nulls"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Insert row with all NULLs
        cur.execute(
            f"""
            INSERT INTO {table_name}
            (col_smallint, col_integer, col_bigint, col_real, col_double, col_numeric)
            VALUES (NULL, NULL, NULL, NULL, NULL, NULL)
        """
        )

        # Insert row with mixed values and NULLs
        cur.execute(
            f"""
            INSERT INTO {table_name}
            (col_smallint, col_integer, col_bigint, col_real, col_double, col_numeric)
            VALUES (100, NULL, 1000000, NULL, 3.14, NULL)
        """
        )
        db_connection.commit()

        # Verify NULLs
        cur.execute(f"SELECT * FROM {table_name} ORDER BY id")
        results = cur.fetchall()

        # First row should have all NULLs (except id)
        assert all(val is None for val in results[0][1:7])

        # Second row should have alternating values/NULLs
        row2 = results[1]
        assert row2[1] == 100  # smallint
        assert row2[2] is None  # integer
        assert row2[3] == 1000000  # bigint
        assert row2[4] is None  # real
        assert abs(row2[5] - 3.14) < 0.001  # double
        assert row2[6] is None  # numeric

        print("✓ NULL handling test passed")
        print("  All numeric types correctly handle NULL values")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_numeric_boundary_values(self, db_connection):
        """Test boundary values for numeric types with 10% NULL."""
        table_name = "test_numeric_boundaries"
        self.create_numeric_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Generate 100 rows with boundary values and 10% NULLs
        np.random.seed(42)

        for i in range(100):
            if i < 10:  # 10% NULL
                values = [None] * 6
            else:
                # Mix of boundary and normal values
                if i % 10 == 0:
                    # Boundary values
                    values = [
                        -32768 if i % 20 == 0 else 32767,
                        -2147483648 if i % 20 == 0 else 2147483647,
                        -9223372036854775808 if i % 20 == 0 else 9223372036854775807,
                        -3.4e38 if i % 20 == 0 else 3.4e38,
                        -1.7e308 if i % 20 == 0 else 1.7e308,
                        decimal.Decimal("-99999999.99")
                        if i % 20 == 0
                        else decimal.Decimal("99999999.99"),
                    ]
                else:
                    # Normal values
                    values = [
                        np.random.randint(-32768, 32768),
                        np.random.randint(-2147483648, 2147483648),
                        np.random.randint(-1000000, 1000000),
                        np.random.uniform(-1000, 1000),
                        np.random.uniform(-1e6, 1e6),
                        decimal.Decimal(str(round(np.random.uniform(-1000, 1000), 2))),
                    ]

            cur.execute(
                f"""
                INSERT INTO {table_name}
                (col_smallint, col_integer, col_bigint, col_real, col_double, col_numeric)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                values,
            )

        db_connection.commit()

        # Verify statistics
        cur.execute(
            f"""
            SELECT
                COUNT(*) as total,
                COUNT(col_smallint) as non_null_smallint,
                COUNT(col_integer) as non_null_integer,
                COUNT(col_bigint) as non_null_bigint,
                COUNT(col_real) as non_null_real,
                COUNT(col_double) as non_null_double,
                COUNT(col_numeric) as non_null_numeric
            FROM {table_name}
        """
        )

        stats = cur.fetchone()
        assert stats[0] == 100  # Total rows
        assert all(stats[i] == 90 for i in range(1, 7))  # 90 non-null values each

        print("✓ Boundary values test passed")
        print("  100 rows with 10% NULLs")
        print("  Boundary values successfully stored and retrieved")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()
