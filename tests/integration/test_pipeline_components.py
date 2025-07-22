"""
Integration Test: Pipeline Components Test (CPU-only)

This test verifies individual pipeline components that don't require GPU.
"""

import decimal
import os
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.integration
class TestPipelineComponents:
    """Test individual pipeline components without GPU requirement."""

    def test_postgres_binary_export(self, db_connection):
        """Test PostgreSQL COPY BINARY format export."""
        table_name = "test_binary_export"

        cur = db_connection.cursor()

        # Create test table
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value DOUBLE PRECISION,
                created DATE
            )
        """
        )

        # Insert test data
        test_data = [
            (1, "First", 3.14, date(2024, 1, 1)),
            (2, "Second", 2.718, date(2024, 1, 2)),
            (3, None, None, None),  # NULL values
            (4, "Fourth", -1.0, date(2024, 1, 4)),
        ]

        for row in test_data:
            cur.execute(f"INSERT INTO {table_name} VALUES (%s, %s, %s, %s)", row)
        db_connection.commit()

        # Export to binary format
        queue_path = Path("/dev/shm") / f"{table_name}.bin"

        try:
            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            # Verify binary file
            assert queue_path.exists()
            file_size = queue_path.stat().st_size
            assert file_size > 0

            # Read binary header
            with open(queue_path, "rb") as f:
                # PostgreSQL binary format header
                signature = f.read(11)
                assert signature == b"PGCOPY\n\xff\r\n\x00"

                # Flags field
                flags = f.read(4)

                # Header extension
                header_ext_len = int.from_bytes(f.read(4), "big")
                if header_ext_len > 0:
                    f.read(header_ext_len)

            print(f"✓ Binary export successful: {file_size} bytes")
            print(f"  PostgreSQL binary format verified")

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            cur.close()

    def test_data_type_mapping(self, db_connection):
        """Test PostgreSQL to Arrow type mapping."""
        table_name = "test_type_mapping"

        cur = db_connection.cursor()

        # Create table with various types
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                col_smallint SMALLINT,
                col_integer INTEGER,
                col_bigint BIGINT,
                col_real REAL,
                col_double DOUBLE PRECISION,
                col_numeric NUMERIC(10, 2),
                col_text TEXT,
                col_varchar VARCHAR(50),
                col_date DATE,
                col_timestamp TIMESTAMP,
                col_boolean BOOLEAN
            )
        """
        )

        # Get column information
        cur.execute(
            f"""
            SELECT column_name, data_type, numeric_precision, numeric_scale,
                   character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """
        )

        columns = cur.fetchall()

        print("✓ PostgreSQL to Arrow type mapping:")

        # Expected Arrow type mappings
        type_map = {
            "smallint": pa.int16(),
            "integer": pa.int32(),
            "bigint": pa.int64(),
            "real": pa.float32(),
            "double precision": pa.float64(),
            "numeric": pa.decimal128,  # Will be created with precision/scale
            "text": pa.string(),
            "character varying": pa.string(),
            "date": pa.date32(),
            "timestamp without time zone": pa.timestamp("us"),
            "boolean": pa.bool_(),
        }

        for col_name, data_type, precision, scale, char_len in columns:
            if data_type == "numeric" and precision and scale:
                arrow_type = f"decimal128({precision}, {scale})"
            else:
                arrow_type = type_map.get(data_type, "unknown")
                if hasattr(arrow_type, "__name__"):
                    arrow_type = arrow_type.__name__
                elif hasattr(arrow_type, "__class__"):
                    arrow_type = str(arrow_type)

            print(f"  {col_name}: {data_type} → {arrow_type}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()
        cur.close()

    def test_null_handling_consistency(self, db_connection):
        """Test NULL handling across different stages."""
        table_name = "test_null_handling"

        cur = db_connection.cursor()

        # Create test table
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER,
                int_val INTEGER,
                text_val TEXT,
                float_val DOUBLE PRECISION,
                bool_val BOOLEAN
            )
        """
        )

        # Insert data with specific NULL patterns
        test_patterns = [
            (1, 100, "text", 3.14, True),  # No NULLs
            (2, None, "text", 3.14, True),  # NULL integer
            (3, 100, None, 3.14, True),  # NULL text
            (4, 100, "text", None, True),  # NULL float
            (5, 100, "text", 3.14, None),  # NULL boolean
            (6, None, None, None, None),  # All NULLs
        ]

        for row in test_patterns:
            cur.execute(f"INSERT INTO {table_name} VALUES (%s, %s, %s, %s, %s)", row)
        db_connection.commit()

        # Export and verify NULL counts
        queue_path = Path("/dev/shm") / f"{table_name}.bin"

        try:
            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            # Get NULL statistics from database
            cur.execute(
                f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT(int_val) as non_null_int,
                    COUNT(text_val) as non_null_text,
                    COUNT(float_val) as non_null_float,
                    COUNT(bool_val) as non_null_bool
                FROM {table_name}
            """
            )

            stats = cur.fetchone()

            print("✓ NULL handling verification:")
            print(f"  Total rows: {stats[0]}")
            print(f"  Non-NULL integers: {stats[1]} (NULLs: {stats[0] - stats[1]})")
            print(f"  Non-NULL text: {stats[2]} (NULLs: {stats[0] - stats[2]})")
            print(f"  Non-NULL floats: {stats[3]} (NULLs: {stats[0] - stats[3]})")
            print(f"  Non-NULL booleans: {stats[4]} (NULLs: {stats[0] - stats[4]})")

            assert stats[0] == 6  # Total rows
            assert stats[1] == 4  # 2 NULL integers
            assert stats[2] == 4  # 2 NULL texts
            assert stats[3] == 4  # 2 NULL floats
            assert stats[4] == 4  # 2 NULL booleans

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            cur.close()

    def test_large_binary_export_chunking(self, db_connection):
        """Test chunked export for large datasets."""
        table_name = "test_chunking"
        chunk_size = 10000
        num_chunks = 5
        total_rows = chunk_size * num_chunks

        cur = db_connection.cursor()

        # Create and populate large table
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """
        )

        # Insert data in batches
        print(f"  Creating test data: {total_rows} rows...")
        for chunk in range(num_chunks):
            values = [
                (chunk * chunk_size + i, f"Row {chunk * chunk_size + i}") for i in range(chunk_size)
            ]

            cur.executemany(f"INSERT INTO {table_name} VALUES (%s, %s)", values)
            db_connection.commit()

        # Export full dataset
        queue_path = Path("/dev/shm") / f"{table_name}.bin"

        try:
            import time

            start_time = time.time()

            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            export_time = time.time() - start_time
            file_size = queue_path.stat().st_size

            print(f"✓ Large dataset export:")
            print(f"  Rows: {total_rows:,}")
            print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
            print(f"  Export time: {export_time:.2f}s")
            print(f"  Throughput: {file_size / 1024 / 1024 / export_time:.1f} MB/s")

            # Verify row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            assert cur.fetchone()[0] == total_rows

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            cur.close()

    def test_arrow_parquet_conversion(self, temp_output_dir):
        """Test Arrow to Parquet conversion without GPU."""
        # Create Arrow table with various types
        num_rows = 10000

        data = {
            "id": pa.array(range(num_rows)),
            "value": pa.array(np.random.randn(num_rows)),
            "name": pa.array([f"item_{i}" for i in range(num_rows)]),
            "active": pa.array(np.random.choice([True, False], num_rows)),
            "timestamp": pa.array([datetime.now()] * num_rows),
        }

        # Add some NULLs
        null_mask = np.random.choice([False, True], num_rows, p=[0.9, 0.1])
        for key in data:
            if key != "id":  # Keep ID non-null
                data[key] = pa.array(
                    [None if null_mask[i] else data[key][i].as_py() for i in range(num_rows)]
                )

        arrow_table = pa.table(data)

        # Save to Parquet with different compression options
        compressions = ["snappy", "gzip", "zstd", "lz4"]

        print("✓ Arrow to Parquet conversion:")
        for compression in compressions:
            output_path = temp_output_dir / f"test_{compression}.parquet"

            start_time = time.time()
            pq.write_table(arrow_table, output_path, compression=compression, row_group_size=5000)
            write_time = time.time() - start_time

            # Read back and verify
            read_table = pq.read_table(output_path)
            assert read_table.num_rows == num_rows

            file_size = output_path.stat().st_size
            print(f"  {compression}: {file_size / 1024:.1f} KB ({write_time:.2f}s)")

        # Check metadata
        metadata = pq.read_metadata(output_path)
        print(f"  Row groups: {metadata.num_row_groups}")
        print(f"  Columns: {metadata.num_columns}")

    def test_data_integrity_verification(self, db_connection, temp_output_dir):
        """Test data integrity through export/import cycle."""
        table_name = "test_integrity"

        cur = db_connection.cursor()

        # Create table with precise numeric values
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                exact_int BIGINT,
                exact_decimal NUMERIC(20, 10),
                text_data TEXT
            )
        """
        )

        # Insert test data with specific values
        test_values = [
            (1, 1234567890123456789, decimal.Decimal("12345.6789012345"), "Test 1"),
            (2, -9223372036854775808, decimal.Decimal("-99999.9999999999"), "Test 2"),
            (3, 9223372036854775807, decimal.Decimal("0.0000000001"), "Test 3"),
            (4, 0, decimal.Decimal("0"), ""),  # Zero and empty string
            (5, None, None, None),  # NULLs
        ]

        for row in test_values:
            cur.execute(f"INSERT INTO {table_name} VALUES (%s, %s, %s, %s)", row)
        db_connection.commit()

        # Export to binary
        queue_path = Path("/dev/shm") / f"{table_name}.bin"

        try:
            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            # Create Arrow table from database data
            cur.execute(f"SELECT * FROM {table_name} ORDER BY id")
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

            # Convert to Arrow
            arrow_data = {}
            for i, col in enumerate(columns):
                col_values = [row[i] for row in rows]

                # Handle decimal type specially
                if col == "exact_decimal":
                    # Convert to float for Arrow (loses precision)
                    col_values = [float(v) if v is not None else None for v in col_values]
                    arrow_data[col] = pa.array(col_values, type=pa.float64())
                else:
                    arrow_data[col] = pa.array(col_values)

            arrow_table = pa.table(arrow_data)

            # Save to Parquet
            output_path = temp_output_dir / f"{table_name}.parquet"
            pq.write_table(arrow_table, output_path)

            # Read back and verify counts
            read_table = pq.read_table(output_path)
            assert read_table.num_rows == len(test_values)

            print("✓ Data integrity verification:")
            print(f"  Original rows: {len(test_values)}")
            print(f"  Binary file: {queue_path.stat().st_size} bytes")
            print(f"  Parquet file: {output_path.stat().st_size} bytes")
            print(f"  All row counts match")

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            cur.close()
