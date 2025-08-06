"""E2E Test for Function 1: PostgreSQL to Binary Extraction and Metadata Generation

This test verifies:
1. PostgreSQL COPY BINARY extraction
2. Metadata generation from PostgreSQL catalog
3. Binary data format validation
4. Type mapping from PostgreSQL to Arrow
5. Rust binary extractor functionality (if available)

Test covers:
- Direct psycopg2 COPY BINARY
- Metadata extraction from pg_attribute
- Binary format parsing
- Type OID to Arrow mapping validation
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import psycopg2
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.types import INT32, PG_OID_TO_ARROW


@pytest.mark.e2e
class TestPostgresToBinary:
    """Test PostgreSQL to Binary extraction and metadata generation."""

    def parse_postgres_binary_header(self, data):
        """Parse PostgreSQL COPY BINARY header."""
        # PGCOPY\n\377\r\n\0
        if data[:6] != b"PGCOPY":
            raise ValueError("Invalid PostgreSQL binary format")

        # Skip header (11 bytes) + flags (4 bytes) + header extension (4 bytes)
        offset = 19
        return offset

    def parse_postgres_binary_row(self, data, offset):
        """Parse a single row from PostgreSQL binary format."""
        # Field count (2 bytes)
        field_count = struct.unpack(">h", data[offset : offset + 2])[0]
        offset += 2

        if field_count == -1:  # End of data marker
            return None, offset

        fields = []
        for _ in range(field_count):
            # Field length (4 bytes)
            field_len = struct.unpack(">i", data[offset : offset + 4])[0]
            offset += 4

            if field_len == -1:  # NULL value
                fields.append(None)
            else:
                # Field data
                field_data = data[offset : offset + field_len]
                offset += field_len
                fields.append(field_data)

        return fields, offset

    def test_postgres_metadata_generation(self, db_connection):
        """Test comprehensive metadata generation from PostgreSQL catalog."""
        table_name = "test_metadata_generation"
        cur = db_connection.cursor()

        # Create table with various data types
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                int_col INTEGER NOT NULL,
                bigint_col BIGINT,
                text_col TEXT,
                varchar_col VARCHAR(50),
                numeric_col NUMERIC(10, 2),
                bool_col BOOLEAN DEFAULT true,
                date_col DATE,
                timestamp_col TIMESTAMP,
                bytea_col BYTEA,
                real_col REAL,
                double_col DOUBLE PRECISION
            )
        """
        )
        db_connection.commit()

        try:
            # Import metadata utilities
            from src.readPostgres.metadata import fetch_column_meta

            # Fetch metadata using project's metadata utilities
            columns = fetch_column_meta(db_connection, f"SELECT * FROM {table_name}")

            # Verify metadata for each column
            assert len(columns) == 12, f"Expected 12 columns, got {len(columns)}"

            # Expected mappings
            expected_mappings = {
                "id": (23, INT32, 4),  # INTEGER
                "int_col": (23, INT32, 4),  # INTEGER
                "bigint_col": (20, 2, 8),  # BIGINT -> INT64
                "text_col": (25, 6, -1),  # TEXT -> UTF8
                "varchar_col": (1043, 6, -1),  # VARCHAR -> UTF8
                "numeric_col": (1700, 5, -1),  # NUMERIC -> DECIMAL128
                "bool_col": (16, 10, 1),  # BOOLEAN -> BOOL
                "date_col": (1082, 8, -1),  # DATE -> TS64_S
                "timestamp_col": (1114, 9, -1),  # TIMESTAMP -> TS64_US
                "bytea_col": (17, 7, -1),  # BYTEA -> BINARY
                "real_col": (700, 3, 4),  # REAL -> FLOAT32
                "double_col": (701, 4, 8),  # DOUBLE -> FLOAT64
            }

            for col in columns:
                assert col.name in expected_mappings, f"Unexpected column: {col.name}"

                expected_oid, expected_arrow_id, expected_size = expected_mappings[col.name]

                # Verify PostgreSQL OID
                assert (
                    col.pg_oid == expected_oid
                ), f"{col.name}: Expected OID {expected_oid}, got {col.pg_oid}"

                # Verify Arrow type mapping
                assert (
                    col.arrow_id == expected_arrow_id
                ), f"{col.name}: Expected Arrow ID {expected_arrow_id}, got {col.arrow_id}"

                # Verify element size
                assert (
                    col.elem_size == expected_size
                ), f"{col.name}: Expected size {expected_size}, got {col.elem_size}"

                # Check variable length flag
                expected_variable = expected_size == -1
                assert (
                    col.is_variable == expected_variable
                ), f"{col.name}: Expected is_variable={expected_variable}, got {col.is_variable}"

                print(
                    f"✓ {col.name}: PG OID {col.pg_oid} → Arrow {col.arrow_id} "
                    f"(size: {col.elem_size}, variable: {col.is_variable})"
                )

            # Test metadata generation for a query with expressions
            complex_query = f"""
                SELECT
                    id,
                    int_col * 2 as doubled,
                    text_col || '_suffix' as concatenated,
                    CASE WHEN bool_col THEN 1 ELSE 0 END as bool_as_int
                FROM {table_name}
            """

            complex_columns = fetch_column_meta(db_connection, complex_query)
            assert (
                len(complex_columns) == 4
            ), f"Expected 4 columns in complex query, got {len(complex_columns)}"

            # Verify derived column types
            col_map = {col.name: col for col in complex_columns}

            assert col_map["doubled"].pg_oid == 23, "Derived INTEGER column"
            assert col_map["concatenated"].pg_oid == 25, "String concatenation returns TEXT"
            assert col_map["bool_as_int"].pg_oid == 23, "CASE expression returns INTEGER"

            print("\n✓ Metadata generation test passed")
            print(f"  - Tested {len(columns)} column types")
            print(f"  - All PostgreSQL OIDs correctly mapped to Arrow types")
            print(f"  - Variable length flags correctly set")
            print(f"  - Complex query metadata correctly derived")

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_integer_rust_extraction(self, db_connection):
        """Test INTEGER type extraction using Rust binary extractor."""
        table_name = "test_integer_rust"
        cur = db_connection.cursor()

        # Create simple table with INTEGER only
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER,
                value INTEGER,
                nullable_value INTEGER
            )
        """
        )

        # Insert test data with NULLs
        test_data = [
            (1, 100, 1000),
            (2, 200, None),
            (3, 300, 3000),
            (4, None, 4000),
            (5, 500, None),
        ]

        for row in test_data:
            cur.execute(f"INSERT INTO {table_name} VALUES (%s, %s, %s)", row)
        db_connection.commit()

        # Test using Rust extractor
        # Get table metadata first
        cur.execute("""SELECT count(*) FROM pg_class WHERE relname = %s""", (table_name,))
        assert cur.fetchone()[0] == 1, f"Table {table_name} not found"

        # Set up environment for Rust extractor
        env = os.environ.copy()
        env.update(
            {
                "GPUPASER_PG_DSN": os.environ.get(
                    "GPUPASER_PG_DSN", "dbname=postgres user=postgres host=localhost port=5432"
                ),
                "TABLE_NAME": table_name,
                "CHUNK_ID": "0",
                "TOTAL_CHUNKS": "1",
                "RUST_LOG": "info",
                "RUST_PARALLEL_CONNECTIONS": "1",  # Single connection for testing
            }
        )

        # Find the Rust binary
        project_root = Path(__file__).parent.parent.parent
        rust_binary = project_root / "rust_pg_binary_extractor/target/release/pg_chunk_extractor"

        if not rust_binary.exists():
            pytest.skip(
                f"Rust binary not found at {rust_binary}. Run 'cd rust_pg_binary_extractor && cargo build --release'"
            )

        try:
            # Run Rust extractor
            result = subprocess.run(
                [str(rust_binary)], env=env, capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Rust extractor failed with code {result.returncode}")

            # Parse JSON output
            output_lines = result.stdout.strip().split("\n")
            json_line = None
            for line in output_lines:
                if line.startswith("{"):
                    json_line = line
                    break

            assert json_line, "No JSON output from Rust extractor"
            rust_result = json.loads(json_line)

            # Verify output file was created
            chunk_file = rust_result["chunk_file"]
            assert os.path.exists(chunk_file), f"Chunk file not created: {chunk_file}"

            # Verify file size
            file_size = os.path.getsize(chunk_file)
            assert file_size > 19, f"File too small: {file_size} bytes"  # At least header
            assert (
                file_size == rust_result["total_bytes"]
            ), f"File size mismatch: {file_size} != {rust_result['total_bytes']}"

            # Parse binary data from Rust output
            with open(chunk_file, "rb") as f:
                binary_data = f.read()

            offset = self.parse_postgres_binary_header(binary_data)
            parsed_rows = []

            while offset < len(binary_data) - 2:  # -2 for trailer
                row_fields, offset = self.parse_postgres_binary_row(binary_data, offset)
                if row_fields is None:
                    break
                parsed_rows.append(row_fields)

            # Verify metadata if present
            if "columns" in rust_result:
                columns = rust_result["columns"]
                assert len(columns) == 3, f"Expected 3 columns, got {len(columns)}"
                for col in columns:
                    assert (
                        col["pg_type_oid"] == 23
                    ), f"Expected INTEGER (OID 23), got {col['pg_type_oid']}"
                    assert col["arrow_type"] == "int32", f"Expected int32, got {col['arrow_type']}"

            # Get data via psycopg2 for comparison
            cur.execute(f"SELECT * FROM {table_name} ORDER BY id")
            psycopg2_rows = cur.fetchall()

            # Compare results
            assert len(parsed_rows) == len(
                psycopg2_rows
            ), f"Row count mismatch: {len(parsed_rows)} != {len(psycopg2_rows)}"

            for i, (binary_row, pg_row) in enumerate(zip(parsed_rows, psycopg2_rows)):
                # Compare each INTEGER field
                for j, (binary_field, pg_field) in enumerate(zip(binary_row, pg_row)):
                    if binary_field is None:
                        assert pg_field is None, f"Row {i}, Field {j}: Expected NULL"
                    else:
                        # All fields are INTEGER (4 bytes, big-endian)
                        binary_value = struct.unpack(">i", binary_field)[0]
                        assert (
                            binary_value == pg_field
                        ), f"Row {i}, Field {j}: {binary_value} != {pg_field}"

            print(f"✓ Rust extraction passed: {len(parsed_rows)} rows extracted")
            print(f"  File size: {file_size} bytes")
            print(f"  Elapsed time: {rust_result.get('elapsed_seconds', 'N/A')} seconds")
            print(f"  NULL handling verified")

            # Clean up
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_rust_producer_integration(self, db_connection):
        """Test integration with rust_producer() function from gpu_pipeline_processor."""
        table_name = "test_rust_producer"
        cur = db_connection.cursor()

        # Create table with INTEGER type
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                value INTEGER
            )
        """
        )

        # Insert more data for realistic test
        for i in range(100):
            cur.execute(f"INSERT INTO {table_name} VALUES (%s, %s)", (i, i * 10))
        db_connection.commit()

        # Import rust_producer function
        try:
            from processors.gpu_pipeline_processor import rust_producer
        except ImportError:
            sys.path.append(str(Path(__file__).parent.parent.parent / "processors"))
            from gpu_pipeline_processor import rust_producer

        # Create queues
        import queue
        import threading

        chunk_queue = queue.Queue(maxsize=3)
        stats_queue = queue.Queue()
        metadata_queue = queue.Queue()

        # Run rust_producer in a thread
        producer_thread = threading.Thread(
            target=rust_producer,
            args=(chunk_queue, 1, stats_queue, table_name, metadata_queue),
            name="TestRustProducer",
        )

        try:
            producer_thread.start()

            # Get metadata from queue
            metadata = metadata_queue.get(timeout=30)
            assert metadata is not None, "No metadata received"
            assert len(metadata) == 2, f"Expected 2 columns, got {len(metadata)}"

            # Get chunk info from queue
            chunk_info = chunk_queue.get(timeout=30)
            assert chunk_info is not None, "No chunk info received"

            # Verify chunk file exists
            assert os.path.exists(
                chunk_info["chunk_file"]
            ), f"Chunk file not found: {chunk_info['chunk_file']}"

            # Check None sentinel
            sentinel = chunk_queue.get(timeout=5)
            assert sentinel is None, "Expected None sentinel"

            # Wait for thread to complete
            producer_thread.join(timeout=5)

            print(f"✓ rust_producer integration passed")
            print(f"  Chunk file: {chunk_info['chunk_file']}")
            print(f"  File size: {chunk_info['file_size']} bytes")
            print(f"  Rust time: {chunk_info['rust_time']} seconds")

            # Clean up
            if os.path.exists(chunk_info["chunk_file"]):
                os.remove(chunk_info["chunk_file"])
            for worker_file in chunk_info.get("worker_files", []):
                if os.path.exists(worker_file):
                    os.remove(worker_file)

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_integer_metadata_mapping(self, db_connection):
        """Test INTEGER type metadata mapping."""
        table_name = "test_integer_metadata"
        cur = db_connection.cursor()

        # Create table with INTEGER type
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER NOT NULL,
                value INTEGER,
                amount INTEGER
            )
        """
        )

        # Get column metadata from PostgreSQL
        cur.execute(
            """
            SELECT
                attname as column_name,
                atttypid as type_oid,
                attnotnull as not_null
            FROM pg_attribute
            WHERE attrelid = %s::regclass
            AND attnum > 0
            AND NOT attisdropped
            ORDER BY attnum
        """,
            (table_name,),
        )

        columns = cur.fetchall()

        # Verify INTEGER type mapping
        for col_name, type_oid, not_null in columns:
            # INTEGER has OID 23
            assert type_oid == 23, f"{col_name}: Expected OID 23, got {type_oid}"

            # Check mapping to Arrow type
            assert type_oid in PG_OID_TO_ARROW, f"OID {type_oid} not in mapping table"
            arrow_id, elem_size = PG_OID_TO_ARROW[type_oid]

            # INTEGER should map to INT32 (Arrow ID 1)
            assert arrow_id == INT32, f"{col_name}: Expected Arrow ID {INT32}, got {arrow_id}"
            assert elem_size == 4, f"{col_name}: Expected size 4, got {elem_size}"

            print(
                f"✓ {col_name}: PostgreSQL INTEGER (OID {type_oid}) → Arrow INT32 (ID {arrow_id})"
            )
            print(f"  NOT NULL: {not_null}, Element size: {elem_size} bytes")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()
