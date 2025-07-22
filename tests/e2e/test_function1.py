"""
E2E Test for Function 1: PostgreSQL → /dev/shm Queue

This test verifies the functionality of reading data from PostgreSQL
and writing it to a shared memory queue.
"""

import os

# Import project modules
import sys
import tempfile
import time
from pathlib import Path

import psycopg2
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.e2e
class TestFunction1PostgresToQueue:
    """Test PostgreSQL to /dev/shm queue functionality."""

    def test_basic_copy_to_queue(self, db_connection, basic_test_table):
        """Test basic COPY BINARY to queue functionality."""
        cur = db_connection.cursor()

        # Create a test queue path in /dev/shm
        queue_path = Path("/dev/shm") / f"test_queue_{os.getpid()}.bin"

        try:
            # Execute COPY BINARY command
            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {basic_test_table} TO STDOUT WITH (FORMAT BINARY)", f)

            # Verify file was created and has content
            assert queue_path.exists()
            file_size = queue_path.stat().st_size
            assert file_size > 0

            # Verify PostgreSQL binary format header (PGCOPY\n\377\r\n\0)
            with open(queue_path, "rb") as f:
                header = f.read(11)
                assert header[:6] == b"PGCOPY"
                assert header[6] == 0x0A  # \n
                assert header[7] == 0xFF  # \377
                assert header[8] == 0x0D  # \r
                assert header[9] == 0x0A  # \n
                assert header[10] == 0x00  # \0

            print(f"Successfully copied data to queue: {file_size} bytes")

        finally:
            # Cleanup
            if queue_path.exists():
                queue_path.unlink()

    def test_large_data_copy(self, db_connection):
        """Test copying large amount of data."""
        cur = db_connection.cursor()

        # Create a larger test table
        table_name = "test_large_data"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                data TEXT,
                value DOUBLE PRECISION,
                created_at TIMESTAMP
            )
        """
        )

        # Insert 10,000 rows
        print("Inserting 10,000 rows...")
        for i in range(100):  # Insert in batches
            values = []
            for j in range(100):
                row_id = i * 100 + j
                values.append(
                    (
                        f"data_{row_id}" * 10,  # ~50 chars per row
                        row_id * 3.14159,
                        f"2024-01-{(row_id % 28) + 1:02d} 12:00:00",
                    )
                )

            cur.executemany(
                f"INSERT INTO {table_name} (data, value, created_at) VALUES (%s, %s, %s)", values
            )
        db_connection.commit()

        # Test copy
        queue_path = Path("/dev/shm") / f"test_large_queue_{os.getpid()}.bin"

        try:
            start_time = time.time()

            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            elapsed_time = time.time() - start_time
            file_size = queue_path.stat().st_size
            throughput = file_size / elapsed_time / (1024 * 1024)  # MB/s

            print(f"Copied {file_size:,} bytes in {elapsed_time:.2f}s")
            print(f"Throughput: {throughput:.2f} MB/s")

            # Verify reasonable performance
            assert throughput > 10  # At least 10 MB/s
            assert file_size > 1_000_000  # At least 1MB

        finally:
            # Cleanup
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            if queue_path.exists():
                queue_path.unlink()

    def test_multiple_data_types(self, db_connection):
        """Test COPY with various PostgreSQL data types."""
        cur = db_connection.cursor()

        # Create table with multiple types
        table_name = "test_multiple_types"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                col_smallint SMALLINT,
                col_integer INTEGER,
                col_bigint BIGINT,
                col_real REAL,
                col_double DOUBLE PRECISION,
                col_text TEXT,
                col_varchar VARCHAR(50),
                col_boolean BOOLEAN,
                col_date DATE,
                col_timestamp TIMESTAMP,
                col_bytea BYTEA,
                col_uuid UUID,
                col_json JSONB
            )
        """
        )

        # Insert test data including NULLs
        test_data = [
            (
                -32768,
                -2147483648,
                -9223372036854775808,
                -3.14,
                -2.718281828,
                "Hello",
                "World",
                True,
                "2024-01-01",
                "2024-01-01 12:00:00",
                b"\x00\x01\x02\x03",
                "550e8400-e29b-41d4-a716-446655440000",
                '{"key": "value"}',
            ),
            (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),  # NULL row
            (
                32767,
                2147483647,
                9223372036854775807,
                3.14,
                2.718281828,
                "Unicode: 你好世界 🌍",
                "Test",
                False,
                "2024-12-31",
                "2024-12-31 23:59:59",
                b"\xFF\xFE\xFD",
                "550e8400-e29b-41d4-a716-446655440001",
                '{"array": [1, 2, 3]}',
            ),
        ]

        for row in test_data:
            cur.execute(
                f"""
                INSERT INTO {table_name} VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """,
                row,
            )
        db_connection.commit()

        # Test copy
        queue_path = Path("/dev/shm") / f"test_types_queue_{os.getpid()}.bin"

        try:
            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            # Verify file contains data
            file_size = queue_path.stat().st_size
            assert file_size > 100  # Should have substantial data

            # Read and verify structure
            with open(queue_path, "rb") as f:
                # Skip header
                f.seek(11)

                # Read flags (4 bytes)
                flags = f.read(4)
                assert len(flags) == 4

                # Read header extension (4 bytes)
                ext_len = f.read(4)
                assert len(ext_len) == 4

            print(f"Successfully handled multiple data types: {file_size} bytes")

        finally:
            # Cleanup
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()
            if queue_path.exists():
                queue_path.unlink()

    @pytest.mark.slow
    def test_concurrent_access(self, db_connection):
        """Test concurrent reading and writing to queue."""
        # This test would require implementing actual queue logic
        # For now, we'll create a simple simulation
        pytest.skip("Concurrent queue access test not yet implemented")

    def test_error_handling(self, db_connection):
        """Test error handling for invalid operations."""
        cur = db_connection.cursor()

        # Test with non-existent table
        queue_path = Path("/dev/shm") / f"test_error_queue_{os.getpid()}.bin"

        try:
            with open(queue_path, "wb") as f:
                with pytest.raises(psycopg2.Error):
                    cur.copy_expert("COPY non_existent_table TO STDOUT WITH (FORMAT BINARY)", f)
        finally:
            if queue_path.exists():
                queue_path.unlink()

        # Test with invalid path (read-only location)
        try:
            with open("/dev/null/invalid_path", "wb") as f:
                pass
        except (OSError, IOError):
            # Expected error
            pass
