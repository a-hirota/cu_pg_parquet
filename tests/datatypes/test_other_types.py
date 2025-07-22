"""
Data Type Tests: Other Types

Tests for BOOLEAN, BYTEA, UUID, JSON/JSONB
"""

import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.datatypes
class TestOtherTypes:
    """Test other data type handling through the full pipeline."""

    def create_other_test_table(self, conn, table_name):
        """Create a test table with other data types."""
        cur = conn.cursor()

        # Register UUID type
        psycopg2.extras.register_uuid()

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                col_boolean BOOLEAN,
                col_bytea BYTEA,
                col_uuid UUID,
                col_json JSON,
                col_jsonb JSONB
            )
        """
        )
        conn.commit()
        cur.close()

    def test_boolean_handling(self, db_connection):
        """Test BOOLEAN type handling."""
        table_name = "test_boolean"
        self.create_other_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values
        test_values = [
            True,
            False,
            None,  # NULL
            True,
            False,
            True,
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_boolean) VALUES (%s)", (val,))
        db_connection.commit()

        # Export to binary format
        queue_path = Path("/dev/shm") / f"{table_name}.bin"
        try:
            with open(queue_path, "wb") as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT BINARY)", f)

            # Verify data was written
            assert queue_path.exists()
            assert queue_path.stat().st_size > 0

            # Read back and verify
            cur.execute(f"SELECT col_boolean FROM {table_name} ORDER BY id")
            results = [row[0] for row in cur.fetchall()]
            assert results == test_values

            # Count true/false/null
            cur.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (WHERE col_boolean = true) as true_count,
                    COUNT(*) FILTER (WHERE col_boolean = false) as false_count,
                    COUNT(*) FILTER (WHERE col_boolean IS NULL) as null_count
                FROM {table_name}
            """
            )
            counts = cur.fetchone()
            assert counts[0] == 3  # True count
            assert counts[1] == 2  # False count
            assert counts[2] == 1  # NULL count

            print("✓ BOOLEAN test passed")
            print(f"  True: {counts[0]}, False: {counts[1]}, NULL: {counts[2]}")

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_bytea_handling(self, db_connection):
        """Test BYTEA (binary data) type handling."""
        table_name = "test_bytea"
        self.create_other_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values - various binary data
        test_values = [
            b"",  # Empty bytes
            b"Hello, World!",  # ASCII text
            b"\x00\x01\x02\x03\x04",  # Binary data
            b"\xff" * 100,  # Repeated bytes
            bytes(range(256)),  # All byte values
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_bytea) VALUES (%s)", (val,))
        db_connection.commit()

        # Verify round trip
        cur.execute(f"SELECT col_bytea FROM {table_name} ORDER BY id")
        results = cur.fetchall()

        # psycopg2 returns memoryview for bytea, convert to bytes
        for i, (result,) in enumerate(results):
            if test_values[i] is not None:
                assert bytes(result) == test_values[i]
            else:
                assert result is None

        print("✓ BYTEA test passed")
        print("  Binary data of various lengths handled")
        print(f"  Largest: {len(test_values[4])} bytes")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_uuid_handling(self, db_connection):
        """Test UUID type handling."""
        table_name = "test_uuid"
        self.create_other_test_table(db_connection, table_name)

        # Register UUID type
        psycopg2.extras.register_uuid()

        cur = db_connection.cursor()

        # Test values
        test_values = [
            uuid.UUID("00000000-0000-0000-0000-000000000000"),  # Nil UUID
            uuid.UUID("123e4567-e89b-12d3-a456-426614174000"),  # Standard UUID
            uuid.uuid4(),  # Random UUID
            uuid.uuid4(),  # Another random
            uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),  # Max UUID
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_uuid) VALUES (%s)", (val,))
        db_connection.commit()

        # Verify round trip
        cur.execute(f"SELECT col_uuid FROM {table_name} ORDER BY id")
        results = [row[0] for row in cur.fetchall()]
        assert results == test_values

        print("✓ UUID test passed")
        print("  Various UUID formats handled")
        print(f"  Example: {test_values[1]}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_json_handling(self, db_connection):
        """Test JSON and JSONB type handling."""
        table_name = "test_json"
        self.create_other_test_table(db_connection, table_name)

        # Register JSON types
        psycopg2.extras.register_default_json()

        cur = db_connection.cursor()

        # Test values - various JSON structures
        test_json_values = [
            {},  # Empty object
            {"key": "value"},  # Simple object
            {"nested": {"level": 2, "data": [1, 2, 3]}},  # Nested
            [1, 2, 3, 4, 5],  # Array
            {"unicode": "Hello 世界 🌍"},  # Unicode
            {"number": 42, "float": 3.14, "bool": True, "null": None},  # Types
            None,  # NULL
        ]

        # Insert test data for both JSON and JSONB
        for val in test_json_values:
            json_str = json.dumps(val) if val is not None else None
            cur.execute(
                f"""INSERT INTO {table_name}
                (col_json, col_jsonb)
                VALUES (%s, %s)""",
                (json_str, json_str),
            )
        db_connection.commit()

        # Verify round trip
        cur.execute(
            f"""
            SELECT col_json, col_jsonb
            FROM {table_name}
            ORDER BY id
        """
        )
        results = cur.fetchall()

        # Check JSON values
        for i, (json_val, jsonb_val) in enumerate(results):
            expected = test_json_values[i]
            if expected is not None:
                # JSON is returned as dict/list by psycopg2
                assert json_val == expected
                assert jsonb_val == expected
            else:
                assert json_val is None
                assert jsonb_val is None

        print("✓ JSON/JSONB test passed")
        print("  Various JSON structures handled")
        print("  Including nested objects, arrays, and Unicode")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_mixed_null_handling(self, db_connection):
        """Test NULL handling for all other types."""
        table_name = "test_other_nulls"
        self.create_other_test_table(db_connection, table_name)

        # Register types
        psycopg2.extras.register_uuid()
        psycopg2.extras.register_default_json()

        cur = db_connection.cursor()

        # Insert rows with various NULL patterns
        patterns = [
            (None, None, None, None, None),  # All NULLs
            (True, b"data", uuid.uuid4(), '{"a": 1}', '{"b": 2}'),  # No NULLs
            (False, None, uuid.uuid4(), None, '{"c": 3}'),  # Mixed
            (None, b"bytes", None, '{"d": 4}', None),  # Mixed
        ]

        for pattern in patterns:
            cur.execute(
                f"""
                INSERT INTO {table_name}
                (col_boolean, col_bytea, col_uuid, col_json, col_jsonb)
                VALUES (%s, %s, %s, %s, %s)
            """,
                pattern,
            )
        db_connection.commit()

        # Verify NULL counts
        cur.execute(
            f"""
            SELECT
                COUNT(*) as total,
                COUNT(col_boolean) as non_null_boolean,
                COUNT(col_bytea) as non_null_bytea,
                COUNT(col_uuid) as non_null_uuid,
                COUNT(col_json) as non_null_json,
                COUNT(col_jsonb) as non_null_jsonb
            FROM {table_name}
        """
        )

        stats = cur.fetchone()
        assert stats[0] == 4  # Total rows
        assert stats[1] == 2  # Non-null boolean
        assert stats[2] == 2  # Non-null bytea
        assert stats[3] == 2  # Non-null uuid
        assert stats[4] == 2  # Non-null json
        assert stats[5] == 2  # Non-null jsonb

        print("✓ NULL handling test passed for other types")
        print("  All types correctly handle NULL values")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_large_json_handling(self, db_connection):
        """Test handling of large JSON documents."""
        table_name = "test_large_json"
        self.create_other_test_table(db_connection, table_name)

        psycopg2.extras.register_default_json()
        cur = db_connection.cursor()

        # Create large JSON documents
        large_json_values = []

        # Large array
        large_array = list(range(1000))
        large_json_values.append(large_array)

        # Deep nesting
        deep_nested = {"level": 0}
        current = deep_nested
        for i in range(1, 20):
            current["next"] = {"level": i}
            current = current["next"]
        large_json_values.append(deep_nested)

        # Large object with many keys
        large_object = {f"key_{i}": f"value_{i}" for i in range(500)}
        large_json_values.append(large_object)

        # Insert large JSON values
        for val in large_json_values:
            json_str = json.dumps(val)
            cur.execute(
                f"INSERT INTO {table_name} (col_json, col_jsonb) VALUES (%s, %s)",
                (json_str, json_str),
            )
        db_connection.commit()

        # Verify data size
        cur.execute(
            f"""
            SELECT
                pg_column_size(col_json) as json_size,
                pg_column_size(col_jsonb) as jsonb_size
            FROM {table_name}
        """
        )

        sizes = cur.fetchall()
        print("✓ Large JSON handling test passed")
        for i, (json_size, jsonb_size) in enumerate(sizes):
            print(f"  Doc {i+1}: JSON {json_size} bytes, JSONB {jsonb_size} bytes")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_other_types_variations(self, db_connection):
        """Test various values for other types with 10% NULL."""
        table_name = "test_other_variations"
        self.create_other_test_table(db_connection, table_name)

        # Register types
        psycopg2.extras.register_uuid()
        psycopg2.extras.register_default_json()

        cur = db_connection.cursor()

        # Generate 100 rows with varying values
        np.random.seed(42)

        for i in range(100):
            if i < 10:  # 10% NULL
                values = [None, None, None, None, None]
            else:
                # Random boolean
                bool_val = bool(np.random.choice([True, False]))

                # Random binary data
                byte_len = np.random.randint(0, 100)
                byte_val = bytes(np.random.randint(0, 256, byte_len, dtype=np.uint8))

                # Random UUID
                uuid_val = uuid.uuid4()

                # Random JSON
                json_choice = np.random.randint(0, 4)
                if json_choice == 0:
                    json_obj = {"type": "object", "id": i}
                elif json_choice == 1:
                    json_obj = [i, i * 2, i * 3]
                elif json_choice == 2:
                    json_obj = {"nested": {"value": i}}
                else:
                    json_obj = f"string_{i}"
                json_val = json.dumps(json_obj)

                values = [bool_val, byte_val, uuid_val, json_val, json_val]

            cur.execute(
                f"""
                INSERT INTO {table_name}
                (col_boolean, col_bytea, col_uuid, col_json, col_jsonb)
                VALUES (%s, %s, %s, %s, %s)
            """,
                values,
            )

        db_connection.commit()

        # Verify statistics
        cur.execute(
            f"""
            SELECT
                COUNT(*) as total,
                COUNT(col_boolean) as non_null_count,
                SUM(CASE WHEN col_boolean = true THEN 1 ELSE 0 END) as true_count,
                AVG(LENGTH(col_bytea)) as avg_bytea_len
            FROM {table_name}
        """
        )

        stats = cur.fetchone()
        assert stats[0] == 100  # Total rows
        assert stats[1] == 90  # Non-null boolean count

        print("✓ Other types variations test passed")
        print("  100 rows with 10% NULLs")
        print(f"  Boolean true count: {stats[2]}")
        print(f"  Average BYTEA length: {stats[3]:.1f} bytes")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()
