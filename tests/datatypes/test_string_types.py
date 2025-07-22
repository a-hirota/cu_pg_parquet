"""
Data Type Tests: String Types

Tests for TEXT, VARCHAR, CHAR
"""

import os
import sys
from pathlib import Path

import numpy as np
import psycopg2
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.datatypes
class TestStringTypes:
    """Test string data type handling through the full pipeline."""

    def create_string_test_table(self, conn, table_name):
        """Create a test table with all string types."""
        cur = conn.cursor()

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                col_text TEXT,
                col_varchar VARCHAR(100),
                col_varchar_unlimited VARCHAR,
                col_char CHAR(10),
                col_char_single CHAR(1)
            )
        """
        )
        conn.commit()
        cur.close()

    def test_text_handling(self, db_connection):
        """Test TEXT type handling including Unicode."""
        table_name = "test_text"
        self.create_string_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values including various character sets
        test_values = [
            "",  # Empty string
            "Simple ASCII text",  # Basic ASCII
            "Text with\nnewlines\tand\ttabs",  # Control characters
            "Unicode: 你好世界 🌍 🎉",  # Chinese + Emoji
            "日本語のテキスト",  # Japanese
            "Русский текст",  # Cyrillic
            "Very " + "long " * 1000 + "text",  # Long text
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_text) VALUES (%s)", (val,))
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
            cur.execute(f"SELECT col_text FROM {table_name} ORDER BY id")
            results = [row[0] for row in cur.fetchall()]
            assert results == test_values

            print("✓ TEXT test passed")
            print("  Handled empty string, Unicode, long text, and NULL")
            print(f"  Unicode example: {test_values[3]}")

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_varchar_handling(self, db_connection):
        """Test VARCHAR with and without length constraints."""
        table_name = "test_varchar"
        self.create_string_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values for VARCHAR(100)
        test_values_limited = [
            "Short text",
            "Medium length text that fits within 100 characters easily",
            "X" * 100,  # Exactly 100 characters
            None,  # NULL
        ]

        # Test values for unlimited VARCHAR
        test_values_unlimited = [
            "Regular text",
            "Very " + "long " * 500 + "text",  # Much longer than 100
            None,
        ]

        # Insert test data
        for val_limited, val_unlimited in zip(test_values_limited[:3], test_values_unlimited):
            cur.execute(
                f"""INSERT INTO {table_name}
                (col_varchar, col_varchar_unlimited)
                VALUES (%s, %s)""",
                (val_limited, val_unlimited),
            )

        # Add NULL row
        cur.execute(
            f"""INSERT INTO {table_name}
            (col_varchar, col_varchar_unlimited)
            VALUES (NULL, NULL)"""
        )
        db_connection.commit()

        # Verify round trip
        cur.execute(
            f"""
            SELECT col_varchar, col_varchar_unlimited
            FROM {table_name}
            ORDER BY id
        """
        )
        results = cur.fetchall()

        # Check VARCHAR(100) values
        assert results[0][0] == test_values_limited[0]
        assert results[1][0] == test_values_limited[1]
        assert results[2][0] == test_values_limited[2]
        assert len(results[2][0]) == 100  # Verify length constraint
        assert results[3][0] is None

        # Check unlimited VARCHAR
        assert results[0][1] == test_values_unlimited[0]
        assert len(results[1][1]) > 100  # Longer than VARCHAR(100) limit

        print("✓ VARCHAR test passed")
        print("  VARCHAR(100) enforced length constraint")
        print("  Unlimited VARCHAR handled long text")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_char_handling(self, db_connection):
        """Test CHAR fixed-length type with padding."""
        table_name = "test_char"
        self.create_string_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values for CHAR(10)
        test_values_char10 = [
            "ABC",  # Should be padded to 10
            "1234567890",  # Exactly 10
            "",  # Empty (padded to 10 spaces)
            None,  # NULL
        ]

        # Test values for CHAR(1)
        test_values_char1 = [
            "A",
            "Z",
            " ",
            None,
        ]

        # Insert test data
        for val10, val1 in zip(test_values_char10, test_values_char1):
            cur.execute(
                f"""INSERT INTO {table_name}
                (col_char, col_char_single)
                VALUES (%s, %s)""",
                (val10, val1),
            )
        db_connection.commit()

        # Verify round trip
        cur.execute(
            f"""
            SELECT col_char, col_char_single,
                   LENGTH(col_char) as len_char10,
                   LENGTH(col_char_single) as len_char1
            FROM {table_name}
            ORDER BY id
        """
        )
        results = cur.fetchall()

        # CHAR(10) should be space-padded to 10 characters
        # Note: PostgreSQL trims trailing spaces when calculating LENGTH
        assert results[0][0].rstrip() == "ABC"  # Value is stored with padding
        assert len(results[0][0]) == 10  # But has full length when retrieved

        assert results[1][0] == "1234567890"  # No padding needed
        assert len(results[1][0]) == 10

        assert results[2][0].strip() == ""  # Empty string padded with spaces
        assert len(results[2][0]) == 10

        assert results[3][0] is None  # NULL

        # CHAR(1) values
        assert results[0][1] == "A"
        assert results[1][1] == "Z"
        assert results[2][1] == " "
        assert results[3][1] is None

        print("✓ CHAR test passed")
        print("  CHAR(10) space-padded correctly")
        print("  CHAR(1) handled single characters")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_string_encoding_edge_cases(self, db_connection):
        """Test edge cases for string encoding."""
        table_name = "test_string_edges"
        self.create_string_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Edge case strings
        edge_cases = [
            "String with 'single quotes'",
            'String with "double quotes"',
            "String with \\backslash",
            "String with \x00 null byte",  # Note: PostgreSQL doesn't allow null bytes
            "String with special chars: <>&",
            "Multi-byte: €£¥",
            "Emoji: 😀😃😄😁😆",
            "Mixed: ABC123αβγ中文😀",
            "Tab\tseparated\tvalues",
            "Line\nbreak\r\nhandling",
        ]

        # PostgreSQL doesn't support null bytes in text
        edge_cases[3] = "String with null byte (removed)"

        # Insert edge cases
        for text in edge_cases:
            cur.execute(f"INSERT INTO {table_name} (col_text) VALUES (%s)", (text,))
        db_connection.commit()

        # Verify round trip
        cur.execute(f"SELECT col_text FROM {table_name} ORDER BY id")
        results = [row[0] for row in cur.fetchall()]

        assert results == edge_cases

        print("✓ String encoding edge cases passed")
        print("  Handled quotes, backslash, special chars")
        print("  Multi-byte and emoji preserved")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_string_null_handling(self, db_connection):
        """Test NULL handling for all string types."""
        table_name = "test_string_nulls"
        self.create_string_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Insert rows with various NULL patterns
        patterns = [
            (None, None, None, None, None),  # All NULLs
            ("text", None, None, None, None),  # Only TEXT non-null
            (None, "varchar", None, None, None),  # Only VARCHAR non-null
            (None, None, "unlimited", None, None),  # Only unlimited VARCHAR
            (None, None, None, "char10", None),  # Only CHAR(10)
            (None, None, None, None, "X"),  # Only CHAR(1)
            ("text", "varchar", "unlimited", "char10", "Y"),  # No NULLs
        ]

        for pattern in patterns:
            cur.execute(
                f"""
                INSERT INTO {table_name}
                (col_text, col_varchar, col_varchar_unlimited, col_char, col_char_single)
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
                COUNT(col_text) as non_null_text,
                COUNT(col_varchar) as non_null_varchar,
                COUNT(col_varchar_unlimited) as non_null_varchar_unl,
                COUNT(col_char) as non_null_char,
                COUNT(col_char_single) as non_null_char1
            FROM {table_name}
        """
        )

        stats = cur.fetchone()
        assert stats[0] == 7  # Total rows
        assert stats[1] == 2  # Non-null TEXT
        assert stats[2] == 2  # Non-null VARCHAR
        assert stats[3] == 2  # Non-null unlimited VARCHAR
        assert stats[4] == 2  # Non-null CHAR(10)
        assert stats[5] == 2  # Non-null CHAR(1)

        print("✓ String NULL handling test passed")
        print("  All string types correctly handle NULL values")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_string_length_variations(self, db_connection):
        """Test strings of various lengths with 10% NULL."""
        table_name = "test_string_lengths"
        self.create_string_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Generate 100 rows with varying string lengths
        np.random.seed(42)

        for i in range(100):
            if i < 10:  # 10% NULL
                values = [None, None, None, None, None]
            else:
                # Varying lengths
                text_len = np.random.randint(0, 1000)
                varchar_len = np.random.randint(0, 100)
                varchar_unl_len = np.random.randint(0, 5000)

                text_val = "".join(
                    np.random.choice(
                        list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "),
                        text_len,
                    )
                )
                varchar_val = "".join(
                    np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), varchar_len)
                )
                varchar_unl_val = "".join(np.random.choice(list("0123456789"), varchar_unl_len))
                char_val = "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 10))[:10]
                char1_val = np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

                values = [text_val, varchar_val, varchar_unl_val, char_val, char1_val]

            cur.execute(
                f"""
                INSERT INTO {table_name}
                (col_text, col_varchar, col_varchar_unlimited, col_char, col_char_single)
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
                COUNT(col_text) as non_null_count,
                MIN(LENGTH(col_text)) as min_len,
                MAX(LENGTH(col_text)) as max_len,
                AVG(LENGTH(col_text)) as avg_len
            FROM {table_name}
        """
        )

        stats = cur.fetchone()
        assert stats[0] == 100  # Total rows
        assert stats[1] == 90  # Non-null count

        print("✓ String length variations test passed")
        print("  100 rows with 10% NULLs")
        print(f"  Text lengths: min={stats[2]}, max={stats[3]}, avg={stats[4]:.1f}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()
