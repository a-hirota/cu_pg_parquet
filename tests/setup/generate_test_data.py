#!/usr/bin/env python3
"""
Test Data Generation Utilities

This module provides functions to generate test data for various PostgreSQL data types.
"""

import datetime
import decimal
import json
import logging
import os
import random
import string
import struct
import uuid

import psycopg2
from psycopg2 import sql

logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate test data for various PostgreSQL types."""

    def __init__(self, null_percentage=10):
        """Initialize generator with configurable null percentage."""
        self.null_percentage = null_percentage

    def should_be_null(self):
        """Randomly decide if a value should be NULL based on null_percentage."""
        return random.randint(1, 100) <= self.null_percentage

    # Numeric type generators
    def generate_smallint(self, include_bounds=False):
        """Generate SMALLINT values (-32768 to 32767)."""
        if self.should_be_null():
            return None
        if include_bounds:
            return random.choice([-32768, 32767])
        return random.randint(-32768, 32767)

    def generate_integer(self, include_bounds=False):
        """Generate INTEGER values (-2147483648 to 2147483647)."""
        if self.should_be_null():
            return None
        if include_bounds:
            return random.choice([-2147483648, 2147483647])
        return random.randint(-2147483648, 2147483647)

    def generate_bigint(self, include_bounds=False):
        """Generate BIGINT values (-9223372036854775808 to 9223372036854775807)."""
        if self.should_be_null():
            return None
        if include_bounds:
            return random.choice([-9223372036854775808, 9223372036854775807])
        return random.randint(-9223372036854775808, 9223372036854775807)

    def generate_real(self, include_bounds=False):
        """Generate REAL (float32) values."""
        if self.should_be_null():
            return None
        if include_bounds:
            # PostgreSQL REAL bounds
            return random.choice([-3.4028235e38, 3.4028235e38, 0.0])
        return random.uniform(-1e6, 1e6)

    def generate_double(self, include_bounds=False):
        """Generate DOUBLE PRECISION (float64) values."""
        if self.should_be_null():
            return None
        if include_bounds:
            # PostgreSQL DOUBLE bounds
            return random.choice([-1.7976931348623157e308, 1.7976931348623157e308, 0.0])
        return random.uniform(-1e10, 1e10)

    def generate_numeric(self, precision=10, scale=2):
        """Generate NUMERIC/DECIMAL values."""
        if self.should_be_null():
            return None
        max_val = 10 ** (precision - scale)
        val = decimal.Decimal(random.uniform(-max_val, max_val))
        return val.quantize(decimal.Decimal(10) ** -scale)

    # String type generators
    def generate_text(self, max_length=1000):
        """Generate TEXT values."""
        if self.should_be_null():
            return None
        length = random.randint(1, max_length)
        # Mix ASCII and some Unicode characters
        if random.random() < 0.3:  # 30% chance of Unicode
            unicode_chars = "".join(random.choices("あいうえおかきくけこ中文字符🌟🎉", k=10))
            ascii_chars = "".join(
                random.choices(string.ascii_letters + string.digits + " ", k=length - 10)
            )
            return unicode_chars + ascii_chars
        return "".join(random.choices(string.ascii_letters + string.digits + " ", k=length))

    def generate_varchar(self, max_length=100):
        """Generate VARCHAR values."""
        if self.should_be_null():
            return None
        length = random.randint(1, min(max_length, 100))
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def generate_char(self, length=10):
        """Generate CHAR values (fixed length)."""
        if self.should_be_null():
            return None
        text = "".join(random.choices(string.ascii_letters + string.digits, k=length))
        return text.ljust(length)  # Pad with spaces

    # Binary type generators
    def generate_bytea(self, max_length=100):
        """Generate BYTEA values."""
        if self.should_be_null():
            return None
        length = random.randint(1, max_length)
        return bytes(random.getrandbits(8) for _ in range(length))

    # Date/Time type generators
    def generate_date(self, include_bounds=False):
        """Generate DATE values."""
        if self.should_be_null():
            return None
        if include_bounds:
            # PostgreSQL DATE bounds
            return random.choice(
                [
                    datetime.date(1, 1, 1),  # Minimum
                    datetime.date(5874897, 12, 31),  # Maximum (year 5874897)
                ]
            )
        start_date = datetime.date(2020, 1, 1)
        end_date = datetime.date(2024, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        return start_date + datetime.timedelta(days=random_days)

    def generate_time(self):
        """Generate TIME values."""
        if self.should_be_null():
            return None
        return datetime.time(
            random.randint(0, 23),
            random.randint(0, 59),
            random.randint(0, 59),
            random.randint(0, 999999),
        )

    def generate_timestamp(self, with_timezone=False):
        """Generate TIMESTAMP values."""
        if self.should_be_null():
            return None
        dt = datetime.datetime(
            random.randint(2020, 2024),
            random.randint(1, 12),
            random.randint(1, 28),
            random.randint(0, 23),
            random.randint(0, 59),
            random.randint(0, 59),
            random.randint(0, 999999),
        )
        if with_timezone:
            import pytz

            tz = random.choice([pytz.UTC, pytz.timezone("US/Eastern"), pytz.timezone("Asia/Tokyo")])
            dt = tz.localize(dt)
        return dt

    # Other type generators
    def generate_boolean(self):
        """Generate BOOLEAN values."""
        if self.should_be_null():
            return None
        return random.choice([True, False])

    def generate_uuid(self):
        """Generate UUID values."""
        if self.should_be_null():
            return None
        return str(uuid.uuid4())

    def generate_json(self):
        """Generate JSON/JSONB values."""
        if self.should_be_null():
            return None
        data = {
            "id": random.randint(1, 1000),
            "name": f"item_{random.randint(1, 100)}",
            "values": [random.randint(1, 10) for _ in range(3)],
            "nested": {"field1": random.choice(["A", "B", "C"]), "field2": random.random()},
        }
        return json.dumps(data)

    def generate_array(self, base_type="integer", length=5):
        """Generate ARRAY values."""
        if self.should_be_null():
            return None
        if base_type == "integer":
            return [random.randint(-100, 100) for _ in range(length)]
        elif base_type == "text":
            return [f"item_{i}" for i in range(length)]
        elif base_type == "real":
            return [random.uniform(-10.0, 10.0) for _ in range(length)]
        else:
            raise ValueError(f"Unsupported array base type: {base_type}")


def create_test_table(conn, table_name, columns):
    """
    Create a test table with specified columns.

    Args:
        conn: psycopg2 connection
        table_name: Name of the table to create
        columns: List of tuples (column_name, column_type)
    """
    cur = conn.cursor()

    # Drop table if exists
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))

    # Build CREATE TABLE statement
    column_defs = []
    for col_name, col_type in columns:
        column_defs.append(sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(col_type)))

    create_stmt = sql.SQL("CREATE TABLE {} ({})").format(
        sql.Identifier(table_name), sql.SQL(", ").join(column_defs)
    )

    cur.execute(create_stmt)
    conn.commit()
    logger.info(f"Created table {table_name} with {len(columns)} columns")


def insert_test_data(conn, table_name, columns, data_rows):
    """
    Insert test data into a table.

    Args:
        conn: psycopg2 connection
        table_name: Name of the table
        columns: List of column names
        data_rows: List of data rows to insert
    """
    cur = conn.cursor()

    # Build INSERT statement
    insert_stmt = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(sql.Identifier(col) for col in columns),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )

    # Insert data in batches
    batch_size = 1000
    for i in range(0, len(data_rows), batch_size):
        batch = data_rows[i : i + batch_size]
        cur.executemany(insert_stmt, batch)
        conn.commit()
        logger.info(f"Inserted {len(batch)} rows into {table_name}")

    logger.info(f"Total rows inserted: {len(data_rows)}")


def create_basic_test_table(conn, rows=100):
    """Create a basic test table with common data types."""
    generator = TestDataGenerator(null_percentage=10)

    table_name = "test_basic_types"
    columns = [
        ("id", "SERIAL PRIMARY KEY"),
        ("col_smallint", "SMALLINT"),
        ("col_integer", "INTEGER"),
        ("col_bigint", "BIGINT"),
        ("col_real", "REAL"),
        ("col_double", "DOUBLE PRECISION"),
        ("col_text", "TEXT"),
        ("col_varchar", "VARCHAR(100)"),
        ("col_boolean", "BOOLEAN"),
        ("col_date", "DATE"),
        ("col_timestamp", "TIMESTAMP"),
        ("col_uuid", "UUID"),
        ("col_json", "JSONB"),
    ]

    # Create table
    create_test_table(conn, table_name, columns)

    # Generate data
    data_rows = []
    for i in range(rows):
        row = (
            generator.generate_smallint(),
            generator.generate_integer(),
            generator.generate_bigint(),
            generator.generate_real(),
            generator.generate_double(),
            generator.generate_text(100),
            generator.generate_varchar(100),
            generator.generate_boolean(),
            generator.generate_date(),
            generator.generate_timestamp(),
            generator.generate_uuid(),
            generator.generate_json(),
        )
        data_rows.append(row)

    # Add boundary value rows
    boundary_rows = [
        # Min values
        (
            -32768,
            -2147483648,
            -9223372036854775808,
            -3.4e38,
            -1.7e308,
            "min",
            "min",
            False,
            datetime.date(1, 1, 1),
            datetime.datetime(1, 1, 1),
            str(uuid.uuid4()),
            '{"type": "min"}',
        ),
        # Max values
        (
            32767,
            2147483647,
            9223372036854775807,
            3.4e38,
            1.7e308,
            "max",
            "max",
            True,
            datetime.date(9999, 12, 31),
            datetime.datetime(9999, 12, 31),
            str(uuid.uuid4()),
            '{"type": "max"}',
        ),
    ]
    data_rows.extend(boundary_rows)

    # Insert data
    column_names = [col[0] for col in columns[1:]]  # Skip id column
    insert_test_data(conn, table_name, column_names, data_rows)

    return table_name


if __name__ == "__main__":
    # Test the generator
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from test_config import get_test_dsn

    logging.basicConfig(level=logging.INFO)

    try:
        conn = psycopg2.connect(get_test_dsn())
        table_name = create_basic_test_table(conn, rows=100)

        # Verify data
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        print(f"Created test table '{table_name}' with {count} rows")

        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Failed to create test data: {e}")
        raise
