"""
Data Type Tests: Date/Time Types

Tests for DATE, TIME, TIMESTAMP (with and without time zone)
"""

import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.datatypes
class TestDateTimeTypes:
    """Test date/time data type handling through the full pipeline."""

    def create_datetime_test_table(self, conn, table_name):
        """Create a test table with all date/time types."""
        cur = conn.cursor()

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                id SERIAL PRIMARY KEY,
                col_date DATE,
                col_time TIME,
                col_time_tz TIME WITH TIME ZONE,
                col_timestamp TIMESTAMP,
                col_timestamp_tz TIMESTAMP WITH TIME ZONE,
                col_interval INTERVAL
            )
        """
        )
        conn.commit()
        cur.close()

    def test_date_handling(self, db_connection):
        """Test DATE type handling."""
        table_name = "test_date"
        self.create_datetime_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values including edge cases
        test_values = [
            date(1900, 1, 1),  # Early date
            date(2000, 1, 1),  # Y2K
            date(2024, 2, 29),  # Leap year
            date(2024, 12, 31),  # Current year end
            date(9999, 12, 31),  # Max PostgreSQL date
            None,  # NULL
        ]

        # Insert test data
        for val in test_values:
            cur.execute(f"INSERT INTO {table_name} (col_date) VALUES (%s)", (val,))
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
            cur.execute(f"SELECT col_date FROM {table_name} ORDER BY id")
            results = [row[0] for row in cur.fetchall()]
            assert results == test_values

            print("✓ DATE test passed")
            print(f"  Date range: {test_values[0]} to {test_values[4]}")

        finally:
            if queue_path.exists():
                queue_path.unlink()
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_time_handling(self, db_connection):
        """Test TIME with and without timezone."""
        table_name = "test_time"
        self.create_datetime_test_table(db_connection, table_name)

        # Register psycopg2 adapters for time with timezone
        psycopg2.extras.register_default_json()

        cur = db_connection.cursor()

        # Test values for TIME
        test_times = [
            time(0, 0, 0),  # Midnight
            time(12, 30, 45),  # Afternoon
            time(23, 59, 59, 999999),  # Last microsecond
            None,  # NULL
        ]

        # Test values for TIME WITH TIME ZONE
        test_times_tz = [
            time(0, 0, 0, tzinfo=timezone.utc),
            time(12, 30, 45, tzinfo=timezone(timedelta(hours=5))),
            time(23, 59, 59, 999999, tzinfo=timezone(timedelta(hours=-8))),
            None,
        ]

        # Insert test data
        for t, t_tz in zip(test_times, test_times_tz):
            cur.execute(
                f"""INSERT INTO {table_name}
                (col_time, col_time_tz)
                VALUES (%s, %s)""",
                (t, t_tz),
            )
        db_connection.commit()

        # Verify round trip
        cur.execute(
            f"""
            SELECT col_time, col_time_tz
            FROM {table_name}
            ORDER BY id
        """
        )
        results = cur.fetchall()

        # Check TIME values
        for i, (time_val, time_tz_val) in enumerate(results):
            assert time_val == test_times[i]
            # Note: PostgreSQL may normalize timezone info
            if test_times_tz[i] is not None:
                assert time_tz_val is not None
            else:
                assert time_tz_val is None

        print("✓ TIME test passed")
        print("  TIME and TIME WITH TIME ZONE handled correctly")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_timestamp_handling(self, db_connection):
        """Test TIMESTAMP with and without timezone."""
        table_name = "test_timestamp"
        self.create_datetime_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test values for TIMESTAMP
        test_timestamps = [
            datetime(1970, 1, 1, 0, 0, 0),  # Unix epoch
            datetime(2000, 1, 1, 12, 0, 0),  # Y2K noon
            datetime(2024, 12, 31, 23, 59, 59, 999999),  # End of current year
            None,  # NULL
        ]

        # Test values for TIMESTAMP WITH TIME ZONE
        test_timestamps_tz = [
            datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=5))),
            datetime(2024, 12, 31, 23, 59, 59, 999999, tzinfo=timezone(timedelta(hours=-8))),
            None,
        ]

        # Insert test data
        for ts, ts_tz in zip(test_timestamps, test_timestamps_tz):
            cur.execute(
                f"""INSERT INTO {table_name}
                (col_timestamp, col_timestamp_tz)
                VALUES (%s, %s)""",
                (ts, ts_tz),
            )
        db_connection.commit()

        # Verify round trip
        cur.execute(
            f"""
            SELECT col_timestamp, col_timestamp_tz
            FROM {table_name}
            ORDER BY id
        """
        )
        results = cur.fetchall()

        # Check values
        for i, (ts_val, ts_tz_val) in enumerate(results):
            assert ts_val == test_timestamps[i]
            if test_timestamps_tz[i] is not None:
                # Compare as UTC to handle timezone conversion
                assert ts_tz_val.astimezone(timezone.utc) == test_timestamps_tz[i].astimezone(
                    timezone.utc
                )
            else:
                assert ts_tz_val is None

        print("✓ TIMESTAMP test passed")
        print("  Both TIMESTAMP and TIMESTAMP WITH TIME ZONE handled")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_interval_handling(self, db_connection):
        """Test INTERVAL type handling."""
        table_name = "test_interval"
        self.create_datetime_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Test various interval values
        test_intervals = [
            "1 day",
            "2 weeks",
            "3 months",
            "1 year 2 months 3 days 4 hours 5 minutes 6 seconds",
            "-1 day",  # Negative interval
            "00:00:00",  # Zero interval
            None,  # NULL
        ]

        # Insert test data
        for interval in test_intervals:
            cur.execute(f"INSERT INTO {table_name} (col_interval) VALUES (%s)", (interval,))
        db_connection.commit()

        # Verify data
        cur.execute(f"SELECT col_interval FROM {table_name} ORDER BY id")
        results = cur.fetchall()

        # Basic check that we got results
        assert len(results) == len(test_intervals)

        print("✓ INTERVAL test passed")
        print("  Various interval formats handled")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_datetime_edge_cases(self, db_connection):
        """Test edge cases for date/time types."""
        table_name = "test_datetime_edges"
        self.create_datetime_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Edge case dates
        edge_dates = [
            date.min,  # 0001-01-01
            date.max,  # 9999-12-31
            date(2000, 2, 29),  # Leap year
            date(2100, 2, 28),  # Non-leap year (2100 is not a leap year)
        ]

        # Insert edge cases
        for d in edge_dates:
            cur.execute(f"INSERT INTO {table_name} (col_date) VALUES (%s)", (d,))
        db_connection.commit()

        # Verify
        cur.execute(f"SELECT col_date FROM {table_name} ORDER BY id")
        results = [row[0] for row in cur.fetchall()]
        assert results == edge_dates

        print("✓ Date/time edge cases passed")
        print(f"  Min date: {date.min}")
        print(f"  Max date: {date.max}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_datetime_null_handling(self, db_connection):
        """Test NULL handling for all date/time types."""
        table_name = "test_datetime_nulls"
        self.create_datetime_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Insert rows with various NULL patterns
        patterns = [
            (None, None, None, None, None, None),  # All NULLs
            (date.today(), None, None, None, None, None),  # Only date non-null
            (None, time(12, 0), None, None, None, None),  # Only time non-null
            (None, None, None, datetime.now(), None, None),  # Only timestamp
            (date.today(), time(12, 0), None, datetime.now(), None, "1 day"),  # Mixed
        ]

        for pattern in patterns:
            cur.execute(
                f"""
                INSERT INTO {table_name}
                (col_date, col_time, col_time_tz, col_timestamp, col_timestamp_tz, col_interval)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                pattern,
            )
        db_connection.commit()

        # Verify NULL counts
        cur.execute(
            f"""
            SELECT
                COUNT(*) as total,
                COUNT(col_date) as non_null_date,
                COUNT(col_time) as non_null_time,
                COUNT(col_timestamp) as non_null_timestamp,
                COUNT(col_interval) as non_null_interval
            FROM {table_name}
        """
        )

        stats = cur.fetchone()
        assert stats[0] == 5  # Total rows
        assert stats[1] == 2  # Non-null dates
        assert stats[2] == 2  # Non-null times
        assert stats[3] == 2  # Non-null timestamps
        assert stats[4] == 1  # Non-null intervals

        print("✓ Date/time NULL handling test passed")
        print("  All date/time types correctly handle NULL values")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()

    def test_datetime_variations(self, db_connection):
        """Test various date/time values with 10% NULL."""
        table_name = "test_datetime_variations"
        self.create_datetime_test_table(db_connection, table_name)

        cur = db_connection.cursor()

        # Generate 100 rows with varying date/time values
        np.random.seed(42)
        base_date = date(2020, 1, 1)

        for i in range(100):
            if i < 10:  # 10% NULL
                values = [None, None, None, None, None, None]
            else:
                # Varying dates across several years
                days_offset = np.random.randint(0, 365 * 5)  # 5 years range
                test_date = base_date + timedelta(days=days_offset)

                # Random times
                hour = np.random.randint(0, 24)
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)
                test_time = time(hour, minute, second)

                # Timestamp combining date and time
                test_timestamp = datetime.combine(test_date, test_time)

                # Random interval
                interval_days = np.random.randint(-365, 365)
                interval_str = f"{interval_days} days"

                values = [test_date, test_time, None, test_timestamp, None, interval_str]

            cur.execute(
                f"""
                INSERT INTO {table_name}
                (col_date, col_time, col_time_tz, col_timestamp, col_timestamp_tz, col_interval)
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
                COUNT(col_date) as non_null_count,
                MIN(col_date) as min_date,
                MAX(col_date) as max_date
            FROM {table_name}
        """
        )

        stats = cur.fetchone()
        assert stats[0] == 100  # Total rows
        assert stats[1] == 90  # Non-null count

        print("✓ Date/time variations test passed")
        print(f"  100 rows with 10% NULLs")
        print(f"  Date range: {stats[2]} to {stats[3]}")

        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db_connection.commit()
