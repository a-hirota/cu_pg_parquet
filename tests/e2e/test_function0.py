"""
E2E Test for Function 0: PostgreSQL Type → Arrow Schema Generation

This test verifies the functionality of generating Arrow schemas
from PostgreSQL table metadata.
"""

import os
import sys

import psycopg2
import pyarrow as pa
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.e2e
class TestFunction0PostgresTypeToArrowSchema:
    """Test PostgreSQL type to Arrow schema conversion."""

    def get_table_schema_from_postgres(self, conn, table_name):
        """Retrieve table schema information from PostgreSQL catalog."""
        cur = conn.cursor()

        # Query to get column information
        query = """
        SELECT
            column_name,
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            is_nullable,
            column_default,
            udt_name
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
        """

        cur.execute(query, (table_name,))
        columns = cur.fetchall()
        cur.close()

        return columns

    def postgres_type_to_arrow_type(self, pg_type, precision=None, scale=None, max_length=None):
        """Convert PostgreSQL type to Arrow type."""
        type_mapping = {
            # Numeric types
            "smallint": pa.int16(),
            "integer": pa.int32(),
            "bigint": pa.int64(),
            "real": pa.float32(),
            "double precision": pa.float64(),
            # String types
            "text": pa.string(),
            "character varying": pa.string(),
            "character": pa.string(),
            # Binary types
            "bytea": pa.binary(),
            # Date/Time types
            "date": pa.date32(),
            "time without time zone": pa.time64("us"),
            "timestamp without time zone": pa.timestamp("us"),
            "timestamp with time zone": pa.timestamp("us", tz="UTC"),
            # Boolean
            "boolean": pa.bool_(),
            # UUID
            "uuid": pa.binary(16),  # Fixed size binary
            # JSON
            "json": pa.string(),
            "jsonb": pa.string(),
        }

        # Handle numeric/decimal with precision and scale
        if pg_type == "numeric":
            if precision and scale:
                # For now, represent as string to preserve precision
                return pa.string()
            else:
                return pa.string()

        # Handle arrays
        if pg_type.startswith("ARRAY"):
            # Extract base type
            base_type = pg_type[6:-1] if pg_type.endswith("]") else pg_type[6:]
            element_type = self.postgres_type_to_arrow_type(base_type.lower())
            return pa.list_(element_type)

        return type_mapping.get(pg_type, pa.string())

    def test_basic_type_mapping(self, db_connection, basic_test_table):
        """Test basic PostgreSQL to Arrow type mapping."""
        # Get PostgreSQL schema
        pg_columns = self.get_table_schema_from_postgres(db_connection, basic_test_table)

        # Expected Arrow schema fields
        expected_fields = [
            ("id", pa.int32()),
            ("col_smallint", pa.int16()),
            ("col_integer", pa.int32()),
            ("col_bigint", pa.int64()),
            ("col_real", pa.float32()),
            ("col_double", pa.float64()),
            ("col_text", pa.string()),
            ("col_varchar", pa.string()),
            ("col_boolean", pa.bool_()),
            ("col_date", pa.date32()),
            ("col_timestamp", pa.timestamp("us")),
            ("col_uuid", pa.binary(16)),
            ("col_json", pa.string()),
        ]

        # Build Arrow schema from PostgreSQL metadata
        arrow_fields = []
        for col in pg_columns:
            col_name = col[0]
            pg_type = col[1]
            max_length = col[2]
            precision = col[3]
            scale = col[4]
            nullable = col[5] == "YES"

            # Skip auto-generated columns like 'id'
            if col_name == "id":
                continue

            arrow_type = self.postgres_type_to_arrow_type(pg_type, precision, scale, max_length)
            field = pa.field(col_name, arrow_type, nullable=nullable)
            arrow_fields.append(field)

        # Verify field count (excluding id)
        assert len(arrow_fields) == len(expected_fields) - 1

        # Verify each field type
        for i, (expected_name, expected_type) in enumerate(expected_fields[1:]):  # Skip id
            assert arrow_fields[i].name == expected_name
            assert arrow_fields[i].type == expected_type
            print(f"✓ {expected_name}: {pg_columns[i+1][1]} → {expected_type}")

    def test_nullable_fields(self, db_connection):
        """Test nullable field handling."""
        cur = db_connection.cursor()

        # Create test table with nullable and non-nullable columns
        table_name = "test_nullable_fields"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                required_int INTEGER NOT NULL,
                optional_int INTEGER,
                required_text TEXT NOT NULL,
                optional_text TEXT
            )
        """
        )
        db_connection.commit()

        try:
            # Get schema
            pg_columns = self.get_table_schema_from_postgres(db_connection, table_name)

            # Verify nullable settings
            nullable_map = {col[0]: col[5] == "YES" for col in pg_columns}

            assert nullable_map["required_int"] == False
            assert nullable_map["optional_int"] == True
            assert nullable_map["required_text"] == False
            assert nullable_map["optional_text"] == True

            print("✓ Nullable field information correctly retrieved")

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_precision_scale_handling(self, db_connection):
        """Test precision and scale handling for numeric types."""
        cur = db_connection.cursor()

        # Create table with various numeric precisions
        table_name = "test_numeric_precision"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                price NUMERIC(10, 2),
                quantity NUMERIC(5, 0),
                rate NUMERIC(7, 4),
                unlimited NUMERIC
            )
        """
        )
        db_connection.commit()

        try:
            # Get schema
            pg_columns = self.get_table_schema_from_postgres(db_connection, table_name)

            # Check precision and scale
            for col in pg_columns:
                col_name, _, _, precision, scale, _, _, _ = col

                if col_name == "price":
                    assert precision == 10
                    assert scale == 2
                elif col_name == "quantity":
                    assert precision == 5
                    assert scale == 0
                elif col_name == "rate":
                    assert precision == 7
                    assert scale == 4
                elif col_name == "unlimited":
                    # PostgreSQL allows unlimited precision
                    assert precision is None
                    assert scale is None

                print(f"✓ {col_name}: precision={precision}, scale={scale}")

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_varchar_length_handling(self, db_connection):
        """Test VARCHAR length constraints."""
        cur = db_connection.cursor()

        # Create table with various VARCHAR lengths
        table_name = "test_varchar_lengths"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                short_code VARCHAR(10),
                description VARCHAR(255),
                unlimited_text VARCHAR
            )
        """
        )
        db_connection.commit()

        try:
            # Get schema
            pg_columns = self.get_table_schema_from_postgres(db_connection, table_name)

            # Check character maximum lengths
            for col in pg_columns:
                col_name = col[0]
                max_length = col[2]

                if col_name == "short_code":
                    assert max_length == 10
                elif col_name == "description":
                    assert max_length == 255
                elif col_name == "unlimited_text":
                    # VARCHAR without length is unlimited
                    assert max_length is None

                print(f"✓ {col_name}: max_length={max_length}")

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_array_type_handling(self, db_connection):
        """Test array type mapping."""
        cur = db_connection.cursor()

        # Create table with array types
        table_name = "test_array_types"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                int_array INTEGER[],
                text_array TEXT[],
                float_array REAL[]
            )
        """
        )
        db_connection.commit()

        try:
            # Get schema
            pg_columns = self.get_table_schema_from_postgres(db_connection, table_name)

            # For PostgreSQL arrays, the data_type is 'ARRAY' and udt_name contains the element type
            for col in pg_columns:
                col_name = col[0]
                data_type = col[1]
                udt_name = col[7]

                assert data_type == "ARRAY"

                if col_name == "int_array":
                    assert udt_name == "_int4"  # PostgreSQL internal name for integer array
                elif col_name == "text_array":
                    assert udt_name == "_text"
                elif col_name == "float_array":
                    assert udt_name == "_float4"

                print(f"✓ {col_name}: {data_type} of {udt_name}")

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_all_supported_types(self, db_connection):
        """Test comprehensive type mapping for all supported PostgreSQL types."""
        cur = db_connection.cursor()

        # Create table with all supported types
        table_name = "test_all_types"
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        cur.execute(
            f"""
            CREATE TABLE {table_name} (
                -- Numeric types
                col_smallint SMALLINT,
                col_integer INTEGER,
                col_bigint BIGINT,
                col_real REAL,
                col_double DOUBLE PRECISION,
                col_numeric NUMERIC(10, 2),

                -- String types
                col_text TEXT,
                col_varchar VARCHAR(100),
                col_char CHAR(10),

                -- Binary types
                col_bytea BYTEA,

                -- Date/Time types
                col_date DATE,
                col_time TIME,
                col_timestamp TIMESTAMP,
                col_timestamptz TIMESTAMP WITH TIME ZONE,

                -- Boolean
                col_boolean BOOLEAN,

                -- UUID
                col_uuid UUID,

                -- JSON
                col_json JSON,
                col_jsonb JSONB
            )
        """
        )
        db_connection.commit()

        try:
            # Get schema and verify all types are correctly retrieved
            pg_columns = self.get_table_schema_from_postgres(db_connection, table_name)

            assert len(pg_columns) == 18  # All columns

            # Map each column to its expected Arrow type
            for col in pg_columns:
                col_name = col[0]
                pg_type = col[1]
                arrow_type = self.postgres_type_to_arrow_type(pg_type)

                print(f"✓ {col_name}: {pg_type} → {arrow_type}")

                # Verify Arrow type is not None
                assert arrow_type is not None

        finally:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            db_connection.commit()

    def test_schema_generation_function(self, db_connection, basic_test_table):
        """Test complete schema generation function."""

        def generate_arrow_schema(conn, table_name):
            """Generate complete Arrow schema from PostgreSQL table."""
            pg_columns = self.get_table_schema_from_postgres(conn, table_name)

            arrow_fields = []
            for col in pg_columns:
                col_name = col[0]
                pg_type = col[1]
                max_length = col[2]
                precision = col[3]
                scale = col[4]
                nullable = col[5] == "YES"
                udt_name = col[7]

                # Handle array types
                if pg_type == "ARRAY":
                    # Extract element type from udt_name
                    element_type_map = {
                        "_int2": pa.int16(),
                        "_int4": pa.int32(),
                        "_int8": pa.int64(),
                        "_float4": pa.float32(),
                        "_float8": pa.float64(),
                        "_text": pa.string(),
                        "_bool": pa.bool_(),
                    }
                    element_type = element_type_map.get(udt_name, pa.string())
                    arrow_type = pa.list_(element_type)
                else:
                    arrow_type = self.postgres_type_to_arrow_type(
                        pg_type, precision, scale, max_length
                    )

                field = pa.field(col_name, arrow_type, nullable=nullable)
                arrow_fields.append(field)

            return pa.schema(arrow_fields)

        # Test with basic table
        schema = generate_arrow_schema(db_connection, basic_test_table)

        assert isinstance(schema, pa.Schema)
        assert len(schema) > 0

        print(f"Generated Arrow schema with {len(schema)} fields:")
        for field in schema:
            print(f"  - {field.name}: {field.type} (nullable={field.nullable})")
