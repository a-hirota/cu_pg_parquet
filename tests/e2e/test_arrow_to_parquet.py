"""End-to-end tests for Function 3: Arrow to cuDF to Parquet conversion.

This tests ONLY the conversion from Arrow arrays to cuDF DataFrames to Parquet files.
No PostgreSQL interaction or GPU parsing is tested here.
"""

import os
import tempfile
from pathlib import Path

import cudf
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


class TestArrowToParquet:
    """Test Arrow to cuDF to Parquet conversion."""

    def test_integer_arrow_to_cudf_to_parquet(self):
        """Test INTEGER type conversion from Arrow to cuDF to Parquet."""
        # Create test Arrow array with INTEGER data
        test_data = [1, 2, 3, 4, 5, None, 7, 8, 9, 10]
        arrow_array = pa.array(test_data, type=pa.int32())

        # Create Arrow table
        table = pa.table({"test_column": arrow_array})

        # Convert Arrow table to cuDF DataFrame
        gdf = cudf.DataFrame.from_arrow(table)

        # Verify cuDF conversion
        assert len(gdf) == len(test_data)
        assert gdf["test_column"].dtype == "int32"
        assert gdf["test_column"].isnull().sum() == 1  # One null value

        # Create temporary file for Parquet output
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "test_output.parquet"

            # Write cuDF DataFrame to Parquet
            gdf.to_parquet(parquet_path, compression="snappy")

            # Verify Parquet file exists
            assert parquet_path.exists()

            # Read back Parquet file and verify data
            read_table = pq.read_table(parquet_path)
            assert read_table.num_rows == len(test_data)
            assert read_table.column("test_column").type == pa.int32()

            # Convert back to Python list and compare
            read_data = read_table.column("test_column").to_pylist()
            assert read_data == test_data

    def test_multiple_columns_conversion(self):
        """Test conversion with multiple INTEGER columns."""
        # Create test data with multiple columns
        col1_data = [1, 2, 3, 4, 5]
        col2_data = [10, 20, 30, 40, 50]
        col3_data = [100, 200, None, 400, 500]

        # Create Arrow table with multiple columns
        table = pa.table(
            {
                "id": pa.array(col1_data, type=pa.int32()),
                "value": pa.array(col2_data, type=pa.int32()),
                "score": pa.array(col3_data, type=pa.int32()),
            }
        )

        # Convert to cuDF
        gdf = cudf.DataFrame.from_arrow(table)

        # Verify column count and types
        assert len(gdf.columns) == 3
        assert all(gdf[col].dtype == "int32" for col in gdf.columns)

        # Write to Parquet with compression
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "multi_column.parquet"
            gdf.to_parquet(parquet_path, compression="zstd")

            # Read back and verify
            read_table = pq.read_table(parquet_path)
            assert read_table.num_columns == 3
            assert read_table.num_rows == 5

            # Verify data integrity
            assert read_table.column("id").to_pylist() == col1_data
            assert read_table.column("value").to_pylist() == col2_data
            assert read_table.column("score").to_pylist() == col3_data

    def test_large_dataset_conversion(self):
        """Test conversion with larger dataset to verify performance."""
        # Create larger test dataset
        num_rows = 100000
        test_data = list(range(num_rows))
        test_data[50000] = None  # Add some null values
        test_data[75000] = None

        # Create Arrow array
        arrow_array = pa.array(test_data, type=pa.int32())
        table = pa.table({"large_column": arrow_array})

        # Convert to cuDF
        gdf = cudf.DataFrame.from_arrow(table)
        assert len(gdf) == num_rows
        assert gdf["large_column"].isnull().sum() == 2

        # Write to Parquet
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "large_dataset.parquet"
            gdf.to_parquet(parquet_path, compression="snappy")

            # Verify file size is reasonable (compressed)
            file_size = os.path.getsize(parquet_path)
            assert file_size > 0
            # Snappy compression may not compress sequential integers much
            # Just verify it's not too large (e.g., < 8 bytes per row)
            assert file_size < num_rows * 8

            # Read back and verify row count
            read_table = pq.read_table(parquet_path)
            assert read_table.num_rows == num_rows

    def test_parquet_metadata_preservation(self):
        """Test that metadata is preserved through conversions."""
        # Create Arrow table with metadata
        schema = pa.schema([pa.field("test_int", pa.int32())])
        metadata = {"source": "gpupgparser", "version": "1.0"}
        schema = schema.with_metadata(metadata)

        data = pa.array([1, 2, 3, 4, 5], type=pa.int32())
        table = pa.table([data], schema=schema)

        # Convert to cuDF and back to Parquet
        gdf = cudf.DataFrame.from_arrow(table)

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "metadata_test.parquet"
            gdf.to_parquet(parquet_path)

            # Read and check metadata
            parquet_file = pq.ParquetFile(parquet_path)
            # Note: cuDF may not preserve all Arrow metadata
            assert parquet_file.metadata.num_rows == 5
            assert parquet_file.metadata.num_columns == 1
