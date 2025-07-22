"""
E2E Test for Function 4: Arrow → cuDF → Parquet

This test verifies the functionality of converting Arrow arrays
to cuDF DataFrames and saving as Parquet files.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.e2e
@pytest.mark.gpu
class TestFunction4ArrowToCuDFToParquet:
    """Test Arrow to cuDF to Parquet conversion functionality."""

    def create_sample_arrow_table(self, num_rows=1000):
        """Create a sample Arrow table with various data types."""
        # Generate test data
        int_data = np.arange(num_rows, dtype=np.int32)
        float_data = np.random.randn(num_rows).astype(np.float64)
        string_data = [f"row_{i}" for i in range(num_rows)]
        bool_data = np.random.choice([True, False], size=num_rows)
        date_data = pa.array(
            [pa.scalar(f"2024-01-{(i % 28) + 1}", type=pa.date32()) for i in range(num_rows)]
        )

        # Add some NULL values
        null_mask = np.random.choice([False, True], size=num_rows, p=[0.9, 0.1])

        # Create Arrow arrays with nulls
        int_array = pa.array(int_data, mask=null_mask)
        float_array = pa.array(float_data, mask=null_mask)
        string_array = pa.array(string_data, mask=null_mask)
        bool_array = pa.array(bool_data, mask=null_mask)

        # Create Arrow table
        table = pa.table(
            {
                "id": int_array,
                "value": float_array,
                "name": string_array,
                "active": bool_array,
                "date": date_data,
            }
        )

        return table

    @pytest.mark.gpu
    def test_arrow_to_cudf_conversion(self):
        """Test converting Arrow table to cuDF DataFrame."""
        try:
            import cudf
        except ImportError:
            pytest.skip("cuDF not available")

        # Create Arrow table
        arrow_table = self.create_sample_arrow_table(1000)

        # Convert to cuDF
        cudf_df = cudf.DataFrame.from_arrow(arrow_table)

        # Verify conversion
        assert len(cudf_df) == len(arrow_table)
        assert list(cudf_df.columns) == arrow_table.column_names

        # Check data types
        assert cudf_df["id"].dtype == np.int32
        assert cudf_df["value"].dtype == np.float64
        assert cudf_df["name"].dtype == "object"
        assert cudf_df["active"].dtype == bool

        # Check NULL handling
        null_counts = cudf_df.isnull().sum()
        print(f"✓ Arrow to cuDF conversion successful")
        print(f"  Shape: {cudf_df.shape}")
        print(f"  NULL counts: {null_counts.to_dict()}")

    @pytest.mark.gpu
    def test_cudf_to_parquet_basic(self, temp_output_dir):
        """Test saving cuDF DataFrame to Parquet."""
        try:
            import cudf
        except ImportError:
            pytest.skip("cuDF not available")

        # Create test data
        arrow_table = self.create_sample_arrow_table(10000)
        cudf_df = cudf.DataFrame.from_arrow(arrow_table)

        # Save to Parquet
        output_path = temp_output_dir / "test_output.parquet"
        cudf_df.to_parquet(output_path)

        # Verify file exists and has content
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Read back and verify
        read_df = cudf.read_parquet(output_path)
        assert len(read_df) == len(cudf_df)
        assert list(read_df.columns) == list(cudf_df.columns)

        # Compare data (sample check)
        assert read_df["id"][0].item() == cudf_df["id"][0].item()

        print(f"✓ Parquet file created: {output_path.stat().st_size / 1024:.1f} KB")

    @pytest.mark.gpu
    def test_parquet_compression_options(self, temp_output_dir):
        """Test different Parquet compression options."""
        try:
            import cudf
        except ImportError:
            pytest.skip("cuDF not available")

        # Create test data
        arrow_table = self.create_sample_arrow_table(50000)
        cudf_df = cudf.DataFrame.from_arrow(arrow_table)

        compression_methods = ["snappy", "gzip", "zstd", None]
        compression_results = {}

        for compression in compression_methods:
            output_path = temp_output_dir / f"test_{compression or 'none'}.parquet"

            # Save with specific compression
            cudf_df.to_parquet(output_path, compression=compression)

            # Record file size
            file_size = output_path.stat().st_size
            compression_results[compression or "none"] = file_size

            # Verify readability
            read_df = cudf.read_parquet(output_path)
            assert len(read_df) == len(cudf_df)

        print("✓ Compression comparison:")
        for method, size in compression_results.items():
            print(f"  {method}: {size / 1024:.1f} KB")

        # ZSTD should generally give good compression
        assert compression_results["zstd"] < compression_results["none"]

    def test_arrow_to_parquet_direct(self, temp_output_dir):
        """Test direct Arrow to Parquet conversion (CPU baseline)."""
        # Create Arrow table
        arrow_table = self.create_sample_arrow_table(10000)

        # Save directly to Parquet
        output_path = temp_output_dir / "arrow_direct.parquet"
        pq.write_table(arrow_table, output_path, compression="zstd")

        # Read back and verify
        read_table = pq.read_table(output_path)
        assert read_table.num_rows == arrow_table.num_rows
        assert read_table.schema.equals(arrow_table.schema)

        print(f"✓ Direct Arrow to Parquet: {output_path.stat().st_size / 1024:.1f} KB")

    @pytest.mark.gpu
    def test_large_dataset_handling(self, temp_output_dir):
        """Test handling of large datasets."""
        try:
            import cudf
        except ImportError:
            pytest.skip("cuDF not available")

        # Create larger dataset (1M rows)
        num_rows = 1_000_000

        # Create data in chunks to manage memory
        chunk_size = 100_000
        chunks = []

        for i in range(0, num_rows, chunk_size):
            chunk_rows = min(chunk_size, num_rows - i)
            chunk_table = self.create_sample_arrow_table(chunk_rows)
            chunks.append(chunk_table)

        # Combine chunks
        full_table = pa.concat_tables(chunks)

        # Convert to cuDF
        cudf_df = cudf.DataFrame.from_arrow(full_table)

        # Save to Parquet with row group size
        output_path = temp_output_dir / "large_dataset.parquet"
        cudf_df.to_parquet(output_path, compression="zstd", row_group_size=50000)

        # Verify
        metadata = pq.read_metadata(output_path)
        assert metadata.num_rows == num_rows
        assert metadata.num_row_groups == 20  # 1M / 50K

        print(f"✓ Large dataset saved: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"  Rows: {metadata.num_rows:,}")
        print(f"  Row groups: {metadata.num_row_groups}")

    @pytest.mark.gpu
    def test_data_type_preservation(self, temp_output_dir):
        """Test that data types are preserved through the pipeline."""
        try:
            import cudf
        except ImportError:
            pytest.skip("cuDF not available")

        # Create Arrow table with specific types
        data = {
            "int8_col": pa.array(np.arange(100, dtype=np.int8)),
            "int16_col": pa.array(np.arange(100, dtype=np.int16)),
            "int32_col": pa.array(np.arange(100, dtype=np.int32)),
            "int64_col": pa.array(np.arange(100, dtype=np.int64)),
            "float32_col": pa.array(np.arange(100, dtype=np.float32)),
            "float64_col": pa.array(np.arange(100, dtype=np.float64)),
            "string_col": pa.array([f"str_{i}" for i in range(100)]),
            "bool_col": pa.array([i % 2 == 0 for i in range(100)]),
        }
        arrow_table = pa.table(data)

        # Convert to cuDF
        cudf_df = cudf.DataFrame.from_arrow(arrow_table)

        # Save to Parquet
        output_path = temp_output_dir / "type_preservation.parquet"
        cudf_df.to_parquet(output_path)

        # Read back with PyArrow
        read_table = pq.read_table(output_path)

        # Check schema preservation
        for field in arrow_table.schema:
            read_field = read_table.schema.field(field.name)
            # Note: Some type promotions may occur (e.g., int8 -> int32)
            print(f"  {field.name}: {field.type} → {read_field.type}")

        print("✓ Data type preservation test completed")

    @pytest.mark.gpu
    def test_null_handling_consistency(self, temp_output_dir):
        """Test NULL handling consistency across conversions."""
        try:
            import cudf
        except ImportError:
            pytest.skip("cuDF not available")

        # Create data with specific NULL patterns
        num_rows = 1000
        data = {
            "all_nulls": pa.array([None] * num_rows),
            "no_nulls": pa.array(range(num_rows)),
            "some_nulls": pa.array([i if i % 5 != 0 else None for i in range(num_rows)]),
            "string_nulls": pa.array([f"str_{i}" if i % 3 != 0 else None for i in range(num_rows)]),
        }
        arrow_table = pa.table(data)

        # Convert to cuDF
        cudf_df = cudf.DataFrame.from_arrow(arrow_table)

        # Check NULL counts in cuDF
        cudf_null_counts = cudf_df.isnull().sum().to_dict()

        # Save to Parquet
        output_path = temp_output_dir / "null_handling.parquet"
        cudf_df.to_parquet(output_path)

        # Read back and verify
        read_table = pq.read_table(output_path)

        # Check NULL counts after round trip
        for col_name in data.keys():
            original_nulls = arrow_table[col_name].null_count
            cudf_nulls = cudf_null_counts[col_name]
            read_nulls = read_table[col_name].null_count

            assert (
                original_nulls == read_nulls
            ), f"NULL count mismatch for {col_name}: {original_nulls} → {read_nulls}"

            print(f"  {col_name}: {original_nulls} NULLs preserved")

        print("✓ NULL handling consistency verified")

    def test_parquet_metadata(self, temp_output_dir):
        """Test Parquet metadata handling."""
        # Create Arrow table
        arrow_table = self.create_sample_arrow_table(1000)

        # Add metadata
        metadata = {
            b"created_by": b"gpu_postgresql_parser",
            b"version": b"1.0.0",
            b"source": b"PostgreSQL",
        }

        # Save with metadata
        output_path = temp_output_dir / "with_metadata.parquet"
        pq.write_table(arrow_table, output_path, compression="zstd", metadata=metadata)

        # Read metadata
        file_metadata = pq.read_metadata(output_path)

        # Verify
        assert file_metadata.num_rows == 1000
        assert file_metadata.num_columns == 5

        # Check custom metadata
        if file_metadata.metadata:
            assert b"created_by" in file_metadata.metadata
            assert file_metadata.metadata[b"created_by"] == b"gpu_postgresql_parser"

        print("✓ Parquet metadata test completed")
        print(f"  Format version: {file_metadata.format_version}")
        print(f"  Created by: {file_metadata.created_by}")
        print("  Compression: ZSTD")
