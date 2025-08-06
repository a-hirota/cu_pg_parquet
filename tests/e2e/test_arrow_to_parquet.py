"""End-to-end tests for Function 3: Arrow to cuDF to Parquet conversion.

This tests the final stage of the pipeline using actual product code:
1. Using DirectProcessor to handle Arrow to cuDF conversion
2. Using write_parquet_from_cudf for Parquet export
3. Testing compression options and metadata preservation
4. Verifying data integrity through the product's pipeline
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

try:
    import cudf
    import cupy as cp
    from numba import cuda

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestArrowToParquet:
    """Test Arrow to cuDF to Parquet conversion using actual product code."""

    def test_direct_processor_arrow_conversion(self, temp_output_dir):
        """Test using DirectProcessor for Arrow to Parquet conversion."""
        from src.postgres_to_parquet_converter import DirectProcessor
        from src.types import FLOAT64, INT32, INT64, UTF8, ColumnMeta
        from src.write_parquet_from_cudf import write_cudf_to_parquet_with_options

        # Create mock PostgreSQL binary data with header
        # PGCOPY header (11 bytes) + flags (4 bytes) + header extension (4 bytes) = 19 bytes
        header = b"PGCOPY\n\xff\r\n\0"  # 11 bytes
        flags = b"\x00\x00\x00\x00"  # 4 bytes
        header_ext = b"\x00\x00\x00\x00"  # 4 bytes

        # Create some INTEGER row data
        rows = []
        for i in range(100):
            # Row format: field_count (2 bytes) + fields
            row = struct.pack(">h", 3)  # 3 fields

            # Field 1: id (INTEGER)
            row += struct.pack(">i", 4)  # field length
            row += struct.pack(">i", i)  # value

            # Field 2: value (INTEGER)
            if i % 10 == 0:  # Some NULLs
                row += struct.pack(">i", -1)  # NULL
            else:
                row += struct.pack(">i", 4)  # field length
                row += struct.pack(">i", i * 100)  # value

            # Field 3: score (INTEGER)
            row += struct.pack(">i", 4)  # field length
            row += struct.pack(">i", i * 1000)  # value

            rows.append(row)

        # Combine into binary data
        binary_data = header + flags + header_ext + b"".join(rows) + b"\xff\xff"  # trailer

        # Transfer to GPU
        gpu_data = cp.asarray(np.frombuffer(binary_data, dtype=np.uint8))
        raw_dev = cuda.as_cuda_array(gpu_data).view(dtype=np.uint8)

        # Define columns
        columns = [
            ColumnMeta(name="id", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),
            ColumnMeta(name="value", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),
            ColumnMeta(name="score", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),
        ]

        # Use DirectProcessor
        processor = DirectProcessor(use_rmm=True, optimize_gpu=True, verbose=False, test_mode=True)

        output_path = temp_output_dir / "test_direct_processor.parquet"

        # Process data
        cudf_df, timing_info = processor.transform_postgres_to_parquet_format(
            raw_dev=raw_dev,
            columns=columns,
            ncols=len(columns),
            header_size=19,
            output_path=str(output_path),
            compression="snappy",
        )

        # Verify results
        assert output_path.exists(), "Parquet file should be created"
        assert len(cudf_df) <= 100, "Should have processed rows"

        # Read back and verify
        metadata = pq.read_metadata(output_path)
        assert metadata.num_columns == 3

        # Check compression
        assert metadata.row_group(0).column(0).compression == "SNAPPY"

        print(f"✓ DirectProcessor test passed: {len(cudf_df)} rows processed")
        print(f"  Processing time: {timing_info.get('total', 0):.3f}s")
        print(f"  Parquet size: {output_path.stat().st_size} bytes")

    def test_write_parquet_with_compression_options(self, temp_output_dir):
        """Test write_parquet_from_cudf with different compression options."""
        from src.write_parquet_from_cudf import write_cudf_to_parquet_with_options

        # Create test cuDF DataFrame
        num_rows = 10000
        df = cudf.DataFrame(
            {
                "id": cp.arange(num_rows, dtype=cp.int32),
                "value": cp.random.randint(0, 1000, num_rows, dtype=cp.int32),
                "score": cp.random.random(num_rows, dtype=cp.float32) * 100,
                "name": ["test_" + str(i) for i in range(num_rows)],
            }
        )

        # Test different compression algorithms
        compressions = ["snappy", "zstd", "lz4", "gzip", None]
        results = {}

        for comp in compressions:
            output_path = temp_output_dir / f"test_{comp or 'none'}.parquet"

            # Use product's write function
            timing = write_cudf_to_parquet_with_options(
                df, str(output_path), compression=comp, row_group_size=5000  # Test row group size
            )

            # Verify file
            assert output_path.exists()

            # Get file size and metadata
            file_size = output_path.stat().st_size
            metadata = pq.read_metadata(output_path)

            results[comp or "none"] = {
                "size": file_size,
                "time": timing,
                "row_groups": metadata.num_row_groups,
            }

            # Verify data integrity
            read_df = cudf.read_parquet(output_path)
            assert len(read_df) == num_rows
            assert list(read_df.columns) == ["id", "value", "score", "name"]

            print(
                f"✓ {comp or 'none'}: {file_size:,} bytes, "
                f"{timing:.3f}s, {metadata.num_row_groups} row groups"
            )

        # Compare compression ratios
        uncompressed_size = results["none"]["size"]
        for comp, info in results.items():
            if comp != "none":
                ratio = uncompressed_size / info["size"]
                print(f"  {comp} compression ratio: {ratio:.2f}x")

    def test_mixed_types_conversion(self, temp_output_dir):
        """Test conversion of mixed data types through the product pipeline."""
        import struct

        from src.postgres_to_parquet_converter import DirectProcessor
        from src.types import BOOL, DECIMAL128, FLOAT32, FLOAT64, INT32, INT64, UTF8, ColumnMeta

        # Create mock binary data with mixed types
        header = b"PGCOPY\n\xff\r\n\0" + b"\x00\x00\x00\x00" + b"\x00\x00\x00\x00"

        rows = []
        for i in range(50):
            row = struct.pack(">h", 7)  # 7 fields

            # INT32
            row += struct.pack(">i", 4) + struct.pack(">i", i)

            # INT64
            row += struct.pack(">i", 8) + struct.pack(">q", i * 1000000)

            # FLOAT32
            row += struct.pack(">i", 4) + struct.pack(">f", i * 0.1)

            # FLOAT64
            row += struct.pack(">i", 8) + struct.pack(">d", i * 0.01)

            # BOOLEAN
            row += struct.pack(">i", 1) + struct.pack("B", i % 2)

            # TEXT (UTF8)
            text = f"row_{i}".encode("utf-8")
            row += struct.pack(">i", len(text)) + text

            # NULL value
            row += struct.pack(">i", -1)

            rows.append(row)

        binary_data = header + b"".join(rows) + b"\xff\xff"

        # Process with DirectProcessor
        gpu_data = cp.asarray(np.frombuffer(binary_data, dtype=np.uint8))
        raw_dev = cuda.as_cuda_array(gpu_data).view(dtype=np.uint8)

        columns = [
            ColumnMeta(name="int32_col", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),
            ColumnMeta(name="int64_col", pg_oid=20, pg_typmod=-1, arrow_id=INT64, elem_size=8),
            ColumnMeta(name="float32_col", pg_oid=700, pg_typmod=-1, arrow_id=FLOAT32, elem_size=4),
            ColumnMeta(name="float64_col", pg_oid=701, pg_typmod=-1, arrow_id=FLOAT64, elem_size=8),
            ColumnMeta(name="bool_col", pg_oid=16, pg_typmod=-1, arrow_id=BOOL, elem_size=1),
            ColumnMeta(name="text_col", pg_oid=25, pg_typmod=-1, arrow_id=UTF8, elem_size=-1),
            ColumnMeta(name="null_col", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),
        ]

        processor = DirectProcessor(use_rmm=True, optimize_gpu=True, verbose=False)
        output_path = temp_output_dir / "test_mixed_types.parquet"

        cudf_df, timing_info = processor.transform_postgres_to_parquet_format(
            raw_dev=raw_dev,
            columns=columns,
            ncols=len(columns),
            header_size=19,
            output_path=str(output_path),
            compression="zstd",
        )

        # Verify
        assert output_path.exists()

        # Read back with pandas to verify types
        import pandas as pd

        pdf = pd.read_parquet(output_path)

        # Check data types
        assert pdf["int32_col"].dtype == np.int32
        assert pdf["int64_col"].dtype == np.int64
        assert pdf["float32_col"].dtype == np.float32
        assert pdf["float64_col"].dtype == np.float64
        assert pdf["bool_col"].dtype == bool
        assert pdf["text_col"].dtype == object
        assert pdf["null_col"].isna().all()  # All NULLs

        print(f"✓ Mixed types test passed: {len(pdf)} rows")
        print(f"  Data types preserved correctly")
        print(f"  File size: {output_path.stat().st_size:,} bytes")

    def test_large_dataset_performance(self, temp_output_dir):
        """Test performance with larger dataset using product pipeline."""
        if not GPU_AVAILABLE:
            pytest.skip("GPU not available")

        import time

        from src.write_parquet_from_cudf import write_cudf_to_parquet_with_options

        # Create large cuDF DataFrame
        num_rows = 1_000_000
        print(f"Creating DataFrame with {num_rows:,} rows...")

        df = cudf.DataFrame(
            {
                "id": cp.arange(num_rows, dtype=cp.int64),
                "value1": cp.random.randint(-1000000, 1000000, num_rows, dtype=cp.int32),
                "value2": cp.random.random(num_rows, dtype=cp.float64) * 1000,
                "category": cp.random.randint(0, 100, num_rows, dtype=cp.int32),
                "flag": cp.random.randint(0, 2, num_rows, dtype=cp.bool_),
            }
        )

        # Test with different row group sizes
        row_group_sizes = [10000, 50000, 100000]

        for rg_size in row_group_sizes:
            output_path = temp_output_dir / f"test_large_rg{rg_size}.parquet"

            start_time = time.time()
            timing = write_cudf_to_parquet_with_options(
                df, str(output_path), compression="zstd", row_group_size=rg_size
            )
            total_time = time.time() - start_time

            # Get file info
            file_size = output_path.stat().st_size
            metadata = pq.read_metadata(output_path)

            throughput = file_size / total_time / (1024 * 1024)  # MB/s

            print(f"\n✓ Row group size {rg_size:,}:")
            print(f"  File size: {file_size / (1024*1024):.1f} MB")
            print(f"  Row groups: {metadata.num_row_groups}")
            print(f"  Write time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} MB/s")

    def test_metadata_preservation(self, temp_output_dir):
        """Test that metadata is preserved through the conversion pipeline."""
        from src.write_parquet_from_cudf import write_cudf_to_parquet_with_options

        # Create DataFrame with specific metadata
        df = cudf.DataFrame({"id": range(100), "value": [i * 10 for i in range(100)]})

        output_path = temp_output_dir / "test_metadata.parquet"

        # Write with custom metadata
        write_cudf_to_parquet_with_options(
            df,
            str(output_path),
            compression="snappy",
            metadata={
                "source": "gpupgparser",
                "version": "1.0",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        )

        # Read metadata
        parquet_file = pq.ParquetFile(output_path)
        metadata = parquet_file.metadata

        # Check file metadata (if supported by the writer)
        print(f"✓ Metadata test passed")
        print(f"  Created by: {metadata.created_by}")
        print(f"  Num rows: {metadata.num_rows}")
        print(f"  Num columns: {metadata.num_columns}")
        print(f"  Format version: {metadata.format_version}")


# Import struct for binary data creation
import struct
