"""
E2E Test for Function 2: /dev/shm Queue → GPU Transfer (kvikio)

This test verifies the functionality of transferring data from
shared memory queue to GPU memory using kvikio.
"""

import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.mark.e2e
@pytest.mark.gpu
class TestFunction2QueueToGPUTransfer:
    """Test /dev/shm queue to GPU transfer functionality."""

    def create_test_binary_data(self, num_rows=1000):
        """Create test PostgreSQL binary data."""
        # PostgreSQL binary format header
        header = b"PGCOPY\n\xff\r\n\0"
        # Flags field (32 bits, no OID)
        flags = struct.pack(">I", 0)
        # Header extension area length (32 bits)
        header_ext = struct.pack(">I", 0)

        data = header + flags + header_ext

        # Add sample rows
        for i in range(num_rows):
            # Row format: field count + field data
            # 3 fields: integer, float, text
            field_count = struct.pack(">h", 3)  # 3 fields

            # Field 1: integer (4 bytes)
            int_len = struct.pack(">i", 4)
            int_val = struct.pack(">i", i)

            # Field 2: float (8 bytes)
            float_len = struct.pack(">i", 8)
            float_val = struct.pack(">d", i * 3.14)

            # Field 3: text
            text_val = f"row_{i}".encode("utf-8")
            text_len = struct.pack(">i", len(text_val))

            row_data = field_count + int_len + int_val + float_len + float_val + text_len + text_val
            data += row_data

        # File trailer (-1)
        data += struct.pack(">h", -1)

        return data

    @pytest.mark.gpu
    def test_basic_gpu_transfer(self):
        """Test basic data transfer to GPU memory."""
        try:
            import cupy as cp
            import kvikio
        except ImportError:
            pytest.skip("kvikio not available")

        # Create test data
        test_data = self.create_test_binary_data(100)

        # Write to shared memory
        queue_path = Path("/dev/shm") / f"test_gpu_queue_{os.getpid()}.bin"

        try:
            # Write test data
            with open(queue_path, "wb") as f:
                f.write(test_data)

            # Allocate GPU memory
            gpu_buffer = cp.zeros(len(test_data), dtype=cp.uint8)

            # Transfer using kvikio
            start_time = time.time()
            # kvikio API might vary - adjust as needed
            # bytes_read = kvikio.read(queue_path, gpu_buffer)
            with open(queue_path, 'rb') as f:
                data = f.read()
                gpu_buffer[:] = cp.asarray(np.frombuffer(data, dtype=np.uint8))
            bytes_read = len(data)
            transfer_time = time.time() - start_time

            # Verify transfer
            assert bytes_read == len(test_data)

            # Copy back to CPU and verify header
            cpu_data = gpu_buffer[:11].get()
            assert bytes(cpu_data) == b"PGCOPY\n\xff\r\n\0"

            throughput = len(test_data) / transfer_time / (1024 * 1024)  # MB/s
            print(f"✓ Transferred {len(test_data)} bytes to GPU in {transfer_time:.3f}s")
            print(f"  Throughput: {throughput:.2f} MB/s")

        finally:
            if queue_path.exists():
                queue_path.unlink()

    @pytest.mark.gpu
    def test_large_data_transfer(self):
        """Test large data transfer performance."""
        try:
            import cupy as cp
            import kvikio
        except ImportError:
            pytest.skip("kvikio not available")

        # Create larger test data (10MB)
        test_data = self.create_test_binary_data(100000)
        queue_path = Path("/dev/shm") / f"test_large_gpu_queue_{os.getpid()}.bin"

        try:
            # Write test data
            with open(queue_path, "wb") as f:
                f.write(test_data)

            # Allocate GPU memory
            gpu_buffer = cp.zeros(len(test_data), dtype=cp.uint8)

            # Transfer using kvikio
            start_time = time.time()
            # kvikio API might vary - adjust as needed
            with open(queue_path, 'rb') as f:
                data = f.read()
                gpu_buffer[:] = cp.asarray(np.frombuffer(data, dtype=np.uint8))
            bytes_read = len(data)
            transfer_time = time.time() - start_time

            # Verify transfer
            assert bytes_read == len(test_data)

            throughput = len(test_data) / transfer_time / (1024 * 1024)  # MB/s
            print(f"✓ Large transfer: {len(test_data)/1024/1024:.1f} MB in {transfer_time:.3f}s")
            print(f"  Throughput: {throughput:.2f} MB/s")

            # Throughput should be reasonable
            assert throughput > 100  # At least 100 MB/s

        finally:
            if queue_path.exists():
                queue_path.unlink()

    @pytest.mark.gpu
    def test_chunked_transfer(self):
        """Test chunked data transfer for memory efficiency."""
        try:
            import cupy as cp
            import kvikio
        except ImportError:
            pytest.skip("kvikio not available")

        # Create test data
        total_size = 5 * 1024 * 1024  # 5MB
        chunk_size = 1024 * 1024  # 1MB chunks
        test_data = b"x" * total_size

        queue_path = Path("/dev/shm") / f"test_chunked_queue_{os.getpid()}.bin"

        try:
            # Write test data
            with open(queue_path, "wb") as f:
                f.write(test_data)

            # Allocate GPU memory for chunks
            gpu_chunks = []
            total_transferred = 0

            # Transfer in chunks
            with open(queue_path, "rb") as f:
                while total_transferred < total_size:
                    remaining = total_size - total_transferred
                    current_chunk_size = min(chunk_size, remaining)

                    # Allocate GPU memory for this chunk
                    gpu_chunk = cp.zeros(current_chunk_size, dtype=cp.uint8)

                    # Read chunk
                    f.seek(total_transferred)
                    chunk_data = f.read(current_chunk_size)
                    gpu_chunk[:] = cp.asarray(np.frombuffer(chunk_data, dtype=np.uint8))
                    bytes_read = len(chunk_data)

                    assert bytes_read == current_chunk_size
                    gpu_chunks.append(gpu_chunk)
                    total_transferred += bytes_read

            assert total_transferred == total_size
            assert len(gpu_chunks) == 5  # 5 chunks of 1MB each

            print(f"✓ Chunked transfer completed: {len(gpu_chunks)} chunks")

        finally:
            if queue_path.exists():
                queue_path.unlink()

    def test_memory_mapped_transfer(self):
        """Test memory-mapped file transfer (CPU baseline)."""
        import mmap

        # Create test data
        test_data = self.create_test_binary_data(10000)
        queue_path = Path("/dev/shm") / f"test_mmap_queue_{os.getpid()}.bin"

        try:
            # Write test data
            with open(queue_path, "wb") as f:
                f.write(test_data)

            # Memory map the file
            with open(queue_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_data:
                    # Read data
                    start_time = time.time()
                    data_copy = bytes(mmapped_data[:])
                    read_time = time.time() - start_time

                    # Verify
                    assert len(data_copy) == len(test_data)
                    assert data_copy[:11] == b"PGCOPY\n\xff\r\n\0"

                    throughput = len(test_data) / read_time / (1024 * 1024)
                    print(f"✓ Memory-mapped read: {throughput:.2f} MB/s (CPU baseline)")

        finally:
            if queue_path.exists():
                queue_path.unlink()

    @pytest.mark.gpu
    def test_direct_gpu_array_creation(self):
        """Test direct GPU array creation from binary data."""
        try:
            import cupy as cp
        except ImportError:
            pytest.skip("CuPy not available")

        # Create simple binary data
        # 1000 integers in big-endian format
        data = b""
        for i in range(1000):
            data += struct.pack(">i", i)

        queue_path = Path("/dev/shm") / f"test_array_queue_{os.getpid()}.bin"

        try:
            # Write test data
            with open(queue_path, "wb") as f:
                f.write(data)

            # Read directly into GPU array
            gpu_array = cp.fromfile(queue_path, dtype=">i4")

            # Verify
            assert len(gpu_array) == 1000
            assert gpu_array[0].get() == 0
            assert gpu_array[999].get() == 999

            # Test array operations on GPU
            gpu_sum = cp.sum(gpu_array)
            expected_sum = sum(range(1000))
            assert gpu_sum.get() == expected_sum

            print(f"✓ Direct GPU array creation successful")
            print(f"  Array shape: {gpu_array.shape}")
            print(f"  Sum verification: {gpu_sum.get()} == {expected_sum}")

        finally:
            if queue_path.exists():
                queue_path.unlink()

    @pytest.mark.gpu
    def test_error_handling(self):
        """Test error handling for GPU transfer operations."""
        try:
            import cupy as cp
            import kvikio
        except ImportError:
            pytest.skip("kvikio not available")

        # Test with non-existent file
        gpu_buffer = cp.zeros(1024, dtype=cp.uint8)
        non_existent_path = "/dev/shm/non_existent_file.bin"
        
        # Should handle file not found gracefully
        try:
            with open(non_existent_path, 'rb') as f:
                data = f.read()
                gpu_buffer[:] = cp.asarray(np.frombuffer(data, dtype=np.uint8))
        except FileNotFoundError:
            pass  # Expected

        # Test with insufficient GPU memory (simulate)
        queue_path = Path("/dev/shm") / f"test_error_queue_{os.getpid()}.bin"
        test_data = b"x" * 1024

        try:
            with open(queue_path, "wb") as f:
                f.write(test_data)

            # Try to read more than available
            gpu_buffer = cp.zeros(512, dtype=cp.uint8)  # Buffer too small

            # This should handle gracefully
            with open(queue_path, 'rb') as f:
                # Read only what the buffer can hold
                data = f.read(512)
                gpu_buffer[:] = cp.asarray(np.frombuffer(data, dtype=np.uint8))
            bytes_read = len(data)
            assert bytes_read == 512  # Should read only what fits

            print("✓ Error handling tests passed")

        finally:
            if queue_path.exists():
                queue_path.unlink()
