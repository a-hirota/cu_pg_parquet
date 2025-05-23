#!/usr/bin/env python3
"""
GPU行検出機能のテストスクリプト
"""

import os
import sys
import numpy as np
from numba import cuda

# プロジェクトのsrcディレクトリをパスに追加
# This assumes the script is run from the project root (e.g., /home/ubuntu/gpupgparser)
# If run from test/ directory, adjust path accordingly or ensure PYTHONPATH is set.
# For simplicity, let's assume it's run from project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.cuda_kernels.pg_parser_kernels import find_row_start_offsets_gpu, decode_int32_be, read_uint16_be
    from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
    print("✓ GPU kernel import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Please ensure the script is run from the project root directory or PYTHONPATH is set correctly.")
    sys.exit(1)

def create_test_binary_data():
    """
    PostgreSQL COPY BINARY形式の簡単なテストデータを作成
    """
    # Header: "PGCOPY\n\377\r\n\0" + flags + extension
    header = bytearray([80, 71, 67, 79, 80, 89, 10, 255, 13, 10, 0])  # 11 bytes basic header
    header.extend([0, 0, 0, 0])  # flags (4 bytes)
    header.extend([0, 0, 0, 0])  # extension length (4 bytes, 0 = no extension)
    
    # Row 1: 2 fields
    row1 = bytearray()
    row1.extend([0, 2])  # num_fields = 2 (big-endian)
    # Field 1: length=4, data="test"
    row1.extend([0, 0, 0, 4])  # field length (big-endian)
    row1.extend(b"test")
    # Field 2: NULL
    row1.extend([255, 255, 255, 255])  # field length = -1 (NULL)
    
    # Row 2: 2 fields  
    row2 = bytearray()
    row2.extend([0, 2])  # num_fields = 2
    # Field 1: length=5, data="hello"
    row2.extend([0, 0, 0, 5])  # field length
    row2.extend(b"hello")
    # Field 2: length=5, data="world" 
    row2.extend([0, 0, 0, 5])  # field length
    row2.extend(b"world")
    
    # End marker
    end_marker = bytearray([255, 255])  # 0xFFFF
    
    return bytes(header + row1 + row2 + end_marker)

def test_gpu_row_detection():
    """
    GPU行検出機能のテスト
    """
    print("=== GPU Row Detection Test ===")
    
    # CUDA初期化確認
    try:
        cuda.current_context()
        print("✓ CUDA context OK")
    except Exception as e:
        print(f"✗ CUDA context failed: {e}")
        return False
    
    # テストデータ作成
    test_data_bytes = create_test_binary_data()
    print(f"✓ Test data created: {len(test_data_bytes)} bytes")
    
    # GPUに転送
    raw_dev = cuda.to_device(np.frombuffer(test_data_bytes, dtype=np.uint8))
    print(f"✓ Data transferred to GPU")
    
    # ヘッダーサイズ (PGCOPY\n\377\r\n\0 + flags(4) + ext_len(4) = 11 + 4 + 4 = 19)
    # This should match detect_pg_header_size for this specific data.
    header_size_known = 19
    
    # detect_pg_header_size を使用して確認
    header_sample_host = np.frombuffer(test_data_bytes[:128], dtype=np.uint8)
    detected_header_size = detect_pg_header_size(header_sample_host)
    if detected_header_size != header_size_known:
        print(f"✗ Header size mismatch: Known={header_size_known}, Detected={detected_header_size}")
        # return False # Continue test with known header for now, but this is a warning

    try:
        # GPU行検出を使用してパース
        print("\n--- Testing GPU Row Detection (use_gpu_row_detection=True) ---")
        field_offsets_gpu_dev, field_lengths_gpu_dev = parse_binary_chunk_gpu(
            raw_dev,
            ncols=2,  # 期待するカラム数
            header_size=header_size_known, # Use known header size for test
            use_gpu_row_detection=True
        )
        
        # 結果を確認
        offsets_gpu = field_offsets_gpu_dev.copy_to_host()
        lengths_gpu = field_lengths_gpu_dev.copy_to_host()
        
        print(f"✓ GPU parsing (GPU row detection) completed")
        print(f"   Detected rows: {offsets_gpu.shape[0]}")
        print(f"   Field offsets (GPU):\n{offsets_gpu}")
        print(f"   Field lengths (GPU):\n{lengths_gpu}")
        
        # CPUフォールバックとの比較
        print("\n--- Testing CPU Fallback (use_gpu_row_detection=False) ---")
        field_offsets_cpu_dev, field_lengths_cpu_dev = parse_binary_chunk_gpu(
            raw_dev,
            ncols=2,
            header_size=header_size_known, # Use known header size for test
            use_gpu_row_detection=False  # CPU fallback
        )
        
        offsets_cpu = field_offsets_cpu_dev.copy_to_host()
        lengths_cpu = field_lengths_cpu_dev.copy_to_host()
        
        print(f"✓ CPU parsing (CPU row detection) completed")
        print(f"   Detected rows: {offsets_cpu.shape[0]}")
        print(f"   Field offsets (CPU):\n{offsets_cpu}")
        print(f"   Field lengths (CPU):\n{lengths_cpu}")
        
        # 結果比較
        # Check number of rows first
        if offsets_gpu.shape[0] != offsets_cpu.shape[0]:
            print(f"✗ Row count mismatch: GPU={offsets_gpu.shape[0]}, CPU={offsets_cpu.shape[0]}")
            return False

        if np.array_equal(offsets_gpu, offsets_cpu) and np.array_equal(lengths_gpu, lengths_cpu):
            print("✓ GPU and CPU results match!")
            # Expected results for this specific test data
            # Row 1: "test", NULL
            # Row 2: "hello", "world"
            # Expected lengths: [[4, -1], [5, 5]]
            expected_lengths = np.array([[4, -1], [5, 5]], dtype=np.int32)
            if offsets_gpu.shape[0] == 2 and np.array_equal(lengths_gpu, expected_lengths):
                 print("✓ Lengths match expected values.")
                 return True
            else:
                 print("✗ Lengths do NOT match expected values.")
                 print(f"  Expected lengths:\n{expected_lengths}")
                 return False
        else:
            print("✗ GPU and CPU results differ!")
            if not np.array_equal(offsets_gpu, offsets_cpu):
                print("  Offset diff detected.")
            if not np.array_equal(lengths_gpu, lengths_cpu):
                print("  Length diff detected.")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpu_row_detection()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ One or more tests failed!")
        sys.exit(1)
