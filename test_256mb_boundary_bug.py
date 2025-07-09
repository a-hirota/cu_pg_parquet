#!/usr/bin/env python3
"""
256MB境界でのGPUスレッドスキップバグを再現するテスト
"""
import os
import sys
import numpy as np
import cupy as cp
import rmm
from numba import cuda
import struct

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cuda_kernels.postgres_binary_parser import parse_rows_and_fields_lite
from docs.benchmark.benchmark_rust_gpu_direct import setup_rmm_pool

def create_test_data_around_256mb():
    """256MB境界付近にテストデータを作成"""
    
    # PostgreSQL COPY BINARYヘッダー
    header = b'PGCOPY\n\xff\r\n\x00'  # 11 bytes
    header += struct.pack('>I', 0)      # flags: 4 bytes
    header += struct.pack('>I', 0)      # header extension: 4 bytes
    # 合計19バイト
    
    # 256MB境界
    MB_256 = 256 * 1024 * 1024
    
    # テスト用の行データ（簡単な構造）
    # 行長(4) + フィールド数(2) + フィールド1長(4) + データ(4) = 14バイト
    def create_row(value):
        row = struct.pack('>I', 14)      # 行長
        row += struct.pack('>H', 1)      # フィールド数
        row += struct.pack('>I', 4)      # フィールド長
        row += struct.pack('>I', value)  # データ
        return row
    
    # 256MB境界の前後にデータを配置
    data = bytearray()
    data.extend(header)
    
    # 境界前のパディング（256MB - ヘッダー - 少し余裕）
    padding_size = MB_256 - len(header) - 1000
    data.extend(b'\x00' * padding_size)
    
    # 境界付近に20行のテストデータを配置
    print(f"境界前の位置: {len(data):,} (0x{len(data):08X})")
    
    for i in range(20):
        row = create_row(1000 + i)
        data.extend(row)
        if len(data) > MB_256 - 100 and len(data) < MB_256 + 100:
            print(f"  行{i}: 位置 {len(data)-14:,} (0x{len(data)-14:08X})")
    
    # 境界後のパディング（合計400MBまで）
    final_size = 400 * 1024 * 1024
    if len(data) < final_size:
        data.extend(b'\x00' * (final_size - len(data)))
    
    # トレーラー
    data.extend(b'\xff\xff')
    
    return bytes(data)

def test_gpu_parsing():
    """GPUパーサーで256MB境界のデータを処理"""
    
    print("=== 256MB境界バグ再現テスト ===\n")
    
    # RMM初期化
    setup_rmm_pool()
    
    # テストデータ作成
    test_data = create_test_data_around_256mb()
    data_size = len(test_data)
    
    print(f"\nテストデータサイズ: {data_size:,} bytes")
    print(f"256MB境界: {256*1024*1024:,} (0x{256*1024*1024:08X})")
    
    # GPUにデータ転送（ファイル経由）
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(test_data)
        tmp_path = tmp.name
    
    # kvikioで読み込み
    import kvikio
    data_gpu = rmm.DeviceBuffer(size=data_size)
    with kvikio.CuFile(tmp_path, "r") as f:
        f.read(data_gpu)
    
    os.unlink(tmp_path)
    
    # GPUパーサー実行
    header_size = 19
    estimated_row_size = 192  # 意図的に大きめ（バグを再現しやすくする）
    num_fields = 1
    
    estimated_rows = (data_size - header_size) // estimated_row_size
    max_rows = int(estimated_rows * 1.5)
    
    print(f"\nGPUパーサー設定:")
    print(f"  推定行サイズ: {estimated_row_size}")
    print(f"  推定行数: {estimated_rows:,}")
    print(f"  最大行数: {max_rows:,}")
    
    # GPU配列確保
    row_info_size = 8 + 4 * num_fields + 4 * num_fields
    row_info_gpu = rmm.DeviceBuffer(size=max_rows * row_info_size)
    detected_rows_gpu = cp.zeros(1, dtype=cp.int32)
    
    # カーネル実行
    threads_per_block = 256
    blocks = (estimated_rows + threads_per_block - 1) // threads_per_block
    total_threads = blocks * threads_per_block
    
    print(f"  ブロック数: {blocks:,}")
    print(f"  総スレッド数: {total_threads:,}")
    
    # 256MB境界付近のスレッドを計算
    mb_256_pos = 256 * 1024 * 1024
    critical_thread = (mb_256_pos - header_size) // estimated_row_size
    print(f"\n256MB境界付近のスレッド:")
    print(f"  臨界スレッド: {critical_thread:,}")
    print(f"  臨界位置: 0x{header_size + critical_thread * estimated_row_size:08X}")
    
    # スレッドスキップが発生するか確認
    for i in range(-5, 6):
        tid = critical_thread + i
        if tid >= 0 and tid < total_threads:
            pos = header_size + tid * estimated_row_size
            print(f"  Thread {tid}: 0x{pos:08X} (ビット28: {(pos >> 28) & 1})")
    
    # 実際にGPUカーネルを実行
    parse_kernel = parse_rows_and_fields_lite.specialize(
        data_gpu,
        row_info_gpu,
        detected_rows_gpu,
        max_rows,
        header_size,
        data_size,
        estimated_row_size,
        num_fields
    )
    parse_kernel[blocks, threads_per_block]()
    cuda.synchronize()
    
    detected_rows = int(detected_rows_gpu[0])
    print(f"\n検出された行数: {detected_rows}")
    print(f"期待される行数: 20")
    print(f"欠落: {20 - detected_rows} 行")
    
    if detected_rows < 20:
        print("\n✅ バグを再現できました！")
    else:
        print("\n❌ バグを再現できませんでした")

if __name__ == "__main__":
    test_gpu_parsing()