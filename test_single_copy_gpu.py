#!/usr/bin/env python3
"""
一括COPYしたデータをGPUで処理して行数を確認
"""
import os
import sys
import time
import rmm
import cupy as cp
import kvikio
from numba import cuda

# 環境設定
os.environ["GPUPASER_PG_DSN"] = "host=localhost dbname=postgres user=postgres"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cuda_kernels.postgres_binary_parser import parse_rows_and_fields_lite, detect_pg_header_size
from docs.benchmark.benchmark_rust_gpu_direct import setup_rmm_pool, get_postgresql_metadata

def main():
    # RMMプール初期化
    setup_rmm_pool()
    
    # メタデータ取得
    columns = get_postgresql_metadata("customer")
    print(f"✅ カラム数: {len(columns)}")
    
    # ファイル読み込み
    file_path = "/dev/shm/customer_single_copy.bin"
    print(f"\n📄 ファイル読み込み: {file_path}")
    
    start = time.time()
    file_size = os.path.getsize(file_path)
    print(f"ファイルサイズ: {file_size:,} bytes ({file_size/1024/1024/1024:.2f} GB)")
    
    # kvikioで読み込み
    data_gpu = rmm.DeviceBuffer(size=file_size)
    with kvikio.CuFile(file_path, "r") as f:
        f.read(data_gpu)
    read_time = time.time() - start
    print(f"kvikio読み込み時間: {read_time:.2f}秒")
    
    # ヘッダーサイズ検出
    header_info = cp.zeros(3, dtype=cp.int32)
    detect_pg_header_size_kernel = detect_pg_header_size.specialize(data_gpu, header_info)
    detect_pg_header_size_kernel[1, 1]()
    cuda.synchronize()
    header_size = int(header_info[0])
    num_fields = int(header_info[1])
    
    print(f"\n🔍 PostgreSQLヘッダー:")
    print(f"  - ヘッダーサイズ: {header_size} バイト")
    print(f"  - フィールド数: {num_fields}")
    
    # GPU解析
    print(f"\n🚀 GPU解析開始...")
    start = time.time()
    
    # 行数推定
    estimated_row_size = 141  # customerテーブルの平均行サイズ
    estimated_rows = (file_size - header_size) // estimated_row_size
    max_rows = int(estimated_rows * 1.5)  # 50%マージン
    
    print(f"推定行数: {estimated_rows:,}")
    print(f"バッファサイズ: {max_rows:,} 行")
    
    # GPU配列確保
    row_info_size = 8 + 4 * num_fields + 4 * num_fields  # row_pos + offsets + lengths
    row_info_gpu = rmm.DeviceBuffer(size=max_rows * row_info_size)
    detected_rows_gpu = cp.zeros(1, dtype=cp.int32)
    
    # カーネル実行
    threads_per_block = 256
    blocks = (estimated_rows + threads_per_block - 1) // threads_per_block
    
    parse_kernel = parse_rows_and_fields_lite.specialize(
        data_gpu,
        row_info_gpu,
        detected_rows_gpu,
        max_rows,
        header_size,
        file_size,
        estimated_row_size,
        num_fields
    )
    parse_kernel[blocks, threads_per_block]()
    cuda.synchronize()
    
    detected_rows = int(detected_rows_gpu[0])
    parse_time = time.time() - start
    
    print(f"\n✅ GPU解析完了:")
    print(f"  - 検出行数: {detected_rows:,} 行")
    print(f"  - 処理時間: {parse_time:.2f}秒")
    print(f"  - PostgreSQL期待値: 12,030,000 行")
    print(f"  - 差分: {12030000 - detected_rows:,} 行 ({(12030000 - detected_rows) / 12030000 * 100:.4f}%)")
    
    if detected_rows == 12030000:
        print("\n🎉 100%の精度を達成！")
    else:
        print(f"\n⚠️  {12030000 - detected_rows}行が欠落しています")

if __name__ == "__main__":
    main()