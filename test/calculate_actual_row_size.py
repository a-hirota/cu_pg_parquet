#!/usr/bin/env python3
"""
実際の行サイズを計算
"""

import os
import psycopg
from src.readPostgres.metadata import fetch_column_meta
from src.cuda_kernels.postgres_binary_parser import estimate_row_size_from_columns

def main():
    # メタデータ取得
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
    conn.close()
    
    # 推定行サイズ
    estimated = estimate_row_size_from_columns(columns)
    print(f"推定行サイズ: {estimated} バイト")
    
    # チャンクファイルから実際の行サイズを計算
    chunk_file = "/dev/shm/chunk_0.bin"
    file_size = os.path.getsize(chunk_file)
    
    # Rust側から報告された推定行数
    rust_estimated_rows = 15578741
    
    # 実際の平均行サイズ
    actual_avg = file_size / rust_estimated_rows
    print(f"実際の平均行サイズ: {actual_avg:.1f} バイト")
    
    # GPU処理された行数（Parquetファイルから）
    gpu_processed_rows = 9331177
    
    # GPUが実際に処理できた部分のサイズ
    gpu_processed_size = gpu_processed_rows * actual_avg
    print(f"\nGPU処理分析:")
    print(f"- Rust推定行数: {rust_estimated_rows:,}")
    print(f"- GPU処理行数: {gpu_processed_rows:,}")
    print(f"- 処理率: {gpu_processed_rows / rust_estimated_rows * 100:.1f}%")
    print(f"- GPU処理データサイズ: {gpu_processed_size / 1024**3:.2f} GB")
    
    # max_rows計算の再現
    min_row_size = max(40, estimated // 2)
    max_rows = int((file_size // min_row_size) * 1.1)
    print(f"\nmax_rows計算:")
    print(f"- min_row_size: {min_row_size}")
    print(f"- max_rows: {max_rows:,}")
    
    # メモリ要件
    ncols = len(columns)
    bytes_per_row = 8 + ncols * (8 + 4)
    total_memory_needed = max_rows * bytes_per_row
    print(f"\nメモリ要件:")
    print(f"- bytes_per_row: {bytes_per_row}")
    print(f"- 必要メモリ: {total_memory_needed / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()