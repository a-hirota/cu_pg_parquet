"""累積和デバッグ用テスト"""

import os
import subprocess

# 環境変数設定（1チャンクのみ）
env = os.environ.copy()
env['CHUNK_ID'] = '0'
env['TOTAL_CHUNKS'] = '1'

print("Rustで1チャンクを生成中...")
process = subprocess.run(
    ['/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk'],
    capture_output=True,
    text=True,
    env=env,
    timeout=30
)

if process.returncode == 0:
    print("✅ Rustチャンク生成成功")
    
    # GPUテスト実行
    print("\nGPU処理テスト実行中...")
    import numpy as np
    from numba import cuda
    from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
    from src.metadata import fetch_column_meta
    from src.main_postgres_to_parquet import ZeroCopyProcessor
    import psycopg
    
    # メタデータ取得
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
    conn.close()
    
    # チャンクファイル読み込み
    with open("/dev/shm/chunk_0.bin", "rb") as f:
        data = f.read()
    raw_host = np.frombuffer(data, dtype=np.uint8)
    raw_dev = cuda.to_device(raw_host)
    
    # ヘッダーサイズ検出
    header_sample = raw_dev[:128].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    
    # GPUパース（最初の1000行のみ）
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
        raw_dev, columns, header_size=header_size
    )
    
    # 行数を制限
    rows = min(1000, field_offsets_dev.shape[0])
    field_offsets_dev = field_offsets_dev[:rows]
    field_lengths_dev = field_lengths_dev[:rows]
    
    print(f"処理行数: {rows}")
    
    # 文字列バッファ作成（デバッグ情報付き）
    processor = ZeroCopyProcessor()
    string_buffers = processor.create_string_buffers(
        columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
    )
    
else:
    print("❌ Rustチャンク生成失敗")
    print(process.stderr)