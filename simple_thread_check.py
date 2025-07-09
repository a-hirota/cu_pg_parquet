#!/usr/bin/env python3
"""シンプルなthread_idギャップ確認"""

import pyarrow.parquet as pq
import numpy as np

# Parquetファイルのメタデータを確認
print("=== Parquetメタデータ確認 ===")
parquet_file = pq.ParquetFile("output/customer_chunk_0_queue.parquet")
print(f"列: {parquet_file.schema.names}")
print(f"行数: {parquet_file.metadata.num_rows}")

# thread_id列だけ読み込み
table = parquet_file.read(columns=['_thread_id'])
thread_ids = table.column('_thread_id').to_numpy()

print(f"\n=== thread_id分析 ===")
unique_threads = np.unique(thread_ids)
print(f"ユニークthread_id数: {len(unique_threads):,}")
print(f"thread_id範囲: {unique_threads[0]} - {unique_threads[-1]}")

# ギャップ確認
expected_threads = set(range(unique_threads[0], unique_threads[-1] + 1))
actual_threads = set(unique_threads)
missing_threads = expected_threads - actual_threads

print(f"\n欠落thread_id数: {len(missing_threads)}")
if missing_threads:
    missing_list = sorted(missing_threads)
    print(f"欠落thread_id（最初の20個）: {missing_list[:20]}")
    
    # 連続する欠落を探す
    gaps = []
    if missing_list:
        start = missing_list[0]
        end = missing_list[0]
        
        for i in range(1, len(missing_list)):
            if missing_list[i] == end + 1:
                end = missing_list[i]
            else:
                gaps.append((start, end))
                start = missing_list[i]
                end = missing_list[i]
        gaps.append((start, end))
        
        print(f"\n連続する欠落グループ: {len(gaps)}個")
        for start, end in gaps[:10]:
            if start == end:
                print(f"  Thread {start}")
            else:
                print(f"  Thread {start} - {end} ({end - start + 1}個)")
else:
    print("thread_idに欠落はありません")