#!/usr/bin/env python3
"""
一括COPYしたデータをGPUで処理して行数を確認（簡易版）
"""
import os
import sys
import time

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 環境変数設定（インポート前に設定）
os.environ["GPUPASER_PG_DSN"] = "host=localhost dbname=postgres user=postgres"

# 必要なモジュールをインポート
from docs.benchmark.benchmark_rust_gpu_direct import (
    setup_rmm_pool, 
    get_postgresql_metadata,
    gpu_consumer
)
import queue
import threading

def main():
    # RMMプール初期化
    setup_rmm_pool()
    
    # メタデータ取得
    columns = get_postgresql_metadata("customer")
    print(f"✅ カラム数: {len(columns)}")
    
    # 単一チャンクとして処理
    chunk_queue = queue.Queue()
    stats_queue = queue.Queue()
    shutdown_flag = threading.Event()
    
    # チャンク情報を作成
    file_size = os.path.getsize('/dev/shm/customer_single_copy.bin')
    chunk_info = {
        'chunk_id': 0,
        'chunk_file': '/dev/shm/customer_single_copy.bin',
        'file_size': file_size,
        'total_bytes': file_size,
        'elapsed_seconds': 0,
        'worker_count': 1,
        'workers': [{'id': 0, 'offset': 0, 'size': file_size, 'actual_size': file_size}],
        'columns': [{'name': col.name, 'data_type': str(col.arrow_id), 'pg_oid': col.pg_oid, 'arrow_type': str(col.arrow_id)} 
                   for col in columns]
    }
    
    # キューに追加
    chunk_queue.put(chunk_info)
    chunk_queue.put(None)  # 終了シグナル
    
    print("\n🚀 GPU処理開始...")
    start = time.time()
    
    # GPU処理を実行
    gpu_consumer(
        consumer_id=1,
        chunk_queue=chunk_queue,
        stats_queue=stats_queue,
        columns=columns,
        table_name="customer",
        total_chunks=1
    )
    
    # 統計情報を収集
    total_rows = 0
    while not stats_queue.empty():
        stat_type, value = stats_queue.get()
        if stat_type == 'rows':
            total_rows = value
    
    elapsed = time.time() - start
    
    print(f"\n✅ 処理完了:")
    print(f"  - 検出行数: {total_rows:,} 行")
    print(f"  - 処理時間: {elapsed:.2f}秒")
    print(f"  - PostgreSQL期待値: 12,030,000 行")
    
    if total_rows == 12030000:
        print("\n🎉 100%の精度を達成！")
    else:
        missing = 12030000 - total_rows
        print(f"\n⚠️  {missing:,}行が欠落しています ({missing/12030000*100:.4f}%)")

if __name__ == "__main__":
    main()