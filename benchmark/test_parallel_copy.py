#!/usr/bin/env python3
"""
PostgreSQL並列COPY速度テスト
並列性を確認するためのテストスクリプト
"""

import os
import time
import psycopg
import ray
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

TABLE_NAME = "lineorder"
LIMIT_ROWS = 10000000  # 1000万行

def test_single_copy():
    """単一COPY処理のテスト"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print("\n=== 単一COPY処理テスト ===")
    
    conn = psycopg.connect(dsn)
    start_time = time.time()
    total_size = 0
    
    try:
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {LIMIT_ROWS}) TO STDOUT (FORMAT binary)"
        
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_obj:
                for chunk in copy_obj:
                    total_size += len(chunk)
        
        elapsed = time.time() - start_time
        print(f"処理時間: {elapsed:.2f}秒")
        print(f"データサイズ: {total_size/(1024*1024):.2f}MB")
        print(f"スループット: {(total_size/(1024*1024))/elapsed:.2f} MB/sec")
        
    finally:
        conn.close()


def test_thread_parallel_copy(parallel_count=4):
    """スレッド並列COPY処理のテスト"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"\n=== スレッド並列COPY処理テスト（{parallel_count}並列） ===")
    
    def copy_worker(worker_id):
        """ワーカー関数"""
        conn = psycopg.connect(dsn)
        start_time = time.time()
        total_size = 0
        
        try:
            # ctid範囲を分割
            start_block = worker_id * 1000000
            end_block = (worker_id + 1) * 1000000
            
            copy_sql = f"""
            COPY (
                SELECT * FROM {TABLE_NAME}
                WHERE ctid >= '({start_block},1)'::tid
                  AND ctid < '({end_block},1)'::tid
                LIMIT {LIMIT_ROWS // parallel_count}
            ) TO STDOUT (FORMAT binary)
            """
            
            with conn.cursor() as cur:
                with cur.copy(copy_sql) as copy_obj:
                    for chunk in copy_obj:
                        total_size += len(chunk)
            
            elapsed = time.time() - start_time
            print(f"[Worker {worker_id}] 完了: {elapsed:.2f}秒, {total_size/(1024*1024):.2f}MB")
            return worker_id, elapsed, total_size
            
        finally:
            conn.close()
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=parallel_count) as executor:
        futures = [executor.submit(copy_worker, i) for i in range(parallel_count)]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    total_size = sum(r[2] for r in results)
    
    print(f"\n総処理時間: {total_time:.2f}秒")
    print(f"総データサイズ: {total_size/(1024*1024):.2f}MB")
    print(f"総スループット: {(total_size/(1024*1024))/total_time:.2f} MB/sec")


@ray.remote
class RayWorker:
    """Ray並列ワーカー"""
    def __init__(self, worker_id, dsn):
        self.worker_id = worker_id
        self.dsn = dsn
    
    def copy_data(self, table_name, start_block, end_block, limit_rows):
        """データコピー"""
        conn = psycopg.connect(self.dsn)
        start_time = time.time()
        total_size = 0
        
        try:
            copy_sql = f"""
            COPY (
                SELECT * FROM {table_name}
                WHERE ctid >= '({start_block},1)'::tid
                  AND ctid < '({end_block},1)'::tid
                LIMIT {limit_rows}
            ) TO STDOUT (FORMAT binary)
            """
            
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Worker {self.worker_id}: COPY開始")
            
            with conn.cursor() as cur:
                with cur.copy(copy_sql) as copy_obj:
                    for chunk in copy_obj:
                        total_size += len(chunk)
            
            elapsed = time.time() - start_time
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Worker {self.worker_id}: COPY完了 "
                  f"({elapsed:.2f}秒, {total_size/(1024*1024):.2f}MB)")
            
            return self.worker_id, elapsed, total_size
            
        finally:
            conn.close()


def test_ray_parallel_copy(parallel_count=4):
    """Ray並列COPY処理のテスト"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"\n=== Ray並列COPY処理テスト（{parallel_count}並列） ===")
    
    if not ray.is_initialized():
        ray.init(num_cpus=parallel_count * 2)
    
    start_time = time.time()
    
    # ワーカー作成
    workers = [RayWorker.remote(i, dsn) for i in range(parallel_count)]
    
    # 並列実行
    futures = []
    for i in range(parallel_count):
        start_block = i * 1000000
        end_block = (i + 1) * 1000000
        future = workers[i].copy_data.remote(
            TABLE_NAME, start_block, end_block, LIMIT_ROWS // parallel_count
        )
        futures.append(future)
    
    # 結果取得
    results = ray.get(futures)
    
    total_time = time.time() - start_time
    total_size = sum(r[2] for r in results)
    
    print(f"\n総処理時間: {total_time:.2f}秒")
    print(f"総データサイズ: {total_size/(1024*1024):.2f}MB")
    print(f"総スループット: {(total_size/(1024*1024))/total_time:.2f} MB/sec")
    
    ray.shutdown()


def main():
    """メイン関数"""
    print("PostgreSQL並列COPY速度テスト")
    print("=" * 50)
    
    # 単一処理
    test_single_copy()
    
    # スレッド並列（2, 4, 8）
    for parallel in [2, 4, 8]:
        test_thread_parallel_copy(parallel)
    
    # Ray並列（2, 4, 8）
    for parallel in [2, 4, 8]:
        test_ray_parallel_copy(parallel)


if __name__ == "__main__":
    main()