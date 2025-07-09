#!/usr/bin/env python3
"""欠落thread_idの一貫性を確認するテスト"""

import os
import sys
import subprocess
import pyarrow.parquet as pq
import numpy as np
import shutil

def run_test(test_number):
    """テストを実行してthread_idを収集"""
    print(f"\n=== テスト実行 #{test_number} ===")
    
    # outputディレクトリをクリア
    if os.path.exists("output"):
        shutil.rmtree("output")
    
    # cu_pg_parquet.pyを実行
    cmd = [
        sys.executable,
        "cu_pg_parquet.py",
        "--test",
        "--table", "customer",
        "--parallel", "8",
        "--chunks", "2",
        "--yes"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"エラーが発生しました: {result.stderr}")
        return None
    
    # thread_idを収集
    try:
        pf0 = pq.ParquetFile("output/customer_chunk_0_queue.parquet")
        table0 = pf0.read(columns=['_thread_id'])
        thread_ids0 = table0.column('_thread_id').to_numpy()
        
        pf1 = pq.ParquetFile("output/customer_chunk_1_queue.parquet")
        table1 = pf1.read(columns=['_thread_id'])
        thread_ids1 = table1.column('_thread_id').to_numpy()
        
        # 全thread_idを結合
        all_thread_ids = np.concatenate([thread_ids0, thread_ids1])
        unique_threads = np.unique(all_thread_ids)
        
        # 欠落thread_idを探す
        expected_threads = set(range(unique_threads.min(), unique_threads.max() + 1))
        actual_threads = set(unique_threads)
        missing_threads = expected_threads - actual_threads
        
        return sorted(missing_threads)
        
    except Exception as e:
        print(f"エラー: {e}")
        return None

def main():
    """複数回実行して欠落thread_idの一貫性を確認"""
    print("=== 欠落thread_idの一貫性テスト ===")
    print("同じコマンドを5回実行して、欠落thread_idが同じかどうか確認します。\n")
    
    # PostgreSQLが起動しているか確認
    result = subprocess.run(["pg_isready"], capture_output=True, text=True)
    if result.returncode != 0:
        print("PostgreSQLが起動していません。起動してください。")
        return
    
    # 5回実行
    all_results = []
    for i in range(5):
        missing = run_test(i + 1)
        if missing is not None:
            all_results.append(missing)
            print(f"欠落thread_id数: {len(missing)}")
            print(f"欠落thread_id: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    
    # 結果を分析
    print("\n\n=== 結果分析 ===")
    
    if not all_results:
        print("有効な結果が得られませんでした")
        return
    
    # すべての実行で同じ欠落thread_idか確認
    first_result = set(all_results[0])
    all_same = True
    
    for i, result in enumerate(all_results[1:], 2):
        if set(result) != first_result:
            all_same = False
            diff_missing = set(result) - first_result
            diff_found = first_result - set(result)
            print(f"\n実行#{i}で違いを検出:")
            if diff_missing:
                print(f"  新たに欠落: {sorted(diff_missing)}")
            if diff_found:
                print(f"  新たに発見: {sorted(diff_found)}")
    
    if all_same:
        print("\n✅ すべての実行で同じthread_idが欠落しています")
        print(f"欠落thread_id数: {len(all_results[0])}")
        print(f"欠落thread_id: {all_results[0]}")
    else:
        print("\n❌ 実行によって欠落thread_idが異なります")
    
    # 統計情報
    print("\n\n=== 統計情報 ===")
    
    # 各thread_idが何回欠落したか
    thread_count = {}
    for result in all_results:
        for tid in result:
            thread_count[tid] = thread_count.get(tid, 0) + 1
    
    # 常に欠落するthread_id
    always_missing = [tid for tid, count in thread_count.items() if count == len(all_results)]
    sometimes_missing = [tid for tid, count in thread_count.items() if count < len(all_results)]
    
    print(f"\n常に欠落するthread_id: {len(always_missing)}個")
    if always_missing:
        print(f"  {sorted(always_missing)}")
    
    if sometimes_missing:
        print(f"\n時々欠落するthread_id: {len(sometimes_missing)}個")
        for tid in sorted(sometimes_missing)[:10]:
            count = thread_count[tid]
            print(f"  thread_id {tid}: {count}/{len(all_results)}回欠落")

if __name__ == "__main__":
    main()