#!/usr/bin/env python3
"""2次元グリッドで生成されたthread_idを確認"""

import pyarrow.parquet as pq
import numpy as np

def check_threads():
    """thread_idの欠落を確認"""
    print("=== 2次元グリッドのthread_id確認 ===\n")
    
    # Parquetファイルを読み込み（PyArrowを使用）
    try:
        pf0 = pq.ParquetFile("output/customer_chunk_0_queue.parquet")
        table0 = pf0.read(columns=['_thread_id'])
        thread_ids0 = table0.column('_thread_id').to_numpy()
        
        print(f"チャンク0: {len(thread_ids0):,}行")
        
        # thread_id統計
        unique_threads = np.unique(thread_ids0)
        print(f"ユニークthread_id数: {len(unique_threads):,}")
        print(f"thread_id範囲: {unique_threads[0]} - {unique_threads[-1]}")
        
        # 欠落thread_idを探す
        expected_threads = set(range(unique_threads[0], unique_threads[-1] + 1))
        actual_threads = set(unique_threads)
        missing_threads = expected_threads - actual_threads
        
        print(f"\n欠落thread_id数: {len(missing_threads)}")
        
        if missing_threads:
            missing_list = sorted(missing_threads)
            print("\n欠落thread_id:")
            for tid in missing_list[:20]:
                print(f"  {tid}")
            
            # 1次元グリッドで欠落していたthread_idとの比較
            original_missing = [1048576, 1398102, 2097153, 2446679]
            print("\n\n1次元グリッドで欠落していたthread_id:")
            for tid in original_missing:
                if tid in missing_threads:
                    print(f"  {tid} - ❌ まだ欠落")
                else:
                    print(f"  {tid} - ✓ 解消！")
            
            # 新たに欠落したthread_idがあるか
            new_missing = missing_threads - set(original_missing)
            if new_missing:
                print(f"\n\n⚠️ 新たに欠落したthread_id: {len(new_missing)}個")
                for tid in sorted(new_missing)[:10]:
                    print(f"  {tid}")
        else:
            print("\n✓ thread_idの欠落なし！すべて正常")
        
        # グリッド計算の確認
        print("\n\n=== グリッド計算の確認 ===")
        print(f"2次元グリッド: (3451, 5)")
        print(f"総ブロック数: 3451 × 5 = 17255")
        print(f"総スレッド数: 17255 × 256 = 4,417,280")
        
        # 欠落thread_idのブロック位置を計算
        if missing_threads:
            print("\n欠落thread_idのブロック位置:")
            for tid in sorted(missing_threads)[:5]:
                block_id = tid // 256
                thread_in_block = tid % 256
                
                # 2次元グリッドでの位置
                block_x = block_id // 5
                block_y = block_id % 5
                
                print(f"\nthread_id {tid}:")
                print(f"  ブロックID: {block_id}")
                print(f"  blockIdx.x = {block_x}, blockIdx.y = {block_y}")
                print(f"  threadIdx.x = {thread_in_block}")
                
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    check_threads()