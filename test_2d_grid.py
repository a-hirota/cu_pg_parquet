#!/usr/bin/env python3
"""2次元グリッドテスト - 欠落thread_id問題の検証"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np

def test_2d_grid():
    """2次元グリッドでの実行テスト"""
    print("=== 2次元グリッドテスト開始 ===\n")
    
    # 既存のParquetファイルをバックアップ
    output_dir = "output"
    backup_dir = "output_1d_grid_backup"
    
    if os.path.exists(output_dir):
        print("既存のoutputディレクトリをバックアップ...")
        if os.path.exists(backup_dir):
            import shutil
            shutil.rmtree(backup_dir)
        os.rename(output_dir, backup_dir)
    
    # 2次元グリッドを強制する環境変数を設定
    env = os.environ.copy()
    env['GPUPGPARSER_FORCE_2D_GRID'] = '3451,5'  # (3451, 5)グリッドを強制
    env['GPUPGPARSER_DEBUG'] = '1'
    
    print("環境変数設定:")
    print(f"  GPUPGPARSER_FORCE_2D_GRID = 3451,5")
    print(f"  GPUPGPARSER_DEBUG = 1\n")
    
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
    
    print(f"コマンド実行: {' '.join(cmd)}")
    print("="*60 + "\n")
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("エラーが発生しました:")
        print(result.stderr)
        return
    
    print("\n実行完了！")
    
    # 生成されたParquetファイルを分析
    analyze_2d_grid_results()

def analyze_2d_grid_results():
    """2次元グリッドの結果を分析"""
    print("\n\n=== 2次元グリッド結果分析 ===\n")
    
    # Parquetファイルを読み込み
    try:
        chunk0 = pd.read_parquet("output/customer_chunk_0_queue.parquet")
        chunk1 = pd.read_parquet("output/customer_chunk_1_queue.parquet")
    except FileNotFoundError:
        print("Parquetファイルが見つかりません")
        return
    
    # 合計行数
    total_rows = len(chunk0) + len(chunk1)
    print(f"総行数: {total_rows:,} (期待値: 12,030,000)")
    print(f"差分: {12030000 - total_rows}")
    
    # thread_id分析
    if '_thread_id' in chunk0.columns:
        all_thread_ids = np.concatenate([
            chunk0['_thread_id'].values,
            chunk1['_thread_id'].values
        ])
        
        unique_threads = np.unique(all_thread_ids)
        print(f"\nユニークthread_id数: {len(unique_threads):,}")
        print(f"thread_id範囲: {unique_threads.min()} - {unique_threads.max()}")
        
        # 欠落thread_idを探す
        expected_threads = set(range(unique_threads.min(), unique_threads.max() + 1))
        actual_threads = set(unique_threads)
        missing_threads = expected_threads - actual_threads
        
        print(f"\n欠落thread_id数: {len(missing_threads)}")
        
        if missing_threads:
            missing_list = sorted(missing_threads)
            print("\n欠落thread_id:")
            for tid in missing_list[:20]:
                print(f"  {tid}")
            
            # 特定の欠落thread_idとの比較
            original_missing = [1048576, 1398102, 2097153, 2446679]
            print("\n1次元グリッドで欠落していたthread_id:")
            for tid in original_missing:
                if tid in missing_threads:
                    print(f"  {tid} - まだ欠落")
                else:
                    print(f"  {tid} - 解消！ ✓")
        else:
            print("\nthread_idの欠落なし！すべて正常 ✓")
    
    # c_custkeyの欠落も確認
    print("\n\n=== c_custkeyの欠落確認 ===")
    
    # Decimal型をintに変換
    chunk0['c_custkey'] = chunk0['c_custkey'].astype('int64')
    chunk1['c_custkey'] = chunk1['c_custkey'].astype('int64')
    
    all_keys = set(chunk0['c_custkey']) | set(chunk1['c_custkey'])
    missing_keys = []
    
    for i in range(1, 12030001):
        if i not in all_keys:
            missing_keys.append(i)
    
    print(f"欠落キー数: {len(missing_keys)} (1次元グリッド時: 14個)")
    
    if missing_keys:
        print("\n欠落キー:")
        for key in missing_keys[:20]:
            print(f"  {key}")

def compare_grid_dimensions():
    """1次元と2次元グリッドの比較"""
    print("\n\n=== グリッド次元の比較 ===\n")
    
    print("1次元グリッド (17255, 1):")
    print(f"  総ブロック数: 17255")
    print(f"  総スレッド数: 17255 × 256 = 4,417,280")
    print(f"  最大blockIdx.x: 17254")
    
    print("\n2次元グリッド (3451, 5):")
    print(f"  総ブロック数: 3451 × 5 = 17255")
    print(f"  総スレッド数: 17255 × 256 = 4,417,280")
    print(f"  最大blockIdx.x: 3450")
    print(f"  最大blockIdx.y: 4")
    
    # 欠落thread_idのブロック位置を計算
    print("\n欠落thread_idのブロック位置:")
    missing_tids = [1048576, 1398102, 2097153, 2446679]
    
    for tid in missing_tids:
        block_id = tid // 256
        thread_in_block = tid % 256
        
        # 1次元グリッド
        block_x_1d = block_id
        
        # 2次元グリッド
        block_x_2d = block_id // 5
        block_y_2d = block_id % 5
        
        print(f"\nthread_id {tid}:")
        print(f"  1次元: blockIdx.x = {block_x_1d}")
        print(f"  2次元: blockIdx.x = {block_x_2d}, blockIdx.y = {block_y_2d}")

if __name__ == "__main__":
    test_2d_grid()
    compare_grid_dimensions()