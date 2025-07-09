#!/usr/bin/env python3
"""デバッグモードで実行してthread_idを確認"""

import subprocess
import os
import pandas as pd

def run_with_debug():
    """デバッグモードで実行"""
    print("=== デバッグモードでの実行 ===\n")
    
    # 環境変数設定
    env = os.environ.copy()
    env.update({
        'RUST_LOG': 'info',
        'RUST_PARALLEL_CONNECTIONS': '16',
        'GPUPGPARSER_TEST_MODE': '1',  # テストモード有効
        'GPUPGPARSER_DEBUG': '1',      # デバッグモード有効
    })
    
    # 小さなテーブルで実行（supplier）
    print("1. supplierテーブルでテスト実行...")
    cmd = [
        'python', 'docs/benchmark/benchmark_rust_gpu_direct.py',
        '--table', 'supplier',
        '--chunks', '1',
        '--parallel', '1'
    ]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("エラー:")
        print(result.stderr)
        return False
    
    print("実行完了")
    
    # 出力ファイルを確認
    parquet_file = "output/supplier_chunk_0_queue.parquet"
    if os.path.exists(parquet_file):
        print(f"\n2. {parquet_file}を確認...")
        df = pd.read_parquet(parquet_file)
        
        print(f"列: {list(df.columns)}")
        print(f"行数: {len(df)}")
        
        if '_thread_id' in df.columns:
            print(f"\n_thread_id列の統計:")
            print(f"  ユニーク値: {df['_thread_id'].nunique()}")
            print(f"  値の分布:\n{df['_thread_id'].value_counts().head()}")
            
            # 0以外の値があるか
            non_zero = df[df['_thread_id'] != 0]
            if len(non_zero) > 0:
                print(f"\n_thread_id != 0 の行:")
                print(non_zero[['s_suppkey', '_thread_id', '_row_position']].head(10))
        else:
            print("\n⚠️ _thread_id列が存在しません")
    else:
        print(f"\n⚠️ {parquet_file}が存在しません")
    
    return True

if __name__ == "__main__":
    run_with_debug()