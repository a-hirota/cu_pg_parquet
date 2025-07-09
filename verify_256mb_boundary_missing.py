#!/usr/bin/env python3
"""
256MB境界での実際の欠落を確認
"""
import os
import subprocess
import pandas as pd

def check_customer_at_256mb():
    """customerテーブルの256MB境界付近のデータを確認"""
    
    print("=== 256MB境界での欠落確認 ===\n")
    
    # 1チャンクでcustomerテーブルをエクスポート
    print("1. customerテーブルを1チャンクでエクスポート...")
    env = os.environ.copy()
    env.update({
        "GPUPASER_PG_DSN": "host=localhost dbname=postgres user=postgres",
        "TABLE_NAME": "customer",
        "RUST_PARALLEL_CONNECTIONS": "16"
    })
    
    # テスト実行
    cmd = ["python", "docs/benchmark/benchmark_rust_gpu_direct.py", 
           "--table", "customer", "--chunks", "1", "--parallel", "16"]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # 結果から欠落行数を抽出
    if "欠落" in result.stdout:
        print("実行結果から欠落を確認:")
        for line in result.stdout.split('\n'):
            if "欠落" in line or "検出行数" in line:
                print(f"  {line}")
    
    print("\n2. Parquetファイルから256MB境界付近のデータを確認...")
    
    # Parquetファイルを読み込み
    parquet_file = "output/customer_chunk_0_queue.parquet"
    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
        print(f"  Parquetファイルの列: {list(df.columns)}")
        print(f"  行数: {len(df):,}")
        
        # 列名を確認（_thread_idまたは_row_positionがあるか）
        if '_thread_id' in df.columns:
            thread_col = '_thread_id'
        else:
            print("\n  ⚠️ _thread_id列がありません。_row_positionから推定します。")
            # _row_positionからthread_idを計算
            HEADER_SIZE = 19
            THREAD_STRIDE = 192
            df['estimated_thread'] = (df['_row_position'] - HEADER_SIZE) // THREAD_STRIDE
            thread_col = 'estimated_thread'
        
        # Thread 1398101付近のデータを確認
        thread_1398101_data = df[df[thread_col] == 1398101]
        print(f"\nThread 1398101のデータ: {len(thread_1398101_data)}行")
        
        if len(thread_1398101_data) > 0:
            print("最初の5行:")
            print(thread_1398101_data.head())
            print("\n最後の5行:")
            print(thread_1398101_data.tail())
        
        # Thread 3404032付近のデータを確認
        thread_3404032_data = df[df[thread_col] == 3404032]
        print(f"\nThread 3404032のデータ: {len(thread_3404032_data)}行")
        
        if len(thread_3404032_data) > 0:
            print("最初の5行:")
            print(thread_3404032_data.head())
        
        # 欠落キー3483451の前後を確認
        print("\n3. 欠落キー3483451の前後を確認...")
        
        # c_custkeyでソート
        df_sorted = df.sort_values('c_custkey')
        
        # 3483451の前後10行を表示
        target_key = 3483451
        idx = df_sorted[df_sorted['c_custkey'] >= target_key - 10].index[0]
        
        print(f"\nc_custkey {target_key}の前後:")
        subset = df_sorted.iloc[max(0, idx-5):idx+15]
        
        for idx, row in subset.iterrows():
            marker = ""
            if row['c_custkey'] == target_key - 1:
                marker = " ← 欠落の直前"
            elif row['c_custkey'] == target_key + 1:
                marker = " ← 欠落の直後"
            elif row['c_custkey'] == target_key:
                marker = " ← これは存在しないはず！"
                
            # DataFrameから直接取得
            thread_id = df.loc[idx, thread_col] if thread_col in df.columns else 0
            print(f"  c_custkey: {row['c_custkey']:8d}, Thread: {thread_id:7d}, "
                  f"Pos: 0x{row['_row_position']:08X}{marker}")
            
            # ギャップを確認
            if row['c_custkey'] == target_key - 1:
                prev_pos = row['_row_position']
            elif row['c_custkey'] == target_key + 1:
                next_pos = row['_row_position']
                gap = next_pos - prev_pos
                print(f"    → ポジションギャップ: {gap:,} bytes (0x{gap:08X})")

if __name__ == "__main__":
    check_customer_at_256mb()