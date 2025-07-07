#!/usr/bin/env python3
"""
Parquetファイルの内容を確認
"""

import cudf
from pathlib import Path

def check_parquet_content():
    """Parquetファイルの最初の数行を表示"""
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    if not parquet_files:
        print("Parquetファイルが見つかりません")
        return
    
    print(f"Parquetファイル数: {len(parquet_files)}")
    
    # 最初のファイルを確認
    pf = parquet_files[0]
    print(f"\n=== {pf.name} ===")
    
    df = cudf.read_parquet(pf)
    print(f"行数: {len(df)}")
    print(f"列: {list(df.columns)}")
    
    if 'c_custkey' in df.columns:
        print(f"\nc_custkeyの型: {df['c_custkey'].dtype}")
        
        # 最初の10行
        print("\n最初の10行のc_custkey:")
        for i in range(min(10, len(df))):
            try:
                key = df['c_custkey'].iloc[i]
                name = df['c_name'].iloc[i] if 'c_name' in df.columns else 'N/A'
                print(f"  {i}: c_custkey={key}, c_name='{name}'")
            except Exception as e:
                print(f"  {i}: エラー - {e}")
        
        # 値の範囲を確認
        try:
            min_key = df['c_custkey'].min()
            max_key = df['c_custkey'].max()
            print(f"\nc_custkeyの範囲: {min_key} ～ {max_key}")
        except Exception as e:
            print(f"\n範囲計算エラー: {e}")
        
        # 535付近を検索
        print("\n535付近の値を検索:")
        for val in [534, 535, 536, 5350000]:
            try:
                if hasattr(df['c_custkey'].dtype, 'precision'):
                    # Decimal型の場合
                    matches = df[df['c_custkey'].astype('int64') == val]
                else:
                    matches = df[df['c_custkey'] == val]
                print(f"  c_custkey={val}: {len(matches)}件")
            except Exception as e:
                print(f"  c_custkey={val}: エラー - {e}")

if __name__ == "__main__":
    check_parquet_content()