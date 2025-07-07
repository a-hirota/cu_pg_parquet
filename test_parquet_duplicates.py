#!/usr/bin/env python3
"""
Parquetファイルの重複データを詳細表示
"""

import cudf
import sys
from pathlib import Path

def test_duplicates():
    # Parquetファイルを読み込み
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    if not parquet_files:
        print("Parquetファイルが見つかりません")
        return
    
    print(f"ファイル数: {len(parquet_files)}")
    
    for pf in parquet_files[:1]:  # 最初のファイルのみ
        print(f"\n=== {pf.name} ===")
        
        df = cudf.read_parquet(pf)
        print(f"行数: {len(df)}")
        print(f"列: {list(df.columns)}")
        
        # c_custkeyでグループ化して重複を見つける
        if 'c_custkey' in df.columns:
            # Decimal型の場合はint64に変換
            if hasattr(df['c_custkey'].dtype, 'precision'):
                key_series = df['c_custkey'].astype('int64')
            else:
                key_series = df['c_custkey']
            
            # 重複をカウント
            key_counts = key_series.value_counts()
            duplicates = key_counts[key_counts > 1].head(5)
            
            print(f"\n重複キー（上位5個）:")
            for key, count in duplicates.to_pandas().items():
                print(f"\nc_custkey = {key}: {count}回出現")
                
                # この重複キーのデータを取得
                if hasattr(df['c_custkey'].dtype, 'precision'):
                    mask = df['c_custkey'].astype('int64') == int(key)
                else:
                    mask = df['c_custkey'] == key
                
                dup_rows = df[mask]
                
                # 各行を表示
                for i in range(len(dup_rows)):
                    print(f"\n  行{i+1}:")
                    
                    # 各列の値を表示
                    for col in df.columns:
                        if not col.startswith('_'):  # デバッグ列を除外
                            try:
                                value = dup_rows[col].iloc[i]
                                
                                # 文字列の場合
                                if isinstance(value, str):
                                    # 末尾の空白を可視化
                                    visible = value.replace(' ', '·')
                                    print(f"    {col}: '{value}' (見える形: '{visible}')")
                                else:
                                    print(f"    {col}: {value}")
                            except Exception as e:
                                print(f"    {col}: エラー - {e}")
                    
                    # デバッグ情報
                    if '_thread_id' in dup_rows.columns:
                        try:
                            tid = dup_rows['_thread_id'].iloc[i]
                            rpos = dup_rows['_row_position'].iloc[i] if '_row_position' in dup_rows.columns else None
                            print(f"    ---")
                            print(f"    thread_id: {tid}")
                            if rpos is not None:
                                print(f"    row_position: {rpos:,}")
                        except:
                            pass


if __name__ == "__main__":
    test_duplicates()