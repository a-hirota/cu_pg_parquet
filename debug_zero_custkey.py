#!/usr/bin/env python3
"""
c_custkey=0の問題をデバッグ
"""

import cudf
from pathlib import Path

def debug_zero_custkey():
    """c_custkey=0の詳細を確認"""
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    for pf in parquet_files:
        df = cudf.read_parquet(pf)
        
        # c_custkey=0を検索
        if 'c_custkey' in df.columns:
            # Decimal型の場合の処理
            if hasattr(df['c_custkey'].dtype, 'precision'):
                zero_rows = df[df['c_custkey'].astype('int64') == 0]
            else:
                zero_rows = df[df['c_custkey'] == 0]
            
            if len(zero_rows) > 0:
                print(f"\n=== {pf.name}: c_custkey=0が{len(zero_rows)}件 ===")
                
                # 各行の詳細
                for i in range(len(zero_rows)):
                    print(f"\n行{i+1}:")
                    
                    # 全列の値を表示
                    for col in df.columns:
                        try:
                            value = zero_rows[col].iloc[i]
                            print(f"  {col}: {value}")
                        except Exception as e:
                            print(f"  {col}: エラー - {e}")
                    
                    # 周辺の行も確認
                    if '_row_position' in zero_rows.columns:
                        row_pos = zero_rows['_row_position'].iloc[i]
                        print(f"\n  周辺の行位置を確認:")
                        
                        # row_positionが近い行を探す
                        nearby = df[
                            (df['_row_position'] >= row_pos - 500) & 
                            (df['_row_position'] <= row_pos + 500)
                        ]
                        
                        print(f"  付近の{len(nearby)}行:")
                        for j in range(min(5, len(nearby))):
                            try:
                                pos = nearby['_row_position'].iloc[j]
                                key = nearby['c_custkey'].iloc[j]
                                name = nearby['c_name'].iloc[j] if 'c_name' in nearby.columns else 'N/A'
                                print(f"    pos={pos}, key={key}, name='{name}'")
                            except:
                                pass

if __name__ == "__main__":
    debug_zero_custkey()