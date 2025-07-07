#!/usr/bin/env python3
"""
customerテーブルのParquetファイルをcuDFで分析
"""

import cudf
from pathlib import Path

def analyze_customer_with_cudf():
    """cuDFを使ってcustomerテーブルを分析"""
    
    output_dir = Path("/home/ubuntu/gpupgparser/output")
    parquet_file = output_dir / "chunk_0_queue.parquet"
    
    if not parquet_file.exists():
        print(f"ファイルが見つかりません: {parquet_file}")
        return
    
    print(f"=== {parquet_file.name} の分析 (cuDF) ===\n")
    
    try:
        # cuDFで読み込み
        gdf = cudf.read_parquet(parquet_file)
        
        print(f"列情報:")
        print(gdf.dtypes)
        
        print(f"\n基本統計:")
        print(f"  総行数: {len(gdf):,}")
        print(f"  列数: {len(gdf.columns)}")
        
        # c_custkeyの範囲を確認
        if 'c_custkey' in gdf.columns:
            print(f"\nc_custkeyの統計:")
            print(f"  最小値: {gdf['c_custkey'].min()}")
            print(f"  最大値: {gdf['c_custkey'].max()}")
            print(f"  ユニーク数: {gdf['c_custkey'].nunique():,}")
            
            # 重複チェック
            duplicates = gdf['c_custkey'].duplicated().sum()
            print(f"  重複数: {duplicates}")
        
        # 最初の5行を表示
        print(f"\n最初の5行:")
        print(gdf.head().to_pandas())
        
        # 期待値との比較
        expected_rows = 6_000_000
        coverage = len(gdf) / expected_rows * 100
        print(f"\n期待値との比較:")
        print(f"  期待行数: {expected_rows:,}")
        print(f"  実際の行数: {len(gdf):,}")
        print(f"  カバー率: {coverage:.2f}%")
        print(f"  不足行数: {expected_rows - len(gdf):,}")
        
    except Exception as e:
        print(f"エラー: {e}")
        print("\n生のParquetメタデータを確認...")
        
        # pyarrowで基本情報だけ取得
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(parquet_file)
        print(f"行数: {pf.metadata.num_rows:,}")
        print(f"列数: {len(pf.schema)}")
        print(f"スキーマ名: {[field.name for field in pf.schema]}")

def main():
    analyze_customer_with_cudf()

if __name__ == "__main__":
    main()