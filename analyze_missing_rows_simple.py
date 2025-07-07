#!/usr/bin/env python3
"""
欠落行分析スクリプト（シンプル版）
===================================

最初のParquetファイルとPostgreSQLの最初の100万行を比較
"""

import pandas as pd
import psycopg2
import numpy as np
import os
import sys

def main():
    print("=== Parquetファイルのキーを抽出 ===")
    
    # 最初のParquetファイルを読み込み
    parquet_file = "output/chunk_0_queue.parquet"
    print(f"読み込み: {parquet_file}")
    
    df = pd.read_parquet(parquet_file, columns=['lo_orderkey', 'lo_linenumber'])
    
    # データ型を確認
    print(f"lo_orderkey dtype: {df['lo_orderkey'].dtype}")
    print(f"lo_linenumber dtype: {df['lo_linenumber'].dtype}")
    
    # Decimal型の場合の変換
    if df['lo_orderkey'].dtype == 'object':
        df['lo_orderkey'] = df['lo_orderkey'].astype(str).str.replace('.0', '', regex=False).astype(np.int64)
    if df['lo_linenumber'].dtype == 'object':
        df['lo_linenumber'] = df['lo_linenumber'].astype(str).str.replace('.0', '', regex=False).astype(np.int32)
    
    print(f"\nParquet行数: {len(df):,}")
    print("\n最初の5行:")
    print(df.head())
    print("\n最後の5行:")
    print(df.tail())
    
    # ユニークなOrderKeyの範囲を確認
    min_orderkey = df['lo_orderkey'].min()
    max_orderkey = df['lo_orderkey'].max()
    print(f"\nOrderKey範囲: {min_orderkey} - {max_orderkey}")
    
    # PostgreSQLからこの範囲のデータを取得
    print("\n=== PostgreSQLから同じ範囲のキーを取得 ===")
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("エラー: GPUPASER_PG_DSN環境変数が設定されていません")
        return 1
    
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        
        # 同じOrderKey範囲のデータを取得
        query = f"""
        SELECT lo_orderkey, lo_linenumber 
        FROM lineorder 
        WHERE lo_orderkey >= {min_orderkey} 
          AND lo_orderkey <= {max_orderkey}
        ORDER BY lo_orderkey, lo_linenumber
        """
        
        print(f"実行中: {query}")
        cur.execute(query)
        
        postgres_data = cur.fetchall()
        print(f"PostgreSQL行数: {len(postgres_data):,}")
        
        # DataFrameに変換
        pg_df = pd.DataFrame(postgres_data, columns=['lo_orderkey', 'lo_linenumber'])
        print("\nPostgreSQL最初の5行:")
        print(pg_df.head())
        
        # 差分を計算
        parquet_keys = set(zip(df['lo_orderkey'], df['lo_linenumber']))
        postgres_keys = set(zip(pg_df['lo_orderkey'], pg_df['lo_linenumber']))
        
        missing_keys = postgres_keys - parquet_keys
        extra_keys = parquet_keys - postgres_keys
        
        print(f"\n=== 差分分析 ===")
        print(f"Parquetキー数: {len(parquet_keys):,}")
        print(f"PostgreSQLキー数: {len(postgres_keys):,}")
        print(f"欠落キー数: {len(missing_keys):,}")
        print(f"余分なキー数: {len(extra_keys):,}")
        
        if missing_keys:
            print("\n欠落キーのサンプル（最初の10個）:")
            for i, key in enumerate(sorted(list(missing_keys)[:10])):
                print(f"  {key}")
        
        # ページ分布を調べる
        if missing_keys:
            print("\n=== ページ分布分析 ===")
            
            # 欠落キーのOrderKeyでページを推定
            missing_orderkeys = [k[0] for k in missing_keys]
            
            # OrderKeyごとのページを取得するクエリ
            placeholders = ','.join(['%s'] * min(100, len(missing_orderkeys)))
            query = f"""
            SELECT lo_orderkey, lo_linenumber,
                   (ctid::text::point)[0]::int as page_num
            FROM lineorder
            WHERE (lo_orderkey, lo_linenumber) IN (
                {','.join([f'({k[0]},{k[1]})' for k in list(missing_keys)[:100]])}
            )
            """
            
            cur.execute(query)
            page_results = cur.fetchall()
            
            if page_results:
                pages_df = pd.DataFrame(page_results, columns=['lo_orderkey', 'lo_linenumber', 'page_num'])
                page_counts = pages_df['page_num'].value_counts().sort_index()
                
                print("\n欠落行のページ分布（最初の20ページ）:")
                for page, count in page_counts.head(20).items():
                    print(f"  ページ{page}: {count}行")
                
                # チャンクごとの集計
                pages_df['chunk_id'] = pages_df['page_num'] // 288430
                chunk_counts = pages_df['chunk_id'].value_counts().sort_index()
                
                print("\nチャンクごとの欠落行数:")
                for chunk, count in chunk_counts.items():
                    print(f"  チャンク{chunk}: {count}行")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"エラー: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())