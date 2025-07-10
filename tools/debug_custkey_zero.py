#!/usr/bin/env python3
"""c_custkey=0の問題をデバッグするスクリプト"""

import pandas as pd
import pyarrow.parquet as pq
import psycopg2
import os
from pathlib import Path

def check_zero_custkeys():
    """c_custkey=0のレコードを詳細に調査"""
    
    print("=== c_custkey=0 問題の詳細調査 ===\n")
    
    # Parquetファイルを検索
    parquet_files = list(Path('.').glob('customer_chunk_*_queue.parquet'))
    
    zero_records = []
    
    for pf in parquet_files:
        print(f"\n{pf}を調査中...")
        df = pd.read_parquet(pf)
        
        # c_custkey=0のレコードを抽出
        zero_df = df[df['c_custkey'] == 0].copy()
        
        if len(zero_df) > 0:
            print(f"  - {len(zero_df)}件のc_custkey=0レコードを発見")
            
            # thread_idとrow_positionも含めて表示
            if '_thread_id' in df.columns and '_row_position' in df.columns:
                for idx, row in zero_df.iterrows():
                    record = {
                        'file': pf.name,
                        'c_custkey': row['c_custkey'],
                        'c_name': row['c_name'],
                        'c_address': row['c_address'],
                        'thread_id': row.get('_thread_id', 'N/A'),
                        'row_position': row.get('_row_position', 'N/A')
                    }
                    zero_records.append(record)
                    
                    print(f"\n  レコード詳細:")
                    print(f"    - thread_id: {record['thread_id']}")
                    print(f"    - row_position: {record['row_position']}")
                    print(f"    - c_custkey: {record['c_custkey']}")
                    print(f"    - c_name: '{record['c_name']}'")
                    print(f"    - c_address: '{record['c_address']}'")
    
    # PostgreSQLで該当するレコードを検索
    if zero_records:
        print("\n\n=== PostgreSQLでの該当レコード検索 ===")
        
        # c_nameでの検索（c_addressの値が入っているため）
        names = list(set([r['c_name'] for r in zero_records]))
        
        try:
            conn = psycopg2.connect("host=localhost dbname=postgres user=postgres")
            cur = conn.cursor()
            
            # c_addressで検索（c_nameの値が誤って入っている可能性）
            addresses = "','".join(names)
            query = f"SELECT c_custkey, c_name, c_address FROM customer WHERE c_address IN ('{addresses}')"
            
            print(f"\nクエリ: {query}")
            cur.execute(query)
            
            results = cur.fetchall()
            print(f"\n{len(results)}件の一致レコードを発見:")
            for row in results:
                print(f"  - c_custkey: {row[0]}, c_name: '{row[1]}', c_address: '{row[2]}'")
            
            # c_custkey=0のレコードも確認
            cur.execute("SELECT COUNT(*) FROM customer WHERE c_custkey = 0")
            zero_count = cur.fetchone()[0]
            print(f"\nPostgreSQLでc_custkey=0のレコード数: {zero_count}")
            
            cur.close()
            conn.close()
            
        except Exception as e:
            print(f"PostgreSQL接続エラー: {e}")
    
    # まとめ
    print("\n\n=== まとめ ===")
    print(f"発見されたc_custkey=0のレコード総数: {len(zero_records)}")
    
    if zero_records:
        # row_positionでソート
        zero_records.sort(key=lambda x: x.get('row_position', 0))
        
        print("\nrow_position順:")
        for r in zero_records:
            print(f"  - pos: {r['row_position']:,}, thread: {r['thread_id']:,}, file: {r['file']}")

if __name__ == "__main__":
    check_zero_custkeys()