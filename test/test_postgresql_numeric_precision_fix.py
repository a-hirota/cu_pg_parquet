#!/usr/bin/env python3
"""
PostgreSQL NUMERIC型 precision=0 修正テスト
==========================================

PostgreSQL NUMERIC型（精度指定なし）がprecision=0で来る問題の修正確認
"""

import os
import sys
import numpy as np
import psycopg
from numba import cuda

# パス設定
sys.path.append('/home/ubuntu/gpupgparser')

from src.meta_fetch import fetch_column_meta
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v2_decimal_column_wise import decode_chunk_decimal_column_wise

def create_test_table(conn):
    """テスト用テーブル作成"""
    with conn.cursor() as cur:
        # テストテーブル作成
        cur.execute("DROP TABLE IF EXISTS test_numeric_precision")
        cur.execute("""
            CREATE TABLE test_numeric_precision (
                id SERIAL PRIMARY KEY,
                num_default NUMERIC,              -- precision=0, scale=0 (可変精度)
                num_fixed NUMERIC(10,2),          -- precision=10, scale=2
                num_precision_only NUMERIC(15),   -- precision=15, scale=0
                description TEXT
            )
        """)
        
        # テストデータ挿入
        test_data = [
            (123.456, 789.12, 12345, 'Test 1'),
            (999.999, 111.11, 999999999999999, 'Test 2'),
            (-456.789, -222.22, -54321, 'Test 3'),
            (0, 0, 0, 'Zero test'),
            (None, None, None, 'NULL test')
        ]
        
        cur.executemany("""
            INSERT INTO test_numeric_precision 
            (num_default, num_fixed, num_precision_only, description) 
            VALUES (%s, %s, %s, %s)
        """, test_data)
        
        conn.commit()
        print("✓ テストテーブル作成・データ挿入完了")

def test_precision_fix():
    """precision=0修正のテスト"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    print("=== PostgreSQL NUMERIC precision=0 修正テスト ===")
    
    conn = psycopg.connect(dsn)
    try:
        # テストテーブル作成
        create_test_table(conn)
        
        # メタデータ取得・確認
        print("\n1. メタデータ確認中...")
        columns = fetch_column_meta(conn, "SELECT * FROM test_numeric_precision")
        
        print("   カラム情報:")
        for col in columns:
            if col.arrow_id == 5:  # DECIMAL128
                print(f"   - {col.name}: arrow_param={col.arrow_param}, arrow_id={col.arrow_id}")
        
        # COPY BINARY実行
        print("\n2. COPY BINARY実行中...")
        copy_sql = "COPY (SELECT * FROM test_numeric_precision) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
        
        print(f"   データサイズ: {len(raw_host)} バイト")
        
    finally:
        conn.close()
    
    # GPU処理
    print("\n3. GPU処理中...")
    raw_dev = cuda.to_device(raw_host)
    
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev,
        ncols=len(columns),
        header_size=header_size
    )
    
    rows = field_offsets_dev.shape[0]
    print(f"   行数: {rows}")
    
    # Column-wise最適化版テスト
    print("\n4. Column-wise最適化版でデコード中...")
    try:
        batch = decode_chunk_decimal_column_wise(
            raw_dev, field_offsets_dev, field_lengths_dev, columns, use_pass1_integration=True
        )
        print("✓ デコード成功")
        
        # 結果確認
        print("\n=== 結果確認 ===")
        print(f"行数: {batch.num_rows}, 列数: {batch.num_columns}")
        
        # Decimal列の型確認
        for col in columns:
            if col.arrow_id == 5:  # DECIMAL128
                arrow_col = batch.column(col.name)
                arrow_type = arrow_col.type
                print(f"{col.name}: Arrow型={arrow_type}")
                
                # サンプルデータ表示
                sample_data = arrow_col.to_pylist()[:3]
                print(f"  サンプルデータ: {sample_data}")
        
        return True
        
    except Exception as e:
        print(f"✗ デコード失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # CUDA初期化
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        sys.exit(1)
    
    success = test_precision_fix()
    
    if success:
        print("\n✓ PostgreSQL NUMERIC precision=0 修正テスト: 成功")
    else:
        print("\n✗ PostgreSQL NUMERIC precision=0 修正テスト: 失敗")
    
    sys.exit(0 if success else 1)