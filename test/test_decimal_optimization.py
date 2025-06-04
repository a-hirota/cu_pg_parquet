#!/usr/bin/env python3
"""
Decimal128最適化の動作確認テスト

PG-Stromアプローチによる高速化をテスト:
1. 列レベルでのスケール統一
2. precision≤18の場合のDecimal64最適化
3. 改良された128ビット演算ヘルパー
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import psycopg
import pyarrow as pa
from numba import cuda

from src.meta_fetch import fetch_column_meta
from src.gpu_decoder_v2 import decode_chunk
from src.cuda_kernels.pg_parser_kernels import pass0_pg_copy_parser

def test_decimal_optimization():
    """Decimal128最適化のテスト"""
    
    print("=" * 60)
    print("Decimal128最適化テスト開始")
    print("=" * 60)
    
    # PostgreSQLに接続
    try:
        conn = psycopg.connect(
            host="localhost",
            port=5432,
            database="postgres",
            user="postgres",
            password="postgres"
        )
        print("✓ PostgreSQL接続成功")
    except Exception as e:
        print(f"✗ PostgreSQL接続失敗: {e}")
        return False
    
    try:
        # テスト用テーブル作成 (異なる精度のDECIMAL列)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS test_decimal_optimization")
            cur.execute("""
                CREATE TABLE test_decimal_optimization (
                    id SERIAL PRIMARY KEY,
                    price_small NUMERIC(10,2),    -- precision≤18 → Decimal64最適化対象
                    price_medium NUMERIC(15,4),   -- precision≤18 → Decimal64最適化対象  
                    price_large NUMERIC(30,6),    -- precision>18 → 標準Decimal128
                    amount NUMERIC(38,0)          -- precision>18 → 標準Decimal128
                )
            """)
            
            # テストデータ挿入
            test_data = [
                (1, 123.45, 1234567.8901, 123456789012345678901234.567890, 12345678901234567890123456789012345678),
                (2, 678.90, 9876543.2109, 987654321098765432109876.543210, 98765432109876543210987654321098765432),
                (3, 0.01, 0.0001, 0.000001, 1),
                (4, -123.45, -1234567.8901, -123456789012345678901234.567890, -12345678901234567890123456789012345678),
                (5, None, None, None, None),  # NULL値テスト
            ]
            
            for data in test_data:
                cur.execute("""
                    INSERT INTO test_decimal_optimization 
                    (id, price_small, price_medium, price_large, amount) 
                    VALUES (%s, %s, %s, %s, %s)
                """, data)
            
            conn.commit()
            print("✓ テストデータ作成完了")
        
        # メタデータ取得
        query = "SELECT * FROM test_decimal_optimization ORDER BY id"
        columns = fetch_column_meta(conn, query)
        
        print(f"✓ カラムメタデータ取得: {len(columns)}列")
        for col in columns:
            if col.arrow_id == 5:  # DECIMAL128
                precision, scale = col.arrow_param or (38, 0)
                optimization = "Decimal64" if precision <= 18 else "Decimal128"
                print(f"  {col.name}: precision={precision}, scale={scale} → {optimization}最適化")
        
        # COPY BINARYでデータ取得
        with conn.cursor() as cur:
            cur.execute(f"COPY ({query}) TO STDOUT WITH (FORMAT BINARY)")
            raw_data = cur.fetchall()[0] if cur.fetchall() else b""
            
        if not raw_data:
            print("✗ COPYデータが空です")
            return False
            
        print(f"✓ COPYバイナリデータ取得: {len(raw_data)}バイト")
        
        # GPU処理
        try:
            # GPU初期化確認
            cuda.select_device(0)
            print("✓ CUDA初期化完了")
            
            # Pass 0: 行解析
            raw_gpu = cuda.to_device(np.frombuffer(raw_data, dtype=np.uint8))
            print(f"✓ データGPU転送完了: {raw_gpu.size}バイト")
            
            # 行数推定 (簡易)
            estimated_rows = len(test_data)
            ncols = len(columns)
            
            # Pass 0実行
            d_row_starts = cuda.device_array(estimated_rows + 1, dtype=np.int32)
            d_field_offsets = cuda.device_array((estimated_rows, ncols), dtype=np.int32)
            d_field_lengths = cuda.device_array((estimated_rows, ncols), dtype=np.int32)
            
            threads = 256
            blocks = (estimated_rows + threads - 1) // threads
            
            # 実際の行数を取得するためのカーネル実行
            pass0_pg_copy_parser[blocks, threads](
                raw_gpu,
                d_row_starts,
                d_field_offsets,
                d_field_lengths,
                ncols
            )
            cuda.synchronize()
            
            # 実際の行数確認
            row_starts_host = d_row_starts.copy_to_host()
            actual_rows = 0
            for i in range(estimated_rows):
                if row_starts_host[i] > 0:
                    actual_rows += 1
                else:
                    break
            
            print(f"✓ Pass 0完了: {actual_rows}行検出")
            
            if actual_rows == 0:
                print("✗ 有効な行が検出されませんでした")
                return False
            
            # 実際の行数でバッファをトリム
            actual_field_offsets = d_field_offsets[:actual_rows, :]
            actual_field_lengths = d_field_lengths[:actual_rows, :]
            
            # Pass 1-2: Arrow変換 (最適化されたDecimalカーネル使用)
            print("--- 最適化されたDecimal変換実行 ---")
            batch = decode_chunk(
                raw_gpu,
                actual_field_offsets,
                actual_field_lengths,
                columns
            )
            
            print(f"✓ Arrow RecordBatch作成完了: {batch.num_rows}行 x {batch.num_columns}列")
            
            # 結果検証
            print("\n--- 結果検証 ---")
            for i, col in enumerate(columns):
                arr = batch.column(i)
                print(f"{col.name} ({arr.type}): {arr.length}行")
                
                if col.arrow_id == 5:  # DECIMAL128
                    precision, scale = col.arrow_param or (38, 0)
                    optimization_type = "Decimal64" if precision <= 18 else "Decimal128"
                    print(f"  最適化タイプ: {optimization_type}")
                    
                    # 値の確認 (NULL以外)
                    valid_values = []
                    for j in range(arr.length):
                        if not arr.is_null(j).as_py():
                            val = arr[j].as_py()
                            valid_values.append(val)
                    
                    if valid_values:
                        print(f"  有効値: {valid_values[:3]}..." if len(valid_values) > 3 else f"  有効値: {valid_values}")
            
            print("\n✓ Decimal128最適化テスト成功!")
            return True
            
        except Exception as e:
            print(f"✗ GPU処理エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # クリーンアップ
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS test_decimal_optimization")
            conn.close()
            print("✓ クリーンアップ完了")
        except:
            pass

if __name__ == "__main__":
    success = test_decimal_optimization()
    sys.exit(0 if success else 1)
