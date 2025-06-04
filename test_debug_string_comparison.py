#!/usr/bin/env python
"""
文字列データ破損デバッグテスト
=========================

従来のPass2処理と新しい並列処理の比較デバッグ
"""

import time
import os
import sys
import numpy as np

# psycopg動的インポート
try:
    import psycopg
    print("Using psycopg3")
except ImportError:
    import psycopg2 as psycopg
    print("Using psycopg2")

# CUDA/numba
from numba import cuda
cuda.select_device(0)
print("CUDA context OK")

# 既存モジュールのインポート
from src.meta_fetch import fetch_column_meta
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v2 import decode_chunk  # 従来版
from src.gpu_decoder_v6_final_parallel import decode_chunk_true_parallel  # 真の並列版

def main():
    print("=== 文字列データ破損デバッグテスト ===")
    
    # PostgreSQL接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    # 小規模テストデータ（デバッグのため）
    test_size = 10  # 非常に小さなサイズでデバッグ
    
    conn = psycopg.connect(dsn)
    
    try:
        print("1. メタデータ取得中...")
        columns = fetch_column_meta(conn, f"SELECT * FROM lineorder LIMIT {test_size}")
        
        print(f"   総列数: {len(columns)}")
        
        # 文字列列の確認
        string_columns = []
        for i, col in enumerate(columns):
            if col.name in ['lo_orderpriority', 'lo_shipmode']:
                string_columns.append((i, col.name))
                print(f"   文字列列発見: {i}番目 '{col.name}'")
        
        print("2. COPY BINARY実行中...")
        copy_sql = f"COPY (SELECT * FROM lineorder LIMIT {test_size}) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
        
        print(f"   データサイズ: {len(raw_host)} bytes")
        
    finally:
        conn.close()
    
    print("3. GPU転送・パース中...")
    raw_dev = cuda.to_device(raw_host)
    header_size = detect_pg_header_size(raw_host[:128])
    
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev, len(columns), header_size=header_size
    )
    rows = field_offsets_dev.shape[0]
    print(f"   行数: {rows}")

    # ===== デバッグ比較テスト =====
    
    print("\n--- デバッグ比較テスト ---")
    
    # 1. 従来版（Pass1+Pass2分離）
    print("4a. 従来版デコード中...")
    try:
        print("  CUDA printf出力 (従来版):")
        batch_traditional = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
        print("  従来版完了")
        traditional_success = True
    except Exception as e:
        print(f"   エラー: {e}")
        traditional_success = False

    print("\n" + "="*50)

    # 2. 真の並列統合版
    print("4b. 真の並列統合版デコード中...")
    try:
        print("  CUDA printf出力 (並列版):")
        batch_final = decode_chunk_true_parallel(raw_dev, field_offsets_dev, field_lengths_dev, columns)
        print("  並列版完了")
        final_success = True
    except Exception as e:
        print(f"   エラー: {e}")
        import traceback
        traceback.print_exc()
        final_success = False

    # ===== データ詳細比較 =====
    
    if traditional_success and final_success:
        print("\n--- データ詳細比較 ---")
        
        # 全文字列列を比較
        for col_idx, col_name in string_columns:
            print(f"\n列 {col_idx}: {col_name}")
            
            traditional_values = batch_traditional.column(col_idx).to_pylist()
            final_values = batch_final.column(col_idx).to_pylist()
            
            print(f"  従来版データ (最初の5行):")
            for i in range(min(5, len(traditional_values))):
                print(f"    [{i}]: '{traditional_values[i]}'")
            
            print(f"  並列版データ (最初の5行):")
            for i in range(min(5, len(final_values))):
                print(f"    [{i}]: '{final_values[i]}'")
            
            # 比較
            mismatches = 0
            for i in range(min(len(traditional_values), len(final_values))):
                if traditional_values[i] != final_values[i]:
                    mismatches += 1
                    if mismatches <= 3:  # 最初の3つの不一致を表示
                        print(f"  ❌ 不一致 行{i}: '{traditional_values[i]}' vs '{final_values[i]}'")
            
            if mismatches == 0:
                print(f"  ✅ 全て一致")
            else:
                print(f"  ❌ {mismatches}個の不一致")
    
    return traditional_success and final_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
