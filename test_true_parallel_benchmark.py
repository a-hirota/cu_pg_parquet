#!/usr/bin/env python
"""
真の並列処理版ベンチマークテスト
===============================

従来版 vs Ultimate統合版 vs 真の並列版の性能比較
"""

import time
import os
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
from src.gpu_decoder_v4_ultimate_integrated import decode_chunk_ultimate_integrated  # Ultimate版
from src.gpu_decoder_v5_true_parallel import decode_chunk_true_parallel  # 真の並列版

def main():
    print("=== 真の並列処理版ベンチマークテスト ===")
    
    # PostgreSQL接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    # テストサイズ（段階的に増加）
    test_sizes = [10000, 50000, 100000]
    
    for test_size in test_sizes:
        print(f"\n{'='*60}")
        print(f"テストサイズ: {test_size:,}行")
        print(f"{'='*60}")
        
        conn = psycopg.connect(dsn)
        
        try:
            print("1. メタデータ取得中...")
            columns = fetch_column_meta(conn, f"SELECT * FROM lineorder LIMIT {test_size}")
            
            print(f"   総列数: {len(columns)}")
            
            print("2. COPY BINARY実行中...")
            start_copy = time.time()
            copy_sql = f"COPY (SELECT * FROM lineorder LIMIT {test_size}) TO STDOUT (FORMAT binary)"
            buf = bytearray()
            with conn.cursor().copy(copy_sql) as cpy:
                while True:
                    chunk = cpy.read()
                    if not chunk:
                        break
                    buf.extend(chunk)
                raw_host = np.frombuffer(buf, dtype=np.uint8)
            copy_time = time.time() - start_copy
            print(f"   完了: {copy_time:.4f}秒, データサイズ: {len(raw_host) / (1024*1024):.2f} MB")
            
        finally:
            conn.close()
        
        print("3. GPU転送・パース中...")
        raw_dev = cuda.to_device(raw_host)
        header_size = detect_pg_header_size(raw_host[:128])
        
        field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
            raw_dev, len(columns), header_size=header_size
        )
        rows = field_lengths_dev.shape[0]
        print(f"   行数: {rows}")

        # ===== 性能比較テスト =====
        
        print("\n--- 性能比較テスト ---")
        
        # 1. 従来版（Pass1+Pass2分離）
        print("4a. 従来版デコード中...")
        try:
            start_traditional = time.time()
            batch_traditional = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
            traditional_time = time.time() - start_traditional
            print(f"   完了: {traditional_time:.4f}秒")
            traditional_success = True
        except Exception as e:
            print(f"   エラー: {e}")
            traditional_time = float('inf')
            traditional_success = False

        # 2. Ultimate統合版（シンプル版）
        print("4b. Ultimate統合版デコード中...")
        try:
            start_ultimate = time.time()
            batch_ultimate = decode_chunk_ultimate_integrated(raw_dev, field_offsets_dev, field_lengths_dev, columns)
            ultimate_time = time.time() - start_ultimate
            print(f"   完了: {ultimate_time:.4f}秒")
            ultimate_success = True
        except Exception as e:
            print(f"   エラー: {e}")
            ultimate_time = float('inf')
            ultimate_success = False

        # 3. 真の並列版
        print("4c. 真の並列版デコード中...")
        try:
            start_parallel = time.time()
            batch_parallel = decode_chunk_true_parallel(raw_dev, field_offsets_dev, field_lengths_dev, columns)
            parallel_time = time.time() - start_parallel
            print(f"   完了: {parallel_time:.4f}秒")
            parallel_success = True
        except Exception as e:
            print(f"   エラー: {e}")
            import traceback
            traceback.print_exc()
            parallel_time = float('inf')
            parallel_success = False

        # ===== 結果まとめ =====
        
        print(f"\n--- 結果まとめ ({test_size:,}行) ---")
        print(f"従来版（Pass1+Pass2分離）: {traditional_time:.4f}秒")
        print(f"Ultimate統合版          : {ultimate_time:.4f}秒")
        print(f"真の並列版              : {parallel_time:.4f}秒")
        
        if traditional_success and ultimate_success:
            ultimate_speedup = traditional_time / ultimate_time
            print(f"Ultimate統合版高速化率  : {ultimate_speedup:.2f}x")
        
        if traditional_success and parallel_success:
            parallel_speedup = traditional_time / parallel_time
            print(f"真の並列版高速化率      : {parallel_speedup:.2f}x")
            
        if ultimate_success and parallel_success:
            parallel_vs_ultimate = ultimate_time / parallel_time
            print(f"真の並列 vs Ultimate    : {parallel_vs_ultimate:.2f}x")
        
        # 期待値チェック
        expected_speedup = 2.0  # 期待高速化率
        
        if parallel_success and traditional_success:
            if parallel_speedup >= expected_speedup:
                print(f"✅ 真の並列版: 期待値達成 ({parallel_speedup:.2f}x >= {expected_speedup:.2f}x)")
            else:
                print(f"⚠️  真の並列版: 期待値未達成 ({parallel_speedup:.2f}x < {expected_speedup:.2f}x)")
        
        # データ整合性チェック（簡易）
        if traditional_success and parallel_success:
            print("\n--- データ整合性チェック ---")
            try:
                traditional_rows = batch_traditional.num_rows
                parallel_rows = batch_parallel.num_rows
                
                if traditional_rows == parallel_rows:
                    print(f"✅ 行数一致: {traditional_rows}")
                    
                    # 最初の列の最初の値を比較
                    if len(columns) > 0:
                        traditional_first = batch_traditional.column(0).to_pylist()[0]
                        parallel_first = batch_parallel.column(0).to_pylist()[0]
                        
                        if traditional_first == parallel_first:
                            print(f"✅ データ一致: 最初の値 = {traditional_first}")
                        else:
                            print(f"❌ データ不一致: {traditional_first} vs {parallel_first}")
                else:
                    print(f"❌ 行数不一致: {traditional_rows} vs {parallel_rows}")
                    
            except Exception as e:
                print(f"⚠️  整合性チェックエラー: {e}")
        
        print("-" * 60)
    
    print(f"\n{'='*60}")
    print("=== 真の並列処理版ベンチマーク完了 ===")
    print("予想される結果:")
    print("- 小規模データ: 従来版が有利（オーバーヘッド影響）")
    print("- 大規模データ: 真の並列版が大幅に高速化")
    print("- 文字列データの並列コピーによる劇的な性能向上期待")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()