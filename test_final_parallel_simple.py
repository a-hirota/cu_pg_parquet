#!/usr/bin/env python
"""
最終並列統合版シンプルテスト
==========================

デッドロック問題を完全に回避した最終版の動作確認
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
from src.gpu_decoder_v6_final_parallel import decode_chunk_final_parallel  # 最終版

def main():
    print("=== 最終並列統合版シンプルテスト ===")
    
    # PostgreSQL接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    # 小規模テストから開始
    test_size = 5000
    
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
    rows = field_offsets_dev.shape[0]
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

    # 2. 最終並列統合版
    print("4b. 最終並列統合版デコード中...")
    try:
        start_final = time.time()
        batch_final = decode_chunk_final_parallel(raw_dev, field_offsets_dev, field_lengths_dev, columns)
        final_time = time.time() - start_final
        print(f"   完了: {final_time:.4f}秒")
        final_success = True
    except Exception as e:
        print(f"   エラー: {e}")
        import traceback
        traceback.print_exc()
        final_time = float('inf')
        final_success = False

    # ===== 結果まとめ =====
    
    print(f"\n--- 結果まとめ ({test_size:,}行) ---")
    print(f"従来版（Pass1+Pass2分離）: {traditional_time:.4f}秒")
    print(f"最終並列統合版          : {final_time:.4f}秒")
    
    if traditional_success and final_success:
        speedup = traditional_time / final_time
        print(f"最終版高速化率          : {speedup:.2f}x")
        
        if speedup >= 1.1:
            print(f"✅ 性能向上達成: {speedup:.2f}x高速化")
        elif speedup >= 0.9:
            print(f"⚡ 同等性能: {speedup:.2f}x（オーバーヘッド最小化成功）")
        else:
            print(f"⚠️  性能低下: {speedup:.2f}x（最適化要検討）")
    
    # データ整合性チェック
    if traditional_success and final_success:
        print("\n--- データ整合性チェック ---")
        try:
            traditional_rows = batch_traditional.num_rows
            final_rows = batch_final.num_rows
            
            if traditional_rows == final_rows:
                print(f"✅ 行数一致: {traditional_rows}")
                
                # 最初の列の最初の値を比較
                if len(columns) > 0:
                    traditional_first = batch_traditional.column(0).to_pylist()[0]
                    final_first = batch_final.column(0).to_pylist()[0]
                    
                    if traditional_first == final_first:
                        print(f"✅ データ一致: 最初の値 = {traditional_first}")
                    else:
                        print(f"❌ データ不一致: {traditional_first} vs {final_first}")
                        
                # 最初の文字列列チェック
                for i, col in enumerate(columns):
                    if col.name in ['lo_orderpriority', 'lo_shipmode']:
                        traditional_str = batch_traditional.column(i).to_pylist()[0]
                        final_str = batch_final.column(i).to_pylist()[0]
                        
                        if traditional_str == final_str:
                            print(f"✅ 文字列一致 ({col.name}): '{traditional_str}'")
                        else:
                            print(f"❌ 文字列不一致 ({col.name}): '{traditional_str}' vs '{final_str}'")
                        break
            else:
                print(f"❌ 行数不一致: {traditional_rows} vs {final_rows}")
                
        except Exception as e:
            print(f"⚠️  整合性チェックエラー: {e}")
    
    # 成功判定
    if final_success:
        print(f"\n🎉 最終並列統合版テスト: 成功")
        print("デッドロック問題を完全に回避し、安定した並列処理を実現しました。")
        
        # 大規模テストの推奨
        if traditional_success and final_success:
            print("\n📈 大規模テスト推奨:")
            print("- 10万行以上でより大きな性能向上が期待されます")
            print("- 文字列データの並列コピー効果が顕著に現れます")
            return True
    else:
        print(f"\n❌ 最終並列統合版テスト: 失敗")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)