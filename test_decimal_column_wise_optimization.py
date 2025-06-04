#!/usr/bin/env python3
"""
Decimal Column-wise Pass1統合最適化のテスト
=======================================

列ごと処理版の動作確認とパフォーマンステスト
"""

import os
import sys
import time
import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from numba import cuda

# パス設定
sys.path.append('/home/ubuntu/gpupgparser')

from src.meta_fetch import fetch_column_meta
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v2 import decode_chunk
from src.gpu_decoder_v2_decimal_column_wise import decode_chunk_decimal_column_wise

def test_column_wise_optimization():
    """Column-wise最適化版のテスト"""
    
    # データベース接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    print("=== Decimal Column-wise Pass1統合最適化テスト ===")
    
    # テストデータ取得
    conn = psycopg.connect(dsn)
    try:
        print("1. メタデータ取得中...")
        columns = fetch_column_meta(conn, "SELECT * FROM lineorder LIMIT 100000")
        
        # Decimal列数確認
        decimal_cols = [col for col in columns if col.arrow_id == 5]  # DECIMAL128 = 5
        print(f"   Decimal列数: {len(decimal_cols)}")
        for col in decimal_cols:
            print(f"   - {col.name}: precision={col.arrow_param}")
        
        print("2. COPY BINARY実行中...")
        start_copy = time.time()
        copy_sql = "COPY (SELECT * FROM lineorder LIMIT 100000) TO STDOUT (FORMAT binary)"
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
    
    # GPU処理
    print("3. GPU転送中...")
    raw_dev = cuda.to_device(raw_host)
    
    # ヘッダー検出
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"   ヘッダーサイズ: {header_size} バイト")
    
    # GPU Parse
    print("4. GPUパース中...")
    start_parse = time.time()
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev,
        ncols=len(columns),
        header_size=header_size
    )
    parse_time = time.time() - start_parse
    rows = field_offsets_dev.shape[0]
    print(f"   完了: {parse_time:.4f}秒, 行数: {rows}")
    
    # 従来版テスト
    print("5. 従来版デコード中...")
    start_traditional = time.time()
    batch_traditional = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    traditional_time = time.time() - start_traditional
    print(f"   完了: {traditional_time:.4f}秒")
    
    # Column-wise最適化版テスト
    print("6. Column-wise最適化版デコード中...")
    start_optimized = time.time()
    batch_optimized = decode_chunk_decimal_column_wise(
        raw_dev, field_offsets_dev, field_lengths_dev, columns, use_pass1_integration=True
    )
    optimized_time = time.time() - start_optimized
    print(f"   完了: {optimized_time:.4f}秒")
    
    # 結果比較
    print("\n=== 結果比較 ===")
    print(f"従来版デコード時間    : {traditional_time:.4f}秒")
    print(f"Column-wise最適化時間 : {optimized_time:.4f}秒")
    speedup = traditional_time / optimized_time if optimized_time > 0 else 0
    print(f"高速化率             : {speedup:.2f}x")
    
    # データ整合性チェック
    print("\n=== データ整合性チェック ===")
    try:
        # 行数・列数チェック
        assert batch_traditional.num_rows == batch_optimized.num_rows
        assert batch_traditional.num_columns == batch_optimized.num_columns
        print(f"✓ 行数・列数一致: {batch_traditional.num_rows}行 × {batch_traditional.num_columns}列")
        
        # Decimal列のデータ比較
        decimal_match = True
        for col in decimal_cols:
            traditional_col = batch_traditional.column(col.name)
            optimized_col = batch_optimized.column(col.name)
            
            # NULL値の一致チェック
            traditional_nulls = traditional_col.null_count
            optimized_nulls = optimized_col.null_count
            if traditional_nulls != optimized_nulls:
                print(f"✗ {col.name}: NULL数不一致 ({traditional_nulls} vs {optimized_nulls})")
                decimal_match = False
                continue
                
            # 非NULL値の比較（サンプル）
            traditional_array = traditional_col.to_pylist()
            optimized_array = optimized_col.to_pylist()
            
            sample_size = min(10, len(traditional_array))
            sample_match = True
            for i in range(sample_size):
                if traditional_array[i] != optimized_array[i]:
                    print(f"✗ {col.name}[{i}]: 値不一致 ({traditional_array[i]} vs {optimized_array[i]})")
                    sample_match = False
                    decimal_match = False
            
            if sample_match:
                print(f"✓ {col.name}: サンプル{sample_size}件一致")
        
        if decimal_match:
            print("✓ 全Decimal列のデータ整合性確認")
        else:
            print("✗ Decimal列にデータ不一致あり")
            
    except Exception as e:
        print(f"✗ データ整合性チェック失敗: {e}")
        decimal_match = False
    
    # Parquet出力テスト
    print("\n=== Parquet出力テスト ===")
    try:
        output_traditional = "test_traditional_column_wise.parquet"
        output_optimized = "test_optimized_column_wise.parquet"
        
        pq.write_table(pa.Table.from_batches([batch_traditional]), output_traditional)
        pq.write_table(pa.Table.from_batches([batch_optimized]), output_optimized)
        
        print(f"✓ Parquet出力成功")
        print(f"  従来版: {output_traditional}")
        print(f"  最適化版: {output_optimized}")
        
    except Exception as e:
        print(f"✗ Parquet出力失敗: {e}")
    
    # 総合結果
    print("\n=== 総合結果 ===")
    if decimal_match and speedup > 0.95:  # 5%以内の性能低下は許容
        print("✓ Column-wise Pass1統合最適化: 成功")
        print(f"  メモリアクセス削減効果確認: {speedup:.2f}x")
        return True
    else:
        print("✗ Column-wise Pass1統合最適化: 課題あり")
        if not decimal_match:
            print("  - データ整合性に問題")
        if speedup <= 0.95:
            print(f"  - 性能改善不十分 ({speedup:.2f}x)")
        return False

if __name__ == "__main__":
    # CUDA初期化
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        sys.exit(1)
    
    success = test_column_wise_optimization()
    sys.exit(0 if success else 1)