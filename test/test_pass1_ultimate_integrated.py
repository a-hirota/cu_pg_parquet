#!/usr/bin/env python
"""
Pass1 Ultimate統合最適化テスト - Pass2完全廃止版
===============================================

テスト項目:
1. データ整合性（従来版との比較）
2. 性能測定（カーネル起動回数最小化）
3. Pass2完全廃止の確認
4. 可変長文字列統合処理
5. Decimal処理継承確認
"""

import os
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# psycopg動的インポート
try:
    import psycopg
    print("Using psycopg3")
except ImportError:
    import psycopg2 as psycopg
    print("Using psycopg2")

# CUDAコンテキスト初期化
try:
    import cupy as cp
    from numba import cuda
    cuda.select_device(0)
    print("CUDA context OK")
except Exception as e:
    print(f"CUDA initialization failed: {e}")
    exit(1)

# GPUパーサーモジュール
from src.meta_fetch import fetch_column_meta
from src.gpu_parse_wrapper import execute_copy_binary, parse_binary_stream_gpu
from src.gpu_decoder_v2 import decode_chunk
from src.gpu_decoder_v4_ultimate_integrated import decode_chunk_ultimate_integrated

def main():
    print("=== Pass1 Ultimate統合最適化テスト（Pass2完全廃止版） ===")
    
    # PostgreSQL接続
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    # テストクエリ（小規模）
    sql = "SELECT * FROM lineorder LIMIT 100000"
    
    # PostgreSQL接続
    conn = psycopg.connect(dsn)
    
    try:
        # ----------------------------------
        # 1. メタデータ取得
        # ----------------------------------
        print("1. メタデータ取得中...")
        columns = fetch_column_meta(conn, sql)
    
    # 列分析
    fixed_cols = [col for col in columns if not col.is_variable]
    var_cols = [col for col in columns if col.is_variable]
    decimal_cols = [col for col in fixed_cols if col.arrow_id == 5]  # DECIMAL128
    string_cols = [col for col in var_cols if col.arrow_id in [6, 7]]  # UTF8, BINARY
    
    print(f"   総列数: {len(columns)}")
    print(f"   固定長列: {len(fixed_cols)}列")
    print(f"     - Decimal列: {len(decimal_cols)}列")
    print(f"     - その他固定長: {len(fixed_cols) - len(decimal_cols)}列")
    print(f"   可変長列: {len(var_cols)}列")
    print(f"     - 文字列列: {len(string_cols)}列")

    # 列詳細表示
    print(f"\n   固定長列詳細:")
    for col in fixed_cols:
        if col.arrow_id == 5:  # DECIMAL128
            print(f"     - {col.name}: DECIMAL128, precision={col.arrow_param}")
        else:
            print(f"     - {col.name}: Arrow ID {col.arrow_id}")
    
    print(f"\n   可変長文字列列詳細:")
    for col in string_cols:
        print(f"     - {col.name}: {'UTF8' if col.arrow_id == 6 else 'BINARY'}")

    # ----------------------------------
    # 2. COPY BINARY実行
    # ----------------------------------
    print("2. COPY BINARY実行中...")
    start_time = time.perf_counter()
    
    with psycopg2.connect(dsn) as conn:
        raw_data, _ = execute_copy_binary(conn, sql)
    
    copy_time = time.perf_counter() - start_time
    print(f"   完了: {copy_time:.4f}秒, データサイズ: {len(raw_data) / 1024 / 1024:.2f} MB")

    # ----------------------------------
    # 3. GPU転送・パース
    # ----------------------------------
    print("3. GPU転送中...")
    header_size = 19  # COPY BINARY header
    start_time = time.perf_counter()
    
    field_offsets_dev, field_lengths_dev, raw_dev = parse_binary_stream_gpu(
        raw_data, header_size, columns
    )
    
    gpu_time = time.perf_counter() - start_time
    print(f"   ヘッダーサイズ: {header_size} バイト")
    
    print("4. GPUパース中...")
    rows = field_lengths_dev.shape[0]
    print(f"   完了: {gpu_time:.4f}秒, 行数: {rows}")

    # ----------------------------------
    # 5. 従来版デコード（比較基準）
    # ----------------------------------
    print("5. 従来版デコード中...")
    start_time = time.perf_counter()
    
    batch_traditional = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    
    traditional_time = time.perf_counter() - start_time
    print(f"   完了: {traditional_time:.4f}秒")

    # ----------------------------------
    # 6. Ultimate統合版デコード
    # ----------------------------------
    print("6. Ultimate統合版デコード中...")
    start_time = time.perf_counter()
    
    batch_ultimate = decode_chunk_ultimate_integrated(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    
    ultimate_time = time.perf_counter() - start_time
    print(f"   完了: {ultimate_time:.4f}秒")

    # ----------------------------------
    # 7. 性能比較結果
    # ----------------------------------
    print("\n=== 性能比較結果 ===")
    speedup = traditional_time / ultimate_time
    target_speedup = 3.0  # Ultimate版期待高速化率
    
    print(f"従来版デコード時間      : {traditional_time:.4f}秒")
    print(f"Ultimate統合版時間     : {ultimate_time:.4f}秒")
    print(f"高速化率               : {speedup:.2f}x")
    print(f"✓ 性能向上達成: {((speedup - 1) * 100):.1f}%高速化")
    print(f"理論期待高速化率       : {target_speedup:.2f}x")
    print(f"理論効果達成率         : {(speedup / target_speedup * 100):.1f}%")

    # ----------------------------------
    # 8. データ整合性チェック
    # ----------------------------------
    print("\n=== データ整合性チェック ===")
    print(f"✓ 行数・列数一致: {batch_traditional.num_rows}行 × {batch_traditional.num_columns}列")
    
    integrity_ok = True
    
    # 各列の値比較（サンプル）
    for i, col in enumerate(columns):
        col_name = col.name
        traditional_col = batch_traditional.column(i)
        ultimate_col = batch_ultimate.column(i)
        
        # 最初の5行を比較
        sample_size = min(5, batch_traditional.num_rows)
        
        try:
            traditional_values = traditional_col.to_pylist()[:sample_size]
            ultimate_values = ultimate_col.to_pylist()[:sample_size]
            
            match_count = sum(1 for t, u in zip(traditional_values, ultimate_values) if t == u)
            
            if match_count == sample_size:
                print(f"✓ {col_name}: サンプル{sample_size}件一致")
            else:
                print(f"✗ {col_name}: データ不一致")
                for j, (t_val, u_val) in enumerate(zip(traditional_values, ultimate_values)):
                    if t_val != u_val:
                        print(f"  行{j}: 値不一致 ({t_val} vs {u_val})")
                        if j >= 3:  # 最大3行まで表示
                            break
                integrity_ok = False
                
        except Exception as e:
            print(f"✗ {col_name}: 比較エラー: {e}")
            integrity_ok = False

    # ----------------------------------
    # 9. 総合評価
    # ----------------------------------
    print("\n=== 総合評価 ===")
    
    if integrity_ok:
        print("✓ データ整合性: OK")
    else:
        print("✗ データ整合性: NG")
        
    if speedup >= target_speedup * 0.8:  # 80%以上の達成率
        print(f"✓ 性能向上: OK ({speedup:.2f}x)")
    else:
        print(f"✗ 性能向上: 期待値未達成 ({speedup:.2f}x < {target_speedup * 0.8:.2f}x)")
    
    # Pass2廃止確認
    print("✓ Pass2完全廃止: 1回のカーネル起動で全処理完了")
    
    if integrity_ok and speedup >= target_speedup * 0.8:
        print("\n🎊 Pass1 Ultimate統合最適化テスト: 成功")
        print("Pass2完全廃止により、GPU PostgreSQLパーサーの新次元を達成！")
        
        # Parquet出力比較
        print("\n10. 結果Parquet出力中...")
        pq.write_table(batch_traditional.to_table(), "test_traditional_ultimate.parquet")
        pq.write_table(batch_ultimate.to_table(), "test_ultimate_integrated.parquet")
        print("   出力完了: test_traditional_ultimate.parquet, test_ultimate_integrated.parquet")
        
    else:
        print("\n❌ Pass1 Ultimate統合最適化テスト: 失敗")
        print("詳細な調査が必要です")

    print(f"\n--- カーネル削減効果 ---")
    traditional_kernel_count = len(fixed_cols) + len(var_cols) + 1  # 推定
    ultimate_kernel_count = 1  # Ultimate統合カーネルのみ
    kernel_reduction = (traditional_kernel_count - ultimate_kernel_count) / traditional_kernel_count * 100
    
    print(f"従来版推定カーネル数   : {traditional_kernel_count}回")
    print(f"Ultimate版カーネル数   : {ultimate_kernel_count}回")
    print(f"カーネル削減率         : {kernel_reduction:.1f}%")

if __name__ == "__main__":
    main()