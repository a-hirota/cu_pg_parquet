#!/usr/bin/env python3
"""
Pass1完全統合版テスト
=====================

1回のカーネル起動で全固定長列を処理する革新的最適化のテスト
期待効果: 5-15倍の性能向上
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
from src.gpu_decoder_v3_fully_integrated import decode_chunk_fully_integrated

def test_fully_integrated_optimization():
    """Pass1完全統合版のテスト"""
    
    # データベース接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    print("=== Pass1完全統合版最適化テスト ===")
    
    # テストデータ取得
    conn = psycopg.connect(dsn)
    try:
        print("1. メタデータ取得中...")
        columns = fetch_column_meta(conn, "SELECT * FROM lineorder LIMIT 100000")
        
        # 列分析
        fixed_columns = [col for col in columns if not col.is_variable]
        variable_columns = [col for col in columns if col.is_variable]
        decimal_columns = [col for col in fixed_columns if col.arrow_id == 5]  # DECIMAL128 = 5
        
        print(f"   総列数: {len(columns)}")
        print(f"   固定長列: {len(fixed_columns)}列")
        print(f"     - Decimal列: {len(decimal_columns)}列")
        print(f"     - その他固定長: {len(fixed_columns) - len(decimal_columns)}列")
        print(f"   可変長列: {len(variable_columns)}列")
        
        print("\n   固定長列詳細:")
        for col in fixed_columns:
            if col.arrow_id == 5:  # DECIMAL128
                print(f"     - {col.name}: DECIMAL128, precision={col.arrow_param}")
            else:
                print(f"     - {col.name}: Arrow ID {col.arrow_id}")
        
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
    
    # Pass1完全統合版テスト
    print("6. Pass1完全統合版デコード中...")
    start_integrated = time.time()
    batch_integrated = decode_chunk_fully_integrated(
        raw_dev, field_offsets_dev, field_lengths_dev, columns
    )
    integrated_time = time.time() - start_integrated
    print(f"   完了: {integrated_time:.4f}秒")
    
    # デバッグ: field_offsetsとfield_lengthsの値を直接確認
    print("\n=== デバッグ: field_offsets/lengths確認 ===")
    field_offsets_host = field_offsets_dev.copy_to_host()
    field_lengths_host = field_lengths_dev.copy_to_host()
    
    # int32列のインデックスを特定
    int32_columns = [(i, col) for i, col in enumerate(columns) if not col.is_variable and col.arrow_id == 1]
    print(f"int32列: {[(col.name, idx) for idx, col in int32_columns]}")
    
    # 最初の行のint32列の値を確認
    for idx, col in int32_columns[:3]:  # 最初の3つのint32列
        offset = field_offsets_host[0, idx]
        length = field_lengths_host[0, idx]
        print(f"{col.name}: offset={offset}, length={length}")
        
        # raw_dataから直接読み取りテスト
        if offset > 0 and length == 4 and offset + 4 <= len(raw_host):
            raw_bytes = raw_host[offset:offset+4]
            # ビッグエンディアンで読み取り
            value = (raw_bytes[0] << 24) | (raw_bytes[1] << 16) | (raw_bytes[2] << 8) | raw_bytes[3]
            print(f"  直接読み取り値: {value}")
            print(f"  生バイト: {[f'{b:02x}' for b in raw_bytes]}")
        else:
            print(f"  読み取り不可: offset={offset}, length={length}")
    
    # 結果比較
    print("\n=== 性能比較結果 ===")
    print(f"従来版デコード時間      : {traditional_time:.4f}秒")
    print(f"Pass1完全統合版時間     : {integrated_time:.4f}秒")
    speedup = traditional_time / integrated_time if integrated_time > 0 else 0
    print(f"高速化率               : {speedup:.2f}x")
    
    if speedup > 1:
        print(f"✓ 性能向上達成: {((speedup - 1) * 100):.1f}%高速化")
    else:
        print(f"✗ 性能低下: {((1 - speedup) * 100):.1f}%低速化")
    
    # 理論効果との比較
    theoretical_speedup = 1 + (len(fixed_columns) / len(columns)) * 2  # 固定長列比率×係数
    print(f"理論期待高速化率       : {theoretical_speedup:.2f}x")
    achievement_rate = speedup / theoretical_speedup * 100 if theoretical_speedup > 0 else 0
    print(f"理論効果達成率         : {achievement_rate:.1f}%")
    
    # データ整合性チェック
    print("\n=== データ整合性チェック ===")
    try:
        # 行数・列数チェック
        assert batch_traditional.num_rows == batch_integrated.num_rows
        assert batch_traditional.num_columns == batch_integrated.num_columns
        print(f"✓ 行数・列数一致: {batch_traditional.num_rows}行 × {batch_traditional.num_columns}列")
        
        # 固定長列のデータ比較
        fixed_match = True
        for col in fixed_columns:
            traditional_col = batch_traditional.column(col.name)
            integrated_col = batch_integrated.column(col.name)
            
            # NULL値の一致チェック
            traditional_nulls = traditional_col.null_count
            integrated_nulls = integrated_col.null_count
            if traditional_nulls != integrated_nulls:
                print(f"✗ {col.name}: NULL数不一致 ({traditional_nulls} vs {integrated_nulls})")
                fixed_match = False
                continue
                
            # 非NULL値の比較（サンプル）
            traditional_array = traditional_col.to_pylist()
            integrated_array = integrated_col.to_pylist()
            
            sample_size = min(5, len(traditional_array))
            sample_match = True
            for i in range(sample_size):
                if traditional_array[i] != integrated_array[i]:
                    print(f"✗ {col.name}[{i}]: 値不一致 ({traditional_array[i]} vs {integrated_array[i]})")
                    sample_match = False
                    fixed_match = False
            
            if sample_match:
                print(f"✓ {col.name}: サンプル{sample_size}件一致")
        
        if fixed_match:
            print("✓ 全固定長列のデータ整合性確認")
        else:
            print("✗ 固定長列にデータ不一致あり")
            
    except Exception as e:
        print(f"✗ データ整合性チェック失敗: {e}")
        fixed_match = False
    
    # Parquet出力テスト
    print("\n=== Parquet出力テスト ===")
    try:
        output_traditional = "test_traditional_fully_integrated.parquet"
        output_integrated = "test_integrated_fully_integrated.parquet"
        
        pq.write_table(pa.Table.from_batches([batch_traditional]), output_traditional)
        pq.write_table(pa.Table.from_batches([batch_integrated]), output_integrated)
        
        print(f"✓ Parquet出力成功")
        print(f"  従来版: {output_traditional}")
        print(f"  Pass1完全統合版: {output_integrated}")
        
    except Exception as e:
        print(f"✗ Parquet出力失敗: {e}")
    
    # 総合結果
    print("\n=== 総合評価 ===")
    success_criteria = fixed_match and speedup > 1.5  # 1.5倍以上の高速化を期待
    
    if success_criteria:
        print("✓ Pass1完全統合最適化: 大成功")
        print(f"  データ整合性: OK")
        print(f"  性能向上: {speedup:.2f}x (目標1.5x以上達成)")
        if speedup > 5:
            print("  🎉 期待を上回る大幅な性能向上！")
        return True
    else:
        print("✗ Pass1完全統合最適化: 課題あり")
        if not fixed_match:
            print("  - データ整合性に問題")
        if speedup <= 1.5:
            print(f"  - 性能改善不十分 ({speedup:.2f}x、目標1.5x)")
        return False

if __name__ == "__main__":
    # CUDA初期化
    try:
        cuda.current_context()
        print("CUDA context OK")
    except Exception as e:
        print(f"CUDA context initialization failed: {e}")
        sys.exit(1)
    
    success = test_fully_integrated_optimization()
    
    if success:
        print("\n🎊 Pass1完全統合最適化テスト: 成功")
        print("GPU最適化の新たな地平を切り開きました！")
    else:
        print("\n😞 Pass1完全統合最適化テスト: 要改善")
        print("実装を見直して再挑戦が必要です。")
    
    sys.exit(0 if success else 1)