#!/usr/bin/env python
"""
Pass1 Ultimate統合最適化テスト - 最小実装版
==========================================

既存のtest_pass1_fully_integrated.pyをベースとして、
Ultimate統合版との比較テストを実行
"""

import time
import os
import numpy as np
import pyarrow as pa

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
from src.gpu_decoder_v2 import decode_chunk
from src.gpu_decoder_v4_ultimate_integrated import decode_chunk_ultimate_integrated

def main():
    print("=== Pass1 Ultimate統合最適化テスト ===")
    
    # PostgreSQL接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    # データベース接続
    conn = psycopg.connect(dsn)
    
    try:
        print("1. メタデータ取得中...")
        columns = fetch_column_meta(conn, "SELECT * FROM lineorder LIMIT 50000")  # 小さめに
        
        # 列分析
        fixed_columns = [col for col in columns if not col.is_variable]
        variable_columns = [col for col in columns if col.is_variable]
        decimal_columns = [col for col in fixed_columns if col.arrow_id == 5]  # DECIMAL128 = 5
        
        print(f"   総列数: {len(columns)}")
        print(f"   固定長列: {len(fixed_columns)}列")
        print(f"   可変長列: {len(variable_columns)}列")
        
        print("2. COPY BINARY実行中...")
        start_copy = time.time()
        copy_sql = "COPY (SELECT * FROM lineorder LIMIT 50000) TO STDOUT (FORMAT binary)"
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
        raw_dev, len(columns), header_size=header_size
    )
    parse_time = time.time() - start_parse
    rows = field_lengths_dev.shape[0]
    print(f"   完了: {parse_time:.4f}秒, 行数: {rows}")

    # ----------------------------------
    # 従来版vs Ultimate版比較
    # ----------------------------------
    
    # 5. 従来版デコード
    print("5. 従来版デコード中...")
    start_traditional = time.time()
    batch_traditional = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    traditional_time = time.time() - start_traditional
    print(f"   完了: {traditional_time:.4f}秒")

    # 6. Ultimate統合版デコード
    print("6. Ultimate統合版デコード中...")
    start_ultimate = time.time()
    try:
        batch_ultimate = decode_chunk_ultimate_integrated(raw_dev, field_offsets_dev, field_lengths_dev, columns)
        ultimate_time = time.time() - start_ultimate
        print(f"   完了: {ultimate_time:.4f}秒")
        
        # 性能比較
        speedup = traditional_time / ultimate_time
        print(f"\n=== 性能比較結果 ===")
        print(f"従来版時間        : {traditional_time:.4f}秒")
        print(f"Ultimate統合版時間: {ultimate_time:.4f}秒")
        print(f"高速化率          : {speedup:.2f}x")
        
        # 簡易データ整合性チェック
        print(f"\n=== データ整合性チェック ===")
        print(f"行数一致: {batch_traditional.num_rows} vs {batch_ultimate.num_rows}")
        print(f"列数一致: {batch_traditional.num_columns} vs {batch_ultimate.num_columns}")
        
        # 最初の列の最初の5値を比較
        if len(columns) > 0:
            col_name = columns[0].name
            traditional_values = batch_traditional.column(0).to_pylist()[:5]
            ultimate_values = batch_ultimate.column(0).to_pylist()[:5]
            
            print(f"最初の列 '{col_name}' サンプル比較:")
            for i, (t, u) in enumerate(zip(traditional_values, ultimate_values)):
                match = "✓" if t == u else "✗"
                print(f"  行{i}: {match} {t} vs {u}")
        
        # 結果評価
        target_speedup = 2.0  # 保守的な目標
        if speedup >= target_speedup:
            print(f"\n🎊 Pass1 Ultimate統合最適化: 大成功！ ({speedup:.2f}x高速化)")
            print("Pass2完全廃止により新次元の性能を達成！")
            return True
        elif speedup >= 1.2:
            print(f"\n✓ Pass1 Ultimate統合最適化: 成功 ({speedup:.2f}x高速化)")
            print("有意な性能向上を確認")
            return True
        else:
            print(f"\n△ Pass1 Ultimate統合最適化: 部分的成功 ({speedup:.2f}x高速化)")
            print("実装は動作するが、期待した性能向上は未達成")
            return False
            
    except Exception as e:
        print(f"   エラー: Ultimate統合版デコード失敗: {e}")
        print("   実装にバグがある可能性があります")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)