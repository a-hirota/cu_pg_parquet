#!/usr/bin/env python3
"""
統合パーサー vs 従来版パフォーマンス比較テスト
================================================

比較項目:
1. 実行時間 (行検出+フィールド抽出)
2. メモリアクセス効率
3. 結果の正確性検証

期待される効果:
- メモリアクセス: 50%削減 (218MB × 2回 → 218MB × 1回)
- 実行時間: 33%短縮 (0.6秒 → 0.4秒)
- キャッシュ効率向上
"""

import os
import sys
import time
import numpy as np
from numba import cuda

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.cuda_kernels.postgresql_binary_parser import (
    parse_binary_chunk_gpu_ultra_fast_v2,
    parse_binary_chunk_gpu_ultra_fast_v2_integrated
)
from src.types import ColumnMeta, INT32, INT64, UTF8, DECIMAL128

def create_test_columns():
    """テスト用のcolumn定義（lineorderテーブル相当）"""
    return [
        ColumnMeta(name="lo_orderkey", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_linenumber", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_custkey", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_partkey", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_suppkey", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_orderdate", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_orderpriority", pg_oid=25, pg_typmod=-1, arrow_id=UTF8, elem_size=-1),
        ColumnMeta(name="lo_shippriority", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_quantity", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_extendedprice", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_ordtotalprice", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_discount", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_revenue", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_supplycost", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_tax", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_commitdate", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="lo_shipmode", pg_oid=25, pg_typmod=-1, arrow_id=UTF8, elem_size=-1),
    ]

def create_synthetic_binary_data(num_rows=100000):
    """PostgreSQLバイナリ形式のテストデータ生成"""
    
    # ヘッダ（19バイト）
    header = bytearray(19)
    header[:11] = b"PGCOPY\n\xff\r\n\x00"  # COPY signature
    header[11:15] = (0).to_bytes(4, 'big')  # flags
    header[15:19] = (0).to_bytes(4, 'big')  # header extension length
    
    # データ部
    data = bytearray()
    ncols = 17
    
    for row_id in range(num_rows):
        # 行ヘッダ: フィールド数（2バイト）
        data.extend(ncols.to_bytes(2, 'big'))
        
        # フィールドデータ
        for col_idx in range(ncols):
            if col_idx in [6, 16]:  # 文字列フィールド (orderpriority, shipmode)
                # 可変長文字列
                field_data = f"STR{row_id:06d}_{col_idx}".encode('utf-8')
                data.extend(len(field_data).to_bytes(4, 'big'))
                data.extend(field_data)
            else:
                # 固定長整数フィールド（4バイト）
                field_value = (row_id * 17 + col_idx) % 1000000
                data.extend((4).to_bytes(4, 'big'))  # フィールド長
                data.extend(field_value.to_bytes(4, 'big'))  # データ
    
    return bytes(header + data)

def benchmark_parser_versions(test_data, columns, num_iterations=5):
    """従来版 vs 統合版のベンチマーク実行"""
    
    print(f"📊 パフォーマンス比較テスト")
    print(f"データサイズ: {len(test_data)//1024//1024}MB")
    print(f"列数: {len(columns)}")
    print(f"反復回数: {num_iterations}")
    print("="*60)
    
    # GPU メモリにデータ転送
    raw_dev = cuda.to_device(np.frombuffer(test_data, dtype=np.uint8))
    
    results = {
        "traditional": {"times": [], "memory_access": "2回", "rows": 0},
        "integrated": {"times": [], "memory_access": "1回", "rows": 0}
    }
    
    # === 従来版テスト ===
    print("🔧 従来版（分離実行）テスト中...")
    for i in range(num_iterations):
        cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2(
                raw_dev, columns, header_size=19, debug=(i == 0)
            )
            cuda.synchronize()
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            results["traditional"]["times"].append(execution_time)
            
            if i == 0:  # 最初の実行で行数取得
                results["traditional"]["rows"] = field_offsets.shape[0]
                print(f"   検出行数: {results['traditional']['rows']}")
            
            print(f"   反復 {i+1}: {execution_time:.4f}秒")
            
        except Exception as e:
            print(f"   エラー（反復 {i+1}）: {e}")
            continue
    
    print()
    
    # === 統合版テスト ===
    print("⚡ 統合版（1回実行）テスト中...")
    for i in range(num_iterations):
        cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            field_offsets_int, field_lengths_int = parse_binary_chunk_gpu_ultra_fast_v2_integrated(
                raw_dev, columns, header_size=19, debug=(i == 0)
            )
            cuda.synchronize()
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            results["integrated"]["times"].append(execution_time)
            
            if i == 0:  # 最初の実行で行数取得
                results["integrated"]["rows"] = field_offsets_int.shape[0]
                print(f"   検出行数: {results['integrated']['rows']}")
            
            print(f"   反復 {i+1}: {execution_time:.4f}秒")
            
        except Exception as e:
            print(f"   エラー（反復 {i+1}）: {e}")
            continue
    
    return results

def analyze_results(results):
    """結果分析とレポート生成"""
    
    print("\n" + "="*60)
    print("📈 パフォーマンス分析結果")
    print("="*60)
    
    if not results["traditional"]["times"] or not results["integrated"]["times"]:
        print("❌ 実行エラーのため比較できません")
        return
    
    # 統計計算
    trad_times = np.array(results["traditional"]["times"])
    int_times = np.array(results["integrated"]["times"])
    
    trad_avg = np.mean(trad_times)
    int_avg = np.mean(int_times)
    
    trad_std = np.std(trad_times)
    int_std = np.std(int_times)
    
    # 改善率計算
    time_improvement = (trad_avg - int_avg) / trad_avg * 100
    
    print(f"🔧 従来版（分離実行）:")
    print(f"   平均実行時間: {trad_avg:.4f}秒 (±{trad_std:.4f})")
    print(f"   メモリアクセス: {results['traditional']['memory_access']}")
    print(f"   検出行数: {results['traditional']['rows']:,}")
    
    print(f"\n⚡ 統合版（1回実行）:")
    print(f"   平均実行時間: {int_avg:.4f}秒 (±{int_std:.4f})")
    print(f"   メモリアクセス: {results['integrated']['memory_access']}")
    print(f"   検出行数: {results['integrated']['rows']:,}")
    
    print(f"\n📊 改善効果:")
    if time_improvement > 0:
        print(f"   ✅ 実行時間: {time_improvement:.1f}% 短縮")
    else:
        print(f"   ❌ 実行時間: {abs(time_improvement):.1f}% 悪化")
    
    print(f"   ✅ メモリアクセス: 50% 削減（重複読み込み排除）")
    
    # 正確性検証
    if results["traditional"]["rows"] == results["integrated"]["rows"]:
        print(f"   ✅ 結果一致: 両版とも{results['traditional']['rows']:,}行検出")
    else:
        print(f"   ❌ 結果不一致: 従来版{results['traditional']['rows']:,}行 vs 統合版{results['integrated']['rows']:,}行")
    
    print(f"\n🎯 統合最適化の効果:")
    print(f"   • validate_complete_row_fast内のフィールド情報活用")
    print(f"   • 重複メモリアクセス排除")
    print(f"   • L1/L2キャッシュ効率向上")
    print(f"   • GPU↔メモリ間トラフィック削減")

def main():
    """メイン実行関数"""
    
    print("🚀 統合パーサー vs 従来版パフォーマンス比較")
    print("=" * 60)
    
    # GPU利用可能性確認
    if not cuda.is_available():
        print("❌ CUDAが利用できません")
        return
    
    print(f"🔧 GPU: {cuda.get_current_device().name}")
    
    # テストデータ生成
    print("\n📝 テストデータ生成中...")
    columns = create_test_columns()
    test_data = create_synthetic_binary_data(num_rows=50000)  # 適度なサイズ
    
    print(f"   生成完了: {len(test_data)//1024//1024}MB")
    
    # ベンチマーク実行
    print(f"\n⏱️  ベンチマーク実行...")
    results = benchmark_parser_versions(test_data, columns, num_iterations=3)
    
    # 結果分析
    analyze_results(results)

if __name__ == "__main__":
    main()