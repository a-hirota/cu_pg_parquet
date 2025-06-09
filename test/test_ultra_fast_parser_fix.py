"""
★100点改修テスト: detect_rows_optimized競合解決の検証
=====================================================
ブロック単位協調処理により3件欠落問題を完全解決したかを検証します。
"""

import numpy as np
from numba import cuda
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cuda_kernels.ultra_fast_parser import (
    parse_binary_chunk_gpu_ultra_fast_v2,
    get_device_properties
)
from src.types import ColumnMeta, INT32, INT64, UTF8

def create_test_data_with_known_rows(num_rows=1000000, ncols=17):
    """既知の行数で完全にコントロールされたテストデータを生成"""
    print(f"[TEST] 🔧 {num_rows}行 × {ncols}列のテストデータ生成中...")
    
    # PostgreSQLバイナリヘッダ（19バイト）
    header = bytearray(19)
    header[:4] = b'PGCOPY\n\377\r\n\0'  # シグネチャ
    header[11:15] = (0).to_bytes(4, 'big')  # フラグ
    header[15:19] = (0).to_bytes(4, 'big')  # ヘッダ拡張長
    
    # 行データ生成
    data = bytearray()
    for row_id in range(num_rows):
        # フィールド数（2バイト）
        data.extend(ncols.to_bytes(2, 'big'))
        
        # 各フィールド（4バイト長 + データ）
        for col_id in range(ncols):
            if col_id == 0:  # 最初のカラムは行ID
                field_data = str(row_id).encode('utf-8')
                data.extend(len(field_data).to_bytes(4, 'big'))
                data.extend(field_data)
            elif col_id < 5:  # 固定長整数フィールド
                field_data = (row_id + col_id).to_bytes(4, 'big')
                data.extend((4).to_bytes(4, 'big'))
                data.extend(field_data)
            else:  # 可変長文字列フィールド
                field_data = f"data_{row_id}_{col_id}".encode('utf-8')
                data.extend(len(field_data).to_bytes(4, 'big'))
                data.extend(field_data)
    
    # 終端マーカー（0xFFFF）
    data.extend((0xFFFF).to_bytes(2, 'big'))
    
    full_data = header + data
    print(f"[TEST] ✅ テストデータ生成完了: {len(full_data)//1024//1024}MB ({len(full_data)}バイト)")
    return full_data, num_rows

def test_competitive_stress_concurrency():
    """★競合ストレステスト: 大量並列での完全性検証"""
    print("\n" + "="*80)
    print("🚀 競合ストレステスト: ブロック単位協調処理の効果検証")
    print("="*80)
    
    # GPU特性取得
    props = get_device_properties()
    sm_count = props.get('MULTIPROCESSOR_COUNT', 108)
    print(f"[TEST] 🔧 GPU: {sm_count}SM, 最大ブロック/SM: {sm_count * 12}")
    
    # テストデータ生成（競合が発生しやすいサイズ）
    test_rows = 1_000_000  # 100万行で精密テスト
    raw_data, expected_rows = create_test_data_with_known_rows(test_rows, ncols=17)
    
    # GPUメモリに転送
    raw_dev = cuda.to_device(np.frombuffer(raw_data, dtype=np.uint8))
    print(f"[TEST] 📤 GPU転送完了: {len(raw_data)//1024//1024}MB")
    
    # ColumnMeta定義（17列）
    columns = [
        ColumnMeta(name="id", pg_oid=25, pg_typmod=-1, arrow_id=UTF8, elem_size=-1),  # varchar
        ColumnMeta(name="col1", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),  # int4
        ColumnMeta(name="col2", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),  # int4
        ColumnMeta(name="col3", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),  # int4
        ColumnMeta(name="col4", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4),  # int4
    ] + [
        ColumnMeta(name=f"data_{i}", pg_oid=25, pg_typmod=-1, arrow_id=UTF8, elem_size=-1)
        for i in range(5, 17)  # 残り12列は可変長文字列
    ]
    
    print(f"[TEST] 🎯 期待行数: {expected_rows:,}行")
    
    # ★100点改修版での解析（5回実行して一貫性確認）
    results = []
    for test_run in range(5):
        print(f"\n[TEST] 🔄 実行 {test_run + 1}/5: ブロック単位協調処理")
        
        field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=19, debug=True
        )
        
        detected_rows = field_offsets.shape[0]
        results.append(detected_rows)
        
        print(f"[TEST] 📊 検出行数: {detected_rows:,} / {expected_rows:,}")
        detection_rate = (detected_rows / expected_rows) * 100
        missing = expected_rows - detected_rows
        
        if detected_rows == expected_rows:
            print(f"[TEST] ✅ 完全検出達成: {detection_rate:.6f}%")
        else:
            print(f"[TEST] ⚠️  未達成: {detection_rate:.6f}% (不足: {missing}行)")
    
    # 結果分析
    print(f"\n[TEST] 📈 結果統計:")
    print(f"  平均検出行数: {np.mean(results):,.1f}")
    print(f"  標準偏差: {np.std(results):,.1f}")
    print(f"  最小値: {np.min(results):,}")
    print(f"  最大値: {np.max(results):,}")
    
    # 100点評価
    perfect_runs = sum(1 for r in results if r == expected_rows)
    success_rate = (perfect_runs / len(results)) * 100
    
    print(f"\n[TEST] 🏆 100点改修評価:")
    print(f"  完全検出回数: {perfect_runs}/5回")
    print(f"  成功率: {success_rate:.1f}%")
    
    if success_rate == 100.0:
        print(f"[TEST] 🎉 100点達成！ブロック単位協調処理による競合完全解決")
        return True
    else:
        print(f"[TEST] 🔧 改修要: まだ {100 - success_rate:.1f}%の確率で競合発生")
        return False

def test_block_collaboration_verification():
    """★ブロック協調処理の内部動作検証"""
    print("\n" + "="*80)
    print("🔍 ブロック協調処理の内部動作検証")
    print("="*80)
    
    # 小さなテストデータで詳細検証
    raw_data, expected_rows = create_test_data_with_known_rows(10000, ncols=17)
    raw_dev = cuda.to_device(np.frombuffer(raw_data, dtype=np.uint8))
    
    columns = [ColumnMeta(name=f"col_{i}", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4) for i in range(17)]
    
    print(f"[TEST] 🎯 小規模検証: {expected_rows:,}行")
    
    field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2(
        raw_dev, columns, header_size=19, debug=True
    )
    
    detected_rows = field_offsets.shape[0]
    print(f"[TEST] 📊 検出結果: {detected_rows} / {expected_rows}")
    
    if detected_rows == expected_rows:
        print(f"[TEST] ✅ 小規模検証成功: ブロック協調処理が正常動作")
        return True
    else:
        print(f"[TEST] ❌ 小規模検証失敗: ブロック協調処理に問題")
        return False

def main():
    """メインテスト実行"""
    print("🧪 100点改修テスト開始: detect_rows_optimized競合解決検証")
    
    try:
        # GPU利用可能性確認
        device = cuda.get_current_device()
        print(f"[TEST] 🖥️  GPU検出: {device.name}")
        
        # テスト実行
        test1_passed = test_block_collaboration_verification()
        test2_passed = test_competitive_stress_concurrency()
        
        # 最終評価
        if test1_passed and test2_passed:
            print(f"\n🎉 100点改修成功！")
            print(f"✅ ブロック単位協調処理により3件欠落問題を完全解決")
            print(f"✅ 競合状態を根本から排除し、100%検出率を達成")
        else:
            print(f"\n🔧 改修継続必要")
            print(f"❌ まだ競合による欠落が発生している可能性")
            
    except Exception as e:
        print(f"[TEST] ❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()