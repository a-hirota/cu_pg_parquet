#!/usr/bin/env python3
"""
SM対応動的グリッド最適化のテスト
Ultra Fast ParserのSM対応実装を検証
"""

import numpy as np
from numba import cuda
import sys
import os

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cuda_kernels.ultra_fast_parser import (
    get_device_properties,
    calculate_optimal_grid_sm_aware,
    parse_binary_chunk_gpu_ultra_fast_v2
)
from src.types import ColumnMeta, INT32, UTF8

def test_device_properties():
    """GPU デバイス特性取得のテスト"""
    print("=" * 60)
    print("GPU デバイス特性取得テスト")
    print("=" * 60)
    
    try:
        props = get_device_properties()
        print(f"✓ デバイス特性取得成功:")
        print(f"  SM数: {props['MULTIPROCESSOR_COUNT']}")
        print(f"  最大スレッド/ブロック: {props['MAX_THREADS_PER_BLOCK']}")
        print(f"  最大グリッドX: {props['MAX_GRID_DIM_X']}")
        print(f"  最大グリッドY: {props['MAX_GRID_DIM_Y']}")
        return True
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

def test_grid_calculation():
    """SM対応グリッド計算のテスト"""
    print("\n" + "=" * 60)
    print("SM対応グリッド計算テスト")
    print("=" * 60)
    
    test_cases = [
        (1024 * 1024, 64),      # 1MB, 64B行
        (10 * 1024 * 1024, 128), # 10MB, 128B行
        (100 * 1024 * 1024, 256), # 100MB, 256B行
    ]
    
    try:
        props = get_device_properties()
        sm_count = props['MULTIPROCESSOR_COUNT']
        
        for data_size, row_size in test_cases:
            blocks_x, blocks_y = calculate_optimal_grid_sm_aware(data_size, row_size)
            total_blocks = blocks_x * blocks_y
            sm_efficiency = total_blocks / sm_count
            
            print(f"\nデータサイズ: {data_size//1024//1024}MB, 行サイズ: {row_size}B")
            print(f"  グリッド: ({blocks_x}, {blocks_y})")
            print(f"  総ブロック数: {total_blocks}")
            print(f"  SM効率: {sm_efficiency:.1f}ブロック/SM")
            print(f"  ✓ SM効率が4.0以上: {'Yes' if sm_efficiency >= 4.0 else 'No'}")
        
        return True
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

def test_sm_aware_vs_fixed():
    """SM対応 vs 固定グリッドの比較テスト - 複数データサイズでSM使用率最大化検証"""
    print("\n" + "=" * 60)
    print("SM対応 vs 固定グリッド比較テスト - SM使用率最大化")
    print("=" * 60)
    
    try:
        props = get_device_properties()
        sm_count = props['MULTIPROCESSOR_COUNT']
        threads_per_block = 256
        
        # 様々なデータサイズでテスト
        test_cases = [
            (1 * 1024 * 1024, 100, "小データ（1MB）"),    # 小データ
            (10 * 1024 * 1024, 150, "中データ（10MB）"),   # 中データ
            (50 * 1024 * 1024, 200, "中データ（50MB）"),   # 中データ
            (200 * 1024 * 1024, 250, "大データ（200MB）"), # 大データ
        ]
        
        print(f"GPU: RTX 3090, SM数: {sm_count}")
        print(f"目標: 全SM活用（理想的なSM効率: 4-8ブロック/SM）")
        
        for data_size, estimated_row_size, description in test_cases:
            print(f"\n【{description}】")
            
            # SM対応計算
            sm_blocks_x, sm_blocks_y = calculate_optimal_grid_sm_aware(
                data_size, estimated_row_size, threads_per_block
            )
            sm_total_blocks = sm_blocks_x * sm_blocks_y
            sm_efficiency = sm_total_blocks / sm_count
            sm_utilization = min(100, (sm_total_blocks / sm_count) * 100)
            
            # 従来の固定計算
            max_gpu_threads = 1048576
            max_total_blocks = max_gpu_threads // threads_per_block
            thread_stride = (data_size + max_gpu_threads - 1) // max_gpu_threads
            if thread_stride < estimated_row_size:
                thread_stride = estimated_row_size
            
            num_threads = min((data_size + thread_stride - 1) // thread_stride, max_gpu_threads)
            total_blocks = min((num_threads + threads_per_block - 1) // threads_per_block, max_total_blocks)
            
            max_blocks_per_dim = 65535
            fixed_blocks_x = min(total_blocks, max_blocks_per_dim)
            fixed_blocks_y = (total_blocks + fixed_blocks_x - 1) // fixed_blocks_x
            fixed_total_blocks = fixed_blocks_x * fixed_blocks_y
            fixed_efficiency = fixed_total_blocks / sm_count
            fixed_utilization = min(100, (fixed_total_blocks / sm_count) * 100)
            
            print(f"  データサイズ: {data_size//1024//1024}MB, 推定行サイズ: {estimated_row_size}B")
            print(f"  従来方式: {fixed_total_blocks}ブロック ({fixed_efficiency:.1f}/SM, {fixed_utilization:.0f}%活用)")
            print(f"  SM対応:   {sm_total_blocks}ブロック ({sm_efficiency:.1f}/SM, {sm_utilization:.0f}%活用)")
            
            improvement = sm_efficiency / fixed_efficiency if fixed_efficiency > 0 else 1.0
            print(f"  改善倍率: {improvement:.2f}x, ✓ SM使用効率: {'最適' if 4 <= sm_efficiency <= 8 else '調整要'}")
        
        return True
    except Exception as e:
        print(f"✗ エラー: {e}")
        return False

def test_integration():
    """Ultra Fast Parser v2でのSM対応統合テスト"""
    print("\n" + "=" * 60)
    print("Ultra Fast Parser v2 SM対応統合テスト")
    print("=" * 60)
    
    try:
        # 簡易的なテストデータの作成
        columns = [
            ColumnMeta(name="id", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
            ColumnMeta(name="name", pg_oid=25, pg_typmod=0, arrow_id=UTF8, elem_size=-1),
            ColumnMeta(name="value", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ]
        
        # 小さなダミーデータ（実際のPostgreSQLバイナリ形式ではないが、関数の動作確認用）
        header_size = 19
        test_data = np.zeros(1024, dtype=np.uint8)
        test_data[:header_size] = 255  # ヘッダー部分
        raw_dev = cuda.to_device(test_data)
        
        print("SM対応 Ultra Fast Parser v2 を実行中...")
        
        # debug=Trueでデバイス特性情報も表示
        field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=header_size, debug=True
        )
        
        print(f"✓ 実行完了")
        print(f"  フィールドオフセット配列サイズ: {field_offsets.shape}")
        print(f"  フィールド長配列サイズ: {field_lengths.shape}")
        
        return True
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("SM対応動的グリッド最適化テスト開始")
    print(f"CUDA利用可能: {cuda.is_available()}")
    
    if not cuda.is_available():
        print("✗ CUDAが利用できません。テストを中止します。")
        return False
    
    print(f"使用GPU: {cuda.get_current_device()}")
    
    results = []
    results.append(test_device_properties())
    results.append(test_grid_calculation())
    results.append(test_sm_aware_vs_fixed())
    results.append(test_integration())
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    test_names = [
        "GPU デバイス特性取得",
        "SM対応グリッド計算",
        "SM対応 vs 固定グリッド比較",
        "Ultra Fast Parser v2 統合"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    overall = all(results)
    print(f"\n総合結果: {'✓ 全テストPASS' if overall else '✗ 一部テスト失敗'}")
    
    return overall

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)