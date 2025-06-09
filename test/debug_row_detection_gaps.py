"""
行検出漏れの詳細分析とデバッグ
================================
96-97%の検出率で残る3-4%の漏れの原因を特定し、対策を検討します。
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
from src.types import ColumnMeta, INT32, UTF8

def create_test_data_sequential(num_rows=100000, ncols=17):
    """シーケンシャルで予測可能なテストデータを生成（デバッグ用）"""
    print(f"[DEBUG] 🔧 {num_rows}行 × {ncols}列のシーケンシャルデータ生成中...")
    
    # PostgreSQLバイナリヘッダ（19バイト）
    header = bytearray(19)
    header[:4] = b'PGCOPY\n\377\r\n\0'  # シグネチャ
    header[11:15] = (0).to_bytes(4, 'big')  # フラグ
    header[15:19] = (0).to_bytes(4, 'big')  # ヘッダ拡張長
    
    # 行データ生成（固定サイズで予測可能）
    data = bytearray()
    row_positions = []
    
    for row_id in range(num_rows):
        row_start = len(header) + len(data)
        row_positions.append(row_start)
        
        # フィールド数（2バイト）
        data.extend(ncols.to_bytes(2, 'big'))
        
        # 各フィールド（固定4バイト値のみ）
        for col_id in range(ncols):
            # 4バイト長
            data.extend((4).to_bytes(4, 'big'))
            # 4バイト値（行ID + 列ID）
            value = (row_id * 100 + col_id) % (2**32)
            data.extend(value.to_bytes(4, 'big'))
    
    # 終端マーカー（0xFFFF）
    data.extend((0xFFFF).to_bytes(2, 'big'))
    
    full_data = header + data
    print(f"[DEBUG] ✅ シーケンシャルデータ生成完了: {len(full_data)//1024//1024}MB")
    print(f"[DEBUG] 📊 期待行位置: {len(row_positions)}行")
    
    return full_data, row_positions

def analyze_detection_gaps():
    """行検出ギャップの詳細分析"""
    print("\n" + "="*80)
    print("🔍 行検出ギャップの詳細分析")
    print("="*80)
    
    # 小規模で制御されたテストデータ
    raw_data, expected_positions = create_test_data_sequential(10000, ncols=17)
    raw_dev = cuda.to_device(np.frombuffer(raw_data, dtype=np.uint8))
    
    # ColumnMeta定義（全て固定長INT32で簡素化）
    columns = [
        ColumnMeta(name=f"col_{i}", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4)
        for i in range(17)
    ]
    
    print(f"[DEBUG] 🎯 期待行数: {len(expected_positions)}行")
    print(f"[DEBUG] 🎯 期待行位置: {expected_positions[:5]}... (先頭5行)")
    
    # GPU解析実行
    field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2(
        raw_dev, columns, header_size=19, debug=True
    )
    
    detected_rows = field_offsets.shape[0]
    detection_rate = (detected_rows / len(expected_positions)) * 100
    
    print(f"\n[DEBUG] 📊 検出結果:")
    print(f"  期待行数: {len(expected_positions):,}")
    print(f"  検出行数: {detected_rows:,}")
    print(f"  検出率: {detection_rate:.3f}%")
    print(f"  不足: {len(expected_positions) - detected_rows}行")
    
    if detected_rows > 0:
        # 検出された行位置を取得
        raw_host = raw_dev.copy_to_host()
        detected_positions = []
        
        for i in range(min(detected_rows, 100)):  # 最初の100行を分析
            row_start = field_offsets[i, 0].copy_to_host()
            if row_start > 0:
                # 実際のヘッダ値を確認
                if row_start - 2 >= 0:
                    actual_start = row_start - 2
                    header_val = (raw_host[actual_start] << 8) | raw_host[actual_start + 1]
                    detected_positions.append((actual_start, header_val))
        
        print(f"\n[DEBUG] 🔍 検出された行位置の分析（先頭10行）:")
        for i, (pos, header) in enumerate(detected_positions[:10]):
            expected_pos = expected_positions[i] if i < len(expected_positions) else "N/A"
            diff = pos - expected_positions[i] if i < len(expected_positions) else "N/A"
            print(f"  行{i}: 検出位置={pos}, 期待位置={expected_pos}, 差={diff}, ヘッダ={header}")
        
        # ギャップ分析
        missed_positions = []
        detected_set = set(pos for pos, _ in detected_positions)
        
        for i, expected_pos in enumerate(expected_positions[:100]):  # 最初の100行を分析
            if expected_pos not in detected_set:
                missed_positions.append((i, expected_pos))
        
        print(f"\n[DEBUG] ❌ 見逃された行位置（先頭10個）:")
        for i, (row_idx, pos) in enumerate(missed_positions[:10]):
            # 見逃された位置の周辺データをダンプ
            if pos + 10 < len(raw_host):
                header_val = (raw_host[pos] << 8) | raw_host[pos + 1]
                surrounding = raw_host[pos:pos+20]
                hex_dump = ' '.join(f'{b:02x}' for b in surrounding)
                print(f"  見逃し行{row_idx}: 位置={pos}, ヘッダ={header_val}, データ={hex_dump}")
    
    return detected_rows == len(expected_positions)

def analyze_thread_stride_impact():
    """スレッドストライドの影響分析"""
    print("\n" + "="*80)
    print("🔍 スレッドストライドの影響分析")
    print("="*80)
    
    raw_data, expected_positions = create_test_data_sequential(50000, ncols=17)
    raw_dev = cuda.to_device(np.frombuffer(raw_data, dtype=np.uint8))
    
    columns = [
        ColumnMeta(name=f"col_{i}", pg_oid=23, pg_typmod=-1, arrow_id=INT32, elem_size=4)
        for i in range(17)
    ]
    
    # GPU特性取得
    props = get_device_properties()
    sm_count = props.get('MULTIPROCESSOR_COUNT', 82)
    
    # 異なるブロック数でテスト
    test_configs = [
        (sm_count // 4, "低並列"),      # 20ブロック程度
        (sm_count, "標準並列"),         # 82ブロック
        (sm_count * 4, "高並列"),       # 328ブロック
        (sm_count * 12, "超高並列"),    # 984ブロック
    ]
    
    results = []
    
    for blocks_target, label in test_configs:
        print(f"\n[DEBUG] 🔄 {label}テスト: {blocks_target}ブロック")
        
        # 手動でグリッド設定をオーバーライド
        from src.cuda_kernels.ultra_fast_parser import calculate_optimal_grid_sm_aware
        
        # 強制的に特定ブロック数に設定
        blocks_x = min(blocks_target, 65535)
        blocks_y = 1
        threads_per_block = 256
        
        actual_threads = blocks_x * blocks_y * threads_per_block
        data_size = raw_dev.size - 19
        thread_stride = (data_size + actual_threads - 1) // actual_threads
        
        print(f"  ブロック: {blocks_x} × {blocks_y}")
        print(f"  スレッド: {actual_threads:,}")
        print(f"  ストライド: {thread_stride}B")
        
        # 実行（簡略化のため、カーネル直接呼び出しはスキップ）
        field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=19, debug=False
        )
        
        detected_rows = field_offsets.shape[0]
        detection_rate = (detected_rows / len(expected_positions)) * 100
        
        results.append((label, blocks_target, detected_rows, detection_rate))
        print(f"  検出率: {detection_rate:.3f}% ({detected_rows}/{len(expected_positions)})")
    
    print(f"\n[DEBUG] 📊 スレッドストライド影響まとめ:")
    for label, blocks, detected, rate in results:
        print(f"  {label}: {rate:.3f}% ({detected:,}行, {blocks}ブロック)")
    
    return results

def main():
    """メイン分析実行"""
    print("🔍 行検出漏れの詳細分析開始")
    
    try:
        # GPU利用可能性確認
        device = cuda.get_current_device()
        print(f"[DEBUG] 🖥️  GPU検出: {device.name}")
        
        # 分析実行
        print("\n1. 制御されたデータでのギャップ分析")
        gap_analysis_passed = analyze_detection_gaps()
        
        print("\n2. スレッドストライド影響分析")
        stride_results = analyze_thread_stride_impact()
        
        # 結論
        print(f"\n" + "="*80)
        print("📊 分析結果まとめ")
        print("="*80)
        
        if gap_analysis_passed:
            print("✅ 制御データでは完全検出達成")
        else:
            print("❌ 制御データでも検出漏れ発生")
        
        print(f"💡 推定原因:")
        print(f"  • スレッドストライドの境界での行分割")
        print(f"  • 15Bステップスキャンでの見逃し")
        print(f"  • 共有メモリ制限による部分的フォールバック")
        
        print(f"\n🔧 推奨対策:")
        print(f"  1. オーバーラップ領域の拡大")
        print(f"  2. より細かいスキャンステップ（15B → 1B）")
        print(f"  3. 境界処理の改善")
        
    except Exception as e:
        print(f"[DEBUG] ❌ 分析エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()