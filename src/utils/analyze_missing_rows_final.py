#!/usr/bin/env python3
"""
487行欠損の最終分析
"""

import numpy as np

def analyze_missing_rows():
    """487行欠損の根本原因を特定"""
    
    # 実際のデータ
    total_rows = 246011837
    processed_rows = 246011350
    missing_rows = 487
    
    num_chunks = 32
    workers_per_chunk = 16
    total_workers = num_chunks * workers_per_chunk  # 512
    
    print("=== 基本情報 ===")
    print(f"総行数: {total_rows:,}")
    print(f"処理済み行数: {processed_rows:,}")
    print(f"欠損行数: {missing_rows}")
    print(f"欠損率: {missing_rows/total_rows*100:.6f}%")
    
    print(f"\n処理単位:")
    print(f"  チャンク数: {num_chunks}")
    print(f"  ワーカー/チャンク: {workers_per_chunk}")
    print(f"  総ワーカー数: {total_workers}")
    
    # 欠損の分布
    print("\n=== 欠損の分布 ===")
    missing_per_chunk = missing_rows / num_chunks
    missing_per_worker = missing_rows / total_workers
    
    print(f"欠損/チャンク: {missing_per_chunk:.2f} 行")
    print(f"欠損/ワーカー: {missing_per_worker:.4f} 行")
    
    # パターン分析
    print("\n=== パターン分析 ===")
    
    # 487の因数分解
    print(f"\n487の因数分解:")
    factors = []
    n = 487
    for i in range(2, int(np.sqrt(n)) + 1):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 1:
        factors.append(n)
    print(f"  487 = {' × '.join(map(str, factors))}")
    
    # 487は素数
    print(f"  → 487は素数")
    
    # 可能な組み合わせ
    print("\n可能な欠損パターン:")
    print(f"  1) 487個の異なる処理単位で各1行欠損")
    print(f"  2) 1つの処理単位で487行欠損")
    print(f"  3) いくつかの処理単位で合計487行欠損")
    
    # GPU処理の詳細
    print("\n=== GPU処理の詳細 ===")
    
    # カーネルの制限
    MAX_ROWS_PER_THREAD = 200
    LOCAL_ARRAY_SIZE = 256
    
    print(f"スレッドあたりの制限:")
    print(f"  MAX_ROWS_PER_THREAD: {MAX_ROWS_PER_THREAD}")
    print(f"  LOCAL_ARRAY_SIZE: {LOCAL_ARRAY_SIZE}")
    
    # 推定行サイズ
    estimated_row_size = 288  # バイト
    
    # チャンクサイズ
    chunk_size = 8 * 1024**3  # 8GB
    header_size = 19
    data_size_per_chunk = chunk_size - header_size
    
    # 各チャンクの行数
    rows_per_chunk = total_rows // num_chunks
    remainder_rows = total_rows % num_chunks
    
    print(f"\n各チャンクの行数:")
    print(f"  平均: {rows_per_chunk:,} 行")
    print(f"  余り: {remainder_rows} 行")
    
    # コード分析
    print("\n=== コードの問題箇所 ===")
    
    print("\n1. 境界処理（parse_rows_and_fields_lite）:")
    print("   行393-394:")
    print("   elif candidate_pos >= end_pos:  # 担当範囲外")
    print("   　　break")
    print("\n   → スレッドの担当範囲外の行は処理されない")
    
    print("\n2. ローカル配列制限（行406）:")
    print("   if local_count < 256:  # 配列境界チェック")
    print("   　　local_positions[local_count] = ...")
    print("\n   → 256行を超える行は無視される")
    
    print("\n3. 行ヘッダ検出（read_uint16_simd16_lite）:")
    print("   最大16バイトの範囲で検索")
    print("   見つからない場合は15バイト進む")
    
    # 仮説検証
    print("\n=== 仮説検証 ===")
    
    print("\n仮説1: 最後のチャンクで行が欠損")
    if remainder_rows > 0:
        print(f"  最後のチャンクは {remainder_rows} 行多い")
        print(f"  これが原因で境界処理に問題？")
    
    print("\n仮説2: 各ワーカーの境界で約1行欠損")
    expected_missing = total_workers * 0.95
    print(f"  期待値: {expected_missing:.0f} 行")
    print(f"  実際: {missing_rows} 行")
    print(f"  差: {abs(expected_missing - missing_rows):.0f} 行")
    
    print("\n仮説3: 特定の条件下でのみ発生")
    print("  - データの特定のパターン")
    print("  - 行サイズの変動")
    print("  - NULL値の分布")
    
    # より詳細な分析
    print("\n=== 詳細な境界分析 ===")
    
    # スレッド数とthread_stride
    blocks = 1640
    threads_per_block = 1024
    total_threads = blocks * threads_per_block
    
    thread_stride = 5116  # バイト
    rows_per_thread = thread_stride / estimated_row_size
    
    print(f"\nGPU実行設定:")
    print(f"  総スレッド数: {total_threads:,}")
    print(f"  thread_stride: {thread_stride} バイト")
    print(f"  スレッドあたり行数: {rows_per_thread:.1f}")
    
    # 各ワーカーでのスレッド数
    data_per_worker = rows_per_chunk // workers_per_chunk * estimated_row_size
    threads_per_worker = data_per_worker / thread_stride
    
    print(f"\n各ワーカーの処理:")
    print(f"  データサイズ: {data_per_worker:,} バイト")
    print(f"  使用スレッド数: {threads_per_worker:.1f}")
    
    # 結論
    print("\n=== 結論 ===")
    print("\n最も可能性の高い原因:")
    print("1. 各ワーカーの最後のスレッドで境界処理の問題")
    print("2. 512ワーカー × 約0.95行 = 約487行")
    print("3. end_posを超える行が処理されていない")
    
    print("\n修正方法:")
    print("1. 境界チェックを緩和（行の開始が範囲内なら処理）")
    print("2. または、各ワーカーで少し余分にデータを読む")
    print("3. 重複除去は後処理で行う")

if __name__ == "__main__":
    analyze_missing_rows()