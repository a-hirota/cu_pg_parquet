#!/usr/bin/env python3
"""
境界スレッドの処理と487行欠損の詳細分析
"""

import numpy as np

def analyze_boundary_threads():
    """境界スレッドでの処理漏れを詳細分析"""
    
    # 実際のパラメータ
    total_rows = 246011837
    missing_rows = 487
    num_chunks = 32
    workers_per_chunk = 16
    
    # GPU設定（実行結果から）
    blocks_x = 1640
    blocks_y = 1  
    threads_per_block = 1024
    total_threads = blocks_x * blocks_y * threads_per_block
    
    # データとスレッド設定
    chunk_size = 8 * 1024**3  # 8GB
    header_size = 19
    data_size_per_chunk = chunk_size - header_size
    estimated_row_size = 288  # バイト
    thread_stride = 5116  # バイト
    rows_per_thread = thread_stride / estimated_row_size  # 17.8行
    
    print("=== 境界スレッドの詳細分析 ===")
    print(f"総行数: {total_rows:,}")
    print(f"欠損行数: {missing_rows}")
    print(f"チャンク数: {num_chunks}")
    print(f"ワーカー数/チャンク: {workers_per_chunk}")
    print(f"総処理単位: {num_chunks * workers_per_chunk}")
    
    print(f"\nGPU設定:")
    print(f"  総スレッド数: {total_threads:,}")
    print(f"  thread_stride: {thread_stride} バイト")
    print(f"  スレッドあたり行数: {rows_per_thread:.1f}")
    
    # 各ワーカーの処理
    print("\n=== ワーカーレベルの分析 ===")
    rows_per_chunk = total_rows // num_chunks
    rows_per_worker = rows_per_chunk // workers_per_chunk
    
    print(f"行/チャンク: {rows_per_chunk:,}")
    print(f"行/ワーカー: {rows_per_worker:,}")
    
    # 各ワーカーのデータサイズ
    data_per_worker = rows_per_worker * estimated_row_size
    print(f"\nデータサイズ/ワーカー: {data_per_worker:,} バイト")
    
    # 各ワーカーが使用するスレッド数
    threads_per_worker = data_per_worker / thread_stride
    print(f"スレッド数/ワーカー: {threads_per_worker:.1f}")
    
    # 最後のスレッドの処理
    last_thread_data = data_per_worker % thread_stride
    last_thread_rows = last_thread_data / estimated_row_size
    print(f"\n最後のスレッドのデータ: {last_thread_data} バイト")
    print(f"最後のスレッドの行数: {last_thread_rows:.1f}")
    
    # カーネルコードの問題点
    print("\n=== カーネルコードの境界処理 ===")
    print("\n1. 担当範囲の計算（行367-369）:")
    print("   start_pos = header_size + tid * thread_stride")
    print("   end_pos = header_size + (tid + 1) * thread_stride")
    
    print("\n2. 境界チェック（行393-394）:")
    print("   elif candidate_pos >= end_pos:  # 担当範囲外")
    print("       break")
    
    print("\n3. 問題: 行が境界をまたぐ場合")
    print("   - 行の開始がend_pos直前にある場合")
    print("   - 行の終了がend_posを超える場合")
    print("   - この行は処理されない可能性")
    
    # 境界をまたぐ行の計算
    print("\n=== 境界をまたぐ行の推定 ===")
    
    # 各スレッド境界での問題
    # thread_strideの最後の位置付近で行が始まる確率
    boundary_zone = estimated_row_size  # 行サイズ分の境界ゾーン
    prob_boundary_row = boundary_zone / thread_stride
    
    print(f"\n境界ゾーン: {boundary_zone} バイト")
    print(f"境界行の確率: {prob_boundary_row:.2%}")
    
    # 全体での影響
    total_thread_boundaries = total_threads * num_chunks * workers_per_chunk
    expected_boundary_rows = total_thread_boundaries * prob_boundary_row
    
    print(f"\n総スレッド境界数: {total_thread_boundaries:,}")
    print(f"期待される境界行数: {expected_boundary_rows:.0f}")
    
    # 実際の欠損との比較
    print(f"\n実際の欠損: {missing_rows}")
    print(f"理論値との差: {abs(expected_boundary_rows - missing_rows):.0f}")
    
    # より詳細な分析
    print("\n=== より詳細な境界分析 ===")
    
    # 各チャンクの最後での処理
    print("\n1. チャンク境界:")
    print(f"   各チャンクの最後で {missing_rows/num_chunks:.1f} 行が欠損")
    
    # 各ワーカーの最後での処理
    print("\n2. ワーカー境界:")
    print(f"   各ワーカーの最後で {missing_rows/(num_chunks*workers_per_chunk):.3f} 行が欠損")
    
    # コード修正の提案
    print("\n=== 修正案 ===")
    print("\n1. 境界処理の改善（行393-394）:")
    print("   # 現在のコード:")
    print("   elif candidate_pos >= end_pos:")
    print("       break")
    print("\n   # 修正案: 行の開始が範囲内なら処理を継続")
    print("   elif candidate_pos >= end_pos:")
    print("       # 行の開始が担当範囲内なら処理を継続")
    print("       if candidate_pos < end_pos:")
    print("           # validate_and_extract_fields_liteで処理")
    print("           pass")
    print("       else:")
    print("           break")
    
    print("\n2. またはコメント（行404-405）の通り:")
    print("   # 境界を越える行も処理する")
    print("   # 行の開始が担当範囲内なら、終了が範囲外でも処理")
    
    # 別の可能性：データ終端
    print("\n=== データ終端の処理 ===")
    
    print("\n可能性: 各チャンクの最後のデータが正しく処理されていない")
    print("- FFFFマーカー（終端）の扱い")
    print("- 最後の不完全な行の扱い")
    
    # thread_strideとデータサイズの関係
    print("\n=== thread_strideの影響 ===")
    
    remainder = data_size_per_chunk % thread_stride
    print(f"\nチャンクサイズ % thread_stride = {remainder} バイト")
    print(f"これは約 {remainder/estimated_row_size:.1f} 行分")
    
    # 最後のスレッドが処理するデータ量
    last_thread_start = (total_threads - 1) * thread_stride + header_size
    last_thread_end = data_size_per_chunk + header_size
    last_thread_size = last_thread_end - last_thread_start
    
    print(f"\n最後のスレッドの処理:")
    print(f"  開始位置: {last_thread_start:,}")
    print(f"  終了位置: {last_thread_end:,}")
    print(f"  データサイズ: {last_thread_size} バイト")
    print(f"  行数: {last_thread_size/estimated_row_size:.1f}")

if __name__ == "__main__":
    analyze_boundary_threads()