#!/usr/bin/env python3
"""
スレッドあたりの行数制限による487行欠損を分析
"""

import numpy as np

def analyze_thread_row_limits():
    """スレッドの行数制限が原因で処理されない行数を計算"""
    
    # 実際の処理状況
    total_rows = 246011837  # 32チャンクの合計
    total_processed = 246011350  # 実際に処理された行数
    missing_rows = total_rows - total_processed  # 487行
    
    print(f"総行数: {total_rows:,}")
    print(f"処理行数: {total_processed:,}")
    print(f"欠損行数: {missing_rows:,}")
    print("\n" + "="*60 + "\n")
    
    # 各チャンクの平均行数
    num_chunks = 32
    avg_rows_per_chunk = total_rows / num_chunks
    print(f"チャンク数: {num_chunks}")
    print(f"平均行数/チャンク: {avg_rows_per_chunk:,.1f}")
    
    # 並列ワーカー数
    workers_per_chunk = 16
    avg_rows_per_worker = avg_rows_per_chunk / workers_per_chunk
    print(f"並列ワーカー数/チャンク: {workers_per_chunk}")
    print(f"平均行数/ワーカー: {avg_rows_per_worker:,.1f}")
    
    print("\n" + "="*60 + "\n")
    
    # GPUカーネルのパラメータ
    MAX_ROWS_PER_THREAD = 200  # スレッドあたり最大行数
    LOCAL_ARRAY_SIZE = 256     # ローカル配列サイズ
    
    # 推定行サイズ（lineorderテーブル）
    estimated_row_size = 230  # バイト
    
    # データサイズとスレッド計算
    data_size_per_worker = avg_rows_per_worker * estimated_row_size
    print(f"推定行サイズ: {estimated_row_size} バイト")
    print(f"平均データサイズ/ワーカー: {data_size_per_worker:,.0f} バイト ({data_size_per_worker/1024/1024:.1f} MB)")
    
    # GPU計算の例（典型的な設定）
    threads_per_block = 512
    sm_count = 108  # A100のSM数
    blocks = sm_count * 12  # 大容量データの場合
    total_threads = blocks * threads_per_block
    
    print(f"\nGPU設定:")
    print(f"  threads_per_block: {threads_per_block}")
    print(f"  blocks: {blocks}")
    print(f"  total_threads: {total_threads:,}")
    
    # thread_strideの計算
    thread_stride_per_worker = data_size_per_worker / total_threads
    thread_stride_capped = min(thread_stride_per_worker, estimated_row_size * MAX_ROWS_PER_THREAD)
    
    print(f"\nthread_stride計算:")
    print(f"  理論値: {thread_stride_per_worker:,.0f} バイト")
    print(f"  制限後: {thread_stride_capped:,.0f} バイト")
    print(f"  スレッドあたり最大行数: {thread_stride_capped / estimated_row_size:.1f}")
    
    print("\n" + "="*60 + "\n")
    
    # 問題の分析：各スレッドのローカル配列制限
    print("問題分析：ローカル配列の制限")
    print(f"- ローカル配列サイズ: {LOCAL_ARRAY_SIZE} 行")
    print(f"- スレッドあたり最大処理行数: {MAX_ROWS_PER_THREAD} 行")
    
    # 各スレッドが256行を超えて処理する必要がある場合
    if thread_stride_capped / estimated_row_size > LOCAL_ARRAY_SIZE:
        print(f"\n⚠️ 警告: スレッドあたりの行数が配列サイズを超過!")
        excess_ratio = (thread_stride_capped / estimated_row_size) / LOCAL_ARRAY_SIZE
        print(f"  超過率: {excess_ratio:.2f}倍")
    
    # カーネルコードの確認ポイント
    print("\n" + "="*60 + "\n")
    print("カーネルコードの重要箇所:")
    print("\n1. ローカル配列の定義（行375-378）:")
    print("   local_positions = cuda.local.array(256, uint64)")
    print("   local_field_offsets = cuda.local.array((256, 17), uint64)")
    print("   local_field_lengths = cuda.local.array((256, 17), int32)")
    print("   local_count = 0")
    
    print("\n2. 行処理時の配列境界チェック（行406）:")
    print("   if local_count < 256:  # 配列境界チェック")
    print("       local_positions[local_count] = uint64(candidate_pos)")
    print("       local_count += 1")
    
    print("\n3. グローバルメモリへの書き込み（行420-434）:")
    print("   if local_count > 0:")
    print("       base_idx = cuda.atomic.add(row_count, 0, local_count)")
    print("       for i in range(local_count):")
    print("           global_idx = base_idx + i")
    print("           if global_idx < max_rows:")
    
    print("\n" + "="*60 + "\n")
    print("仮説：487行が欠損する理由")
    print("\n1. ローカル配列サイズ制限:")
    print("   - 各スレッドは最大256行しか処理できない")
    print("   - それ以上の行は `if local_count < 256` でスキップされる")
    
    print("\n2. 境界スレッドの処理:")
    print("   - チャンクの境界付近のスレッドが256行を超えるデータを担当")
    print("   - 超過分の行が処理されずにスキップ")
    
    # 欠損行数の推定計算
    print("\n3. 欠損行数の推定:")
    # 32チャンク × 16ワーカー = 512処理単位
    total_processing_units = num_chunks * workers_per_chunk
    missing_per_unit = missing_rows / total_processing_units
    print(f"   - 総処理単位数: {total_processing_units}")
    print(f"   - 平均欠損/処理単位: {missing_per_unit:.2f} 行")
    
    # もしいくつかのスレッドが256行制限に達した場合
    threads_hitting_limit = missing_rows / (256 - MAX_ROWS_PER_THREAD)
    print(f"   - 256行制限に達したスレッド数（推定）: {threads_hitting_limit:.0f}")
    
    print("\n" + "="*60 + "\n")
    print("解決策の提案:")
    print("\n1. ローカル配列サイズを増やす（256 → 512）")
    print("2. MAX_ROWS_PER_THREADを調整（200 → 256）")
    print("3. thread_strideの計算を見直して、各スレッドが256行以下になるよう調整")
    print("4. 複数パスで処理（256行を超える場合は2回目のパスで処理）")

if __name__ == "__main__":
    analyze_thread_row_limits()