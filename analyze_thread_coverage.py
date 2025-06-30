#!/usr/bin/env python3
"""
スレッドカバー範囲の分析
"""

def analyze_thread_coverage():
    """スレッド数とカバー範囲の関係を分析"""
    
    # 8GBチャンクの例
    chunk_size = 8 * 1024**3  # 8GB
    header_size = 19
    data_size = chunk_size - header_size
    
    # 現在の計算方法（calculate_optimal_grid_sm_aware関数より）
    sm_count = 82
    threads_per_block = 1024
    
    # 推定行サイズ（lineorderテーブル）
    estimated_row_size = 288
    MAX_ROWS_PER_THREAD = 200
    
    # 必要なスレッド数を計算
    estimated_total_rows = data_size // estimated_row_size
    required_threads = (estimated_total_rows + MAX_ROWS_PER_THREAD - 1) // MAX_ROWS_PER_THREAD
    
    # ブロック数を計算
    base_blocks = (required_threads + threads_per_block - 1) // threads_per_block
    
    # 実際に使用されるブロック数（1GB以上の場合）
    target_blocks = max(base_blocks, sm_count * 20)  # 1640
    
    # 総スレッド数
    actual_threads = target_blocks * threads_per_block  # 1,679,360
    
    # thread_stride計算
    thread_stride = (data_size + actual_threads - 1) // actual_threads
    
    # カバー範囲
    covered_size = actual_threads * thread_stride
    
    print("=== スレッドカバー範囲分析 ===")
    print(f"データサイズ: {data_size:,} bytes")
    print(f"推定行数: {estimated_total_rows:,}")
    print(f"必要スレッド数: {required_threads:,}")
    print(f"実際のスレッド数: {actual_threads:,}")
    print(f"thread_stride: {thread_stride:,} bytes")
    print(f"カバー範囲: {covered_size:,} bytes")
    print(f"不足: {data_size - covered_size:,} bytes")
    
    # 不足分の行数
    missing_bytes = data_size - covered_size
    if missing_bytes > 0:
        missing_rows = missing_bytes // estimated_row_size
        print(f"不足行数（推定）: {missing_rows} 行")
        
        # 修正案
        print("\n=== 修正案 ===")
        
        # 案1: thread_strideを切り上げ
        corrected_stride = (data_size + actual_threads - 1) // actual_threads
        if data_size % actual_threads != 0:
            corrected_stride += 1
        print(f"案1: thread_strideを切り上げ: {corrected_stride}")
        
        # 案2: 必要なスレッド数を再計算
        exact_threads_needed = (data_size + thread_stride - 1) // thread_stride
        print(f"案2: 必要な正確なスレッド数: {exact_threads_needed:,}")
        
        # 案3: 最後のスレッドの担当範囲を拡張
        last_thread_extra = data_size - (actual_threads - 1) * thread_stride
        print(f"案3: 最後のスレッドの担当サイズ: {last_thread_extra:,} bytes")

if __name__ == "__main__":
    analyze_thread_coverage()