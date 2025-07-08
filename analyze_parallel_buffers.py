#!/usr/bin/env python3
"""
16並列ワーカーのバッファ境界を分析
"""

def analyze_parallel_buffers():
    """16並列ワーカーの64MBバッファ境界を計算"""
    
    print("16並列ワーカーのバッファ境界分析:")
    print("="*80)
    
    buffer_size = 64 * 1024 * 1024  # 64MB
    num_workers = 16
    
    print(f"ワーカー数: {num_workers}")
    print(f"各ワーカーのバッファサイズ: {buffer_size:,} bytes ({buffer_size/1024/1024}MB)")
    
    # 理論的な境界位置
    print("\n理論的なバッファ境界位置:")
    boundaries = []
    for i in range(1, 20):  # 最初の20境界
        boundary = i * buffer_size
        boundaries.append(boundary)
        print(f"  境界{i:2d}: 0x{boundary:08X} ({boundary/1024/1024:6.1f}MB)")
    
    # 実際に問題が発生した境界
    actual_boundaries = [
        0x04000000,  # 64MB
        0x08000000,  # 128MB
    ]
    
    print("\n実際に問題が発生した境界:")
    for boundary in actual_boundaries:
        boundary_num = boundary // buffer_size
        print(f"  境界{boundary_num}: 0x{boundary:08X} ({boundary/1024/1024}MB)")
    
    # なぜ2箇所だけか
    print("\nなぜ2箇所だけで問題が発生するのか:")
    print("1. 各ワーカーは独立してデータを処理")
    print("2. ワーカーのバッファ境界は必ずしも64MBの倍数にならない")
    print("3. たまたま2つのワーカーが64MB/128MB境界で行を分断した")
    
    # ワーカーの処理範囲を推定
    print("\nワーカーの処理範囲（推定）:")
    total_size = 848_104_872  # 実際のチャンクサイズ（約848MB）
    chunk_size = total_size // num_workers
    
    for i in range(num_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_workers - 1 else total_size
        
        # このワーカーのバッファ境界
        worker_boundaries = []
        pos = start
        while pos < end:
            pos += buffer_size
            if pos < end:
                worker_boundaries.append(pos)
        
        if worker_boundaries:
            print(f"\n  ワーカー{i:2d}: {start:,} - {end:,} ({(end-start)/1024/1024:.1f}MB)")
            for b in worker_boundaries:
                print(f"    バッファ境界: 0x{b:08X} ({b/1024/1024:.1f}MB)")
                if b in actual_boundaries:
                    print(f"    → 問題発生！")

def calculate_probability():
    """境界で行が分断される確率を計算"""
    
    print("\n\n行が境界で分断される確率:")
    print("="*80)
    
    avg_row_size = 100  # 平均行サイズ（バイト）
    buffer_size = 64 * 1024 * 1024
    total_rows = 6_015_000  # チャンクあたりの行数
    
    # 境界付近の行数
    rows_near_boundary = 2  # 境界前後の2行が影響を受ける可能性
    
    # 16ワーカーそれぞれが複数回バッファをフラッシュ
    bytes_per_worker = 848_104_872 // 16  # 約53MB/ワーカー
    flushes_per_worker = max(1, bytes_per_worker // buffer_size)  # 64MBバッファなので0-1回
    
    # 実際には各ワーカーがデータを書き込むたびに境界が発生
    # 16ワーカーが並列で動作し、それぞれが独自のオフセットで書き込む
    total_boundaries = 16  # 各ワーカーが最低1回は書き込む
    
    probability = (rows_near_boundary * total_boundaries) / total_rows
    
    print(f"平均行サイズ: {avg_row_size} bytes")
    print(f"総行数: {total_rows:,}")
    print(f"ワーカーあたりのフラッシュ回数: 約{flushes_per_worker}")
    print(f"総境界数: 約{total_boundaries}")
    print(f"境界で分断される確率: 約{probability*100:.2f}%")
    print(f"期待される分断行数: 約{int(total_rows * probability)}行")

if __name__ == "__main__":
    analyze_parallel_buffers()
    calculate_probability()