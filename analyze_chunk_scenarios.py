#!/usr/bin/env python3
"""
チャンク数を変えた場合の64MB境界問題の発生確率を分析
"""

def analyze_chunk_scenarios():
    """異なるチャンク数での境界問題発生を分析"""
    
    print("チャンク数による64MB境界問題の分析:")
    print("="*80)
    
    # 総データサイズ（customerテーブル）
    total_data_size = 97710505856  # 91GB
    num_workers = 16
    buffer_size = 64 * 1024 * 1024  # 64MB
    
    # 異なるチャンク数でのシナリオ
    chunk_scenarios = [2, 4, 8, 16, 32]
    
    for num_chunks in chunk_scenarios:
        print(f"\n■ {num_chunks}チャンクの場合:")
        print("-" * 60)
        
        chunk_size = total_data_size // num_chunks
        print(f"  1チャンクのサイズ: {chunk_size:,} bytes ({chunk_size/1024/1024/1024:.1f} GB)")
        
        # 各ワーカーの処理量
        data_per_worker = chunk_size // num_workers
        print(f"  1ワーカーあたり: {data_per_worker:,} bytes ({data_per_worker/1024/1024:.1f} MB)")
        
        # バッファフラッシュ回数
        flushes_per_worker = data_per_worker // buffer_size
        remaining_data = data_per_worker % buffer_size
        
        print(f"  フラッシュ回数/ワーカー: {flushes_per_worker}回")
        print(f"  最後のバッファサイズ: {remaining_data/1024/1024:.1f} MB")
        
        # 64MB境界に到達する可能性
        if flushes_per_worker == 0 and remaining_data < buffer_size:
            print(f"  → バッファフラッシュなし（データ量 < 64MB）")
            print(f"  → 64MB境界問題の可能性: 低い")
        else:
            print(f"  → 複数回フラッシュあり")
            print(f"  → 64MB境界問題の可能性: 高い")
        
        # 実際の境界位置を計算
        print(f"\n  予想される境界位置:")
        cumulative_offset = 0
        boundaries_hit = []
        
        for worker in range(min(4, num_workers)):  # 最初の4ワーカーのみ表示
            # ワーカーが書き込むデータ
            if flushes_per_worker > 0:
                # フルバッファを複数回書き込み
                for flush in range(flushes_per_worker):
                    next_offset = cumulative_offset + buffer_size
                    
                    # 64MBの倍数境界をチェック
                    for mb in [64, 128, 192, 256, 320, 384, 448, 512]:
                        boundary = mb * 1024 * 1024
                        if cumulative_offset < boundary <= next_offset:
                            boundaries_hit.append(mb)
                            print(f"    ワーカー{worker}: {mb}MB境界をまたぐ")
                    
                    cumulative_offset = next_offset
            
            # 残りのデータ
            if remaining_data > 0:
                next_offset = cumulative_offset + remaining_data
                
                for mb in [64, 128, 192, 256, 320, 384, 448, 512]:
                    boundary = mb * 1024 * 1024
                    if cumulative_offset < boundary <= next_offset:
                        boundaries_hit.append(mb)
                        print(f"    ワーカー{worker}: {mb}MB境界をまたぐ（最後のバッファ）")
                
                cumulative_offset = next_offset
        
        print(f"\n  境界問題の発生予想: 約{len(boundaries_hit)}箇所")

def calculate_optimal_chunks():
    """最適なチャンク数を計算"""
    
    print("\n\n最適なチャンク数の提案:")
    print("="*80)
    
    total_data_size = 97710505856  # 91GB
    num_workers = 16
    buffer_size = 64 * 1024 * 1024  # 64MB
    
    print(f"総データサイズ: {total_data_size/1024/1024/1024:.1f} GB")
    print(f"ワーカー数: {num_workers}")
    print(f"バッファサイズ: {buffer_size/1024/1024} MB")
    
    print("\n推奨:")
    
    # ワーカーあたり50MB未満になるチャンク数を計算
    target_per_worker = 50 * 1024 * 1024  # 50MB
    min_chunks = int(total_data_size / (target_per_worker * num_workers)) + 1
    
    print(f"1. 境界問題を最小化: {min_chunks}チャンク以上")
    print(f"   → 1ワーカーあたり < 50MB")
    print(f"   → バッファフラッシュなし")
    
    # パフォーマンスとのバランス
    print(f"\n2. パフォーマンスとのバランス: 8-16チャンク")
    print(f"   → 適度なチャンクサイズ")
    print(f"   → 管理オーバーヘッドが少ない")
    
    # 現在の設定
    print(f"\n3. 現在の設定（2チャンク）の問題:")
    print(f"   → 1ワーカーあたり約2.8GB")
    print(f"   → 多数の64MB境界をまたぐ")
    print(f"   → 境界問題が発生しやすい")

if __name__ == "__main__":
    analyze_chunk_scenarios()
    calculate_optimal_chunks()