#!/usr/bin/env python3
"""
32チャンクでのページ境界を分析するスクリプト
"""

def analyze_32_chunk_pages(total_pages=1897885):
    """32チャンクでのページ分割を分析"""
    print(f"総ページ数: {total_pages:,}")
    print(f"チャンクあたりページ数（切り捨て）: {total_pages // 32:,}")
    print(f"余りページ数: {total_pages % 32}")
    print()
    
    pages_per_chunk = total_pages // 32
    
    print("チャンク別ページ割り当て:")
    print("-" * 60)
    
    total_assigned = 0
    for i in range(32):
        start = i * pages_per_chunk
        if i == 31:
            end = total_pages
        else:
            end = (i + 1) * pages_per_chunk
        
        page_count = end - start
        total_assigned += page_count
        
        # 問題のあるチャンク周辺を強調表示
        if i >= 24:
            print(f"チャンク{i:2d}: ページ {start:,} - {end:,} ({page_count:,}ページ) ★")
        else:
            print(f"チャンク{i:2d}: ページ {start:,} - {end:,} ({page_count:,}ページ)")
    
    print("-" * 60)
    print(f"割り当てられた総ページ数: {total_assigned:,}")
    print(f"カバレッジ: {total_assigned / total_pages * 100:.2f}%")
    
    # 問題のあるチャンク26の詳細
    print("\n問題のチャンク26の詳細:")
    chunk_26_start = 26 * pages_per_chunk
    chunk_26_end = 27 * pages_per_chunk
    print(f"チャンク26: ページ {chunk_26_start:,} - {chunk_26_end:,}")
    print(f"これは最大ページ{total_pages:,}を超えていますか？ {chunk_26_start >= total_pages}")


def check_overlap_and_gaps():
    """重複やギャップがないか確認"""
    total_pages = 1897885
    pages_per_chunk = total_pages // 32
    
    print("\n重複・ギャップチェック:")
    print("-" * 60)
    
    prev_end = 0
    for i in range(32):
        start = i * pages_per_chunk
        if i == 31:
            end = total_pages
        else:
            end = (i + 1) * pages_per_chunk
        
        if i > 0 and start != prev_end:
            print(f"⚠️  チャンク{i-1}と{i}の間にギャップ: {prev_end} != {start}")
        
        prev_end = end
    
    print("チェック完了")


def calculate_correct_distribution():
    """正しい分散方法を計算"""
    total_pages = 1897885
    
    print("\n推奨される分散方法:")
    print("-" * 60)
    
    # 方法1: 均等分割 + 余りを最後のチャンクに
    base_pages = total_pages // 32
    remainder = total_pages % 32
    
    print(f"方法1: 基本{base_pages:,}ページ × 31チャンク + 最後のチャンク{base_pages + remainder:,}ページ")
    
    # 方法2: 余りを前方のチャンクに分散
    print(f"\n方法2: 前{remainder}チャンクに{base_pages+1:,}ページ、残りに{base_pages:,}ページ")
    
    # 実際の分布を表示
    print("\n方法2の実装例:")
    for i in range(32):
        if i < remainder:
            pages = base_pages + 1
            start = i * (base_pages + 1)
            end = start + pages
        else:
            pages = base_pages
            start = remainder * (base_pages + 1) + (i - remainder) * base_pages
            end = start + pages
        
        if i >= 24:
            print(f"チャンク{i:2d}: ページ {start:,} - {end:,} ({pages:,}ページ) ★")


if __name__ == "__main__":
    analyze_32_chunk_pages()
    check_overlap_and_gaps()
    calculate_correct_distribution()