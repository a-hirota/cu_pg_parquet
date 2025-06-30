#!/usr/bin/env python3
"""
82チャンクでの総行数を計算
"""

def calculate_82_chunks_total():
    """82チャンクでの推定総行数"""
    
    print("=== 82チャンクでの推定総行数計算 ===\n")
    
    # 80チャンクでの実績
    total_80_chunks = 242_386_050
    avg_per_chunk_80 = total_80_chunks / 80  # 3,029,825行/チャンク
    
    # チャンク81のテスト結果
    chunk_81_bytes = 710_582_622
    chunk_81_rows = chunk_81_bytes // 352  # 約2,018,130行
    
    print(f"80チャンクでの実績:")
    print(f"  総行数: {total_80_chunks:,}")
    print(f"  平均行数/チャンク: {avg_per_chunk_80:,.0f}")
    
    print(f"\nチャンク81（82チャンクの最終）:")
    print(f"  転送バイト数: {chunk_81_bytes:,}")
    print(f"  推定行数: {chunk_81_rows:,}")
    
    # 82チャンクでの推定
    # 仮定: チャンク0-79は80チャンク時と同じ比率で縮小
    # チャンク80,81が新規追加
    
    # ページ数の比率から計算
    pages_80_chunks = 4_614_912 // 80  # 57,686ページ/チャンク
    pages_82_chunks = 4_614_912 // 82  # 56,279ページ/チャンク
    ratio = pages_82_chunks / pages_80_chunks  # 0.9756
    
    print(f"\nページ数の変化:")
    print(f"  80チャンク: {pages_80_chunks:,}ページ/チャンク")
    print(f"  82チャンク: {pages_82_chunks:,}ページ/チャンク")
    print(f"  比率: {ratio:.4f}")
    
    # 推定計算
    # チャンク0-79の行数は比率で縮小
    rows_0_to_79 = total_80_chunks * ratio * (80 / 82)
    
    # チャンク80,81の推定（チャンク81と同程度と仮定）
    rows_80_to_81 = chunk_81_rows * 2
    
    estimated_total = rows_0_to_79 + rows_80_to_81
    
    print(f"\n推定行数:")
    print(f"  チャンク0-79: {rows_0_to_79:,.0f}")
    print(f"  チャンク80-81: {rows_80_to_81:,.0f}")
    print(f"  推定総行数: {estimated_total:,.0f}")
    
    # 目標との比較
    target = 246_012_324
    coverage = estimated_total / target * 100
    
    print(f"\n目標行数: {target:,}")
    print(f"カバレッジ: {coverage:.2f}%")
    
    if coverage >= 99.5:
        print("\n✅ 82チャンクで100%達成の可能性が高い")
    else:
        print(f"\n⚠️ さらに{target - estimated_total:,.0f}行不足")
        
    # より正確な推定
    print("\n=== より正確な推定（空ページ分布を考慮） ===")
    
    # 最後のページ付近は空ページが多い
    # 中間のチャンクは平均的な密度
    # 前半のチャンクも平均的な密度
    
    # 80チャンクの実績から平均密度を計算
    avg_rows_per_page = total_80_chunks / (80 * pages_80_chunks) * 129.6  # ページあたり期待行数で補正
    
    print(f"\n実効的な行密度: {avg_rows_per_page:.1f}行/ページ")
    
    # 82チャンクでの推定
    total_pages_82 = 82 * pages_82_chunks
    estimated_rows_82 = total_pages_82 * avg_rows_per_page
    
    print(f"82チャンクの総ページ数: {total_pages_82:,}")
    print(f"推定総行数（密度ベース）: {estimated_rows_82:,.0f}")
    print(f"カバレッジ: {estimated_rows_82 / target * 100:.2f}%")


if __name__ == "__main__":
    calculate_82_chunks_total()