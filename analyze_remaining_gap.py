#!/usr/bin/env python3
"""
80チャンクで残る1.47%の欠損原因を分析
"""

def analyze_remaining_gap():
    """残りの欠損を分析"""
    
    total_expected = 246_012_324
    processed_80chunks = 242_386_050
    missing = total_expected - processed_80chunks
    missing_rate = missing / total_expected
    
    print("=== 80チャンク処理後の分析 ===")
    print(f"期待される総行数: {total_expected:,}")
    print(f"80チャンクで処理した行数: {processed_80chunks:,}")
    print(f"欠損行数: {missing:,} ({missing_rate:.2%})")
    
    # チャンクあたりの平均
    avg_per_chunk = processed_80chunks / 80
    print(f"\nチャンクあたり平均行数: {avg_per_chunk:,.0f}")
    
    # 欠損の推定原因
    print("\n=== 推定される欠損原因 ===")
    
    # 1. 最後のページの部分的なデータ
    print("1. 最後のページの部分的なデータ")
    print("   - PostgreSQLのページは8KB")
    print("   - 最後のページが完全に埋まっていない可能性")
    
    # 2. 削除されたがVACUUMされていないタプル
    print("\n2. 削除済みタプル（VACUUM未実行）")
    print("   - ctidは削除されたタプルの位置も含む")
    print("   - Binary COPYでは削除済みタプルは含まれない")
    
    # 3. システムカラムやTOAST
    print("\n3. TOASTテーブルのデータ")
    print("   - 大きなデータはTOASTテーブルに格納")
    print("   - ctidベースの読み取りでは含まれない可能性")
    
    # 解決策
    print("\n=== 100%処理を達成する方法 ===")
    print("1. チャンク数を85-90に増やす（短期的）")
    print("2. VACUUM FULLを実行してテーブルを最適化（推奨）")
    print("3. 行数ベースの分割に切り替える（長期的）")
    
    # 追加チャンク数の計算
    target_coverage = 0.995  # 99.5%を目標
    current_coverage = processed_80chunks / total_expected
    required_chunks = int(80 * target_coverage / current_coverage)
    
    print(f"\n99.5%処理に必要なチャンク数: {required_chunks}")


def estimate_page_utilization():
    """ページ利用率を推定"""
    
    total_pages = 4_614_888
    chunks = 80
    pages_per_chunk = total_pages // chunks
    
    # 実際のデータ
    processed_rows = 242_386_050
    expected_rows = 246_012_324
    
    # ページ利用率
    utilization = processed_rows / expected_rows
    
    print(f"\n=== ページ利用率の推定 ===")
    print(f"総ページ数: {total_pages:,}")
    print(f"80チャンクでのページ/チャンク: {pages_per_chunk:,}")
    print(f"実効的なページ利用率: {utilization:.2%}")
    print(f"空ページや部分ページの割合: {(1-utilization):.2%}")


if __name__ == "__main__":
    analyze_remaining_gap()
    estimate_page_utilization()