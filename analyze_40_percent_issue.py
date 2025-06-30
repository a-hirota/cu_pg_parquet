#!/usr/bin/env python3
"""
32チャンクで40%しか処理できない問題を分析
"""

def analyze_processing_issue():
    """処理率問題を分析"""
    
    # 既知の情報
    total_expected_rows = 246_012_324
    processed_rows_32chunks = 98_397_131
    processing_rate = processed_rows_32chunks / total_expected_rows
    
    print("=== 32チャンク処理の分析 ===")
    print(f"期待される総行数: {total_expected_rows:,}")
    print(f"実際の処理行数: {processed_rows_32chunks:,}")
    print(f"処理率: {processing_rate:.2%}")
    print(f"欠損行数: {total_expected_rows - processed_rows_32chunks:,}")
    
    # ページ計算の確認
    print("\n=== ページ計算の確認 ===")
    total_pages = 1_897_885  # 実際のデータページ数
    chunks = 32
    pages_per_chunk = total_pages // chunks
    
    print(f"総ページ数: {total_pages:,}")
    print(f"チャンク数: {chunks}")
    print(f"チャンクあたりページ数: {pages_per_chunk:,}")
    
    # 1チャンクあたりの行数
    rows_per_chunk = processed_rows_32chunks / chunks
    print(f"\n実際の1チャンクあたり行数: {rows_per_chunk:,.0f}")
    
    # 推定される問題
    print("\n=== 推定される問題 ===")
    
    # 1. 空のページが多い可能性
    avg_rows_per_page = total_expected_rows / total_pages
    print(f"1ページあたりの平均行数（期待）: {avg_rows_per_page:.1f}")
    
    actual_rows_per_page = processed_rows_32chunks / total_pages
    print(f"1ページあたりの平均行数（実際）: {actual_rows_per_page:.1f}")
    
    # 2. ページが均等に分布していない可能性
    print("\n可能性のある原因:")
    print("1. PostgreSQLのVACUUM等により、多くの空ページが存在")
    print("2. データが前方のページに偏っている")
    print("3. ctidベースのページ読み取りで、削除されたタプルのページも含まれている")
    
    # 解決策の提案
    print("\n=== 推奨される解決策 ===")
    print("1. VACUUMを実行してテーブルを最適化")
    print("2. ページ数ではなく行数ベースでチャンク分割")
    print("3. 実際のデータ分布を事前に分析してチャンク境界を決定")


def calculate_optimal_chunks():
    """最適なチャンク数を計算"""
    total_rows = 246_012_324
    rows_per_chunk_target = 7_700_000  # 約7.7M行/チャンク
    
    optimal_chunks = total_rows // rows_per_chunk_target
    print(f"\n=== 最適なチャンク数の計算 ===")
    print(f"目標行数/チャンク: {rows_per_chunk_target:,}")
    print(f"推奨チャンク数: {optimal_chunks}")
    

if __name__ == "__main__":
    analyze_processing_issue()
    calculate_optimal_chunks()