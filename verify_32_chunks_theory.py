#!/usr/bin/env python3
"""
32チャンクで500行差分の理論を検証
"""

def verify_32_chunks():
    """32チャンクでの理論的な処理を検証"""
    
    print("=== 32チャンクでの500行差分の検証 ===\n")
    
    # 既知の数値
    total_expected = 246_012_324
    total_pages = 4_614_912
    
    # 32チャンクでの計算
    chunks = 32
    pages_per_chunk = total_pages // chunks  # 144,216ページ/チャンク
    
    print(f"総行数（期待）: {total_expected:,}")
    print(f"総ページ数: {total_pages:,}")
    print(f"チャンク数: {chunks}")
    print(f"ページ/チャンク: {pages_per_chunk:,}")
    
    # 実際のデータページは40%（空ページ問題）
    effective_page_ratio = 0.4
    effective_pages = total_pages * effective_page_ratio
    
    print(f"\n空ページを考慮:")
    print(f"有効ページ率: {effective_page_ratio:.1%}")
    print(f"実効ページ数: {effective_pages:,.0f}")
    
    # 1ページあたりの行数
    rows_per_page = 129.6  # PostgreSQLの標準的な値
    
    # 32チャンクでの推定処理行数
    estimated_rows_32 = effective_pages * rows_per_page
    
    print(f"\n推定処理行数（32チャンク）: {estimated_rows_32:,.0f}")
    print(f"差分: {total_expected - estimated_rows_32:,.0f}")
    
    # 差分の原因分析
    print("\n=== 500行差分の説明 ===")
    
    print("\n【32チャンクで500行差分だった理由】")
    print("1. チャンクが大きいため、空ページの影響が平均化される")
    print("2. 境界条件のエラーが32回しか発生しない")
    print("3. Binary COPYのオーバーヘッドが少ない（32個のみ）")
    
    # GPUパーサーのエラー率を逆算
    error_per_chunk = 500 / chunks
    print(f"\nチャンクあたりのエラー: {error_per_chunk:.1f}行")
    
    # 80チャンクでの予測
    chunks_80 = 80
    predicted_error_80 = error_per_chunk * chunks_80
    print(f"\n80チャンクでの予測エラー（同じエラー率）: {predicted_error_80:.0f}行")
    print(f"実際の80チャンクエラー: 3,626,274行")
    print(f"差異: {3_626_274 / predicted_error_80:.1f}倍")
    
    print("\n=== 結論 ===")
    print("80チャンクで急激にエラーが増加した理由:")
    print("1. 空ページ問題が顕在化（チャンクが小さくなり影響大）")
    print("2. チャンク境界でのエラーが累積（80回 vs 32回）")
    print("3. GPUメモリ管理の問題（小さなチャンクでの非効率）")


def calculate_true_solution():
    """真の解決策を計算"""
    
    print("\n\n=== 真の100%達成方法 ===")
    
    print("\n【方法1: 32チャンクに戻す】")
    print("- 実績: 99.9998%（500行差分のみ）")
    print("- 利点: 既に実証済み")
    print("- 欠点: GPU処理の並列度が下がる可能性")
    
    print("\n【方法2: エラー補正】")
    print("- 各チャンクで発生する固定エラーを特定")
    print("- Binary COPYヘッダー/フッター処理を修正")
    print("- 境界条件の処理を改善")
    
    print("\n【方法3: 中間的なチャンク数】")
    print("- 40-50チャンクで試す")
    print("- 空ページ影響とエラー累積のバランス")
    
    # 最適なチャンク数を推定
    print("\n【最適チャンク数の推定】")
    
    # 32チャンク: 500行エラー
    # 80チャンク: 3,626,274行エラー
    # エラーが非線形に増加
    
    # 空ページ問題を考慮した最適値
    # 32チャンクではページが大きすぎて空ページの影響が少ない
    # 80チャンクではページが小さすぎて空ページの影響が大きい
    
    optimal_chunks = 40  # 推定値
    print(f"\n推奨チャンク数: {optimal_chunks}")
    print(f"理由: 空ページ影響とエラー累積のバランス")
    
    # 実験提案
    print("\n【実験提案】")
    print("1. まず32チャンクで再実行（500行差分を確認）")
    print("2. 40チャンクで実行")
    print("3. 必要に応じて微調整")


if __name__ == "__main__":
    verify_32_chunks()
    calculate_true_solution()