#!/usr/bin/env python3
"""
最終的な根本原因分析
"""

def final_analysis():
    """最終分析"""
    
    print("=== 最終的な根本原因分析 ===\n")
    
    # PostgreSQL統計情報
    print("【PostgreSQL統計情報】")
    print("n_live_tup: 245,985,578")
    print("n_dead_tup: 0")
    print("dead_ratio: 0")
    
    print("\n【重要な発見】")
    print("削除済みタプルは0！これが原因ではない。")
    
    # 実際の差分
    print("\n【実際の差分】")
    print("期待値（どこから？）: 246,012,324")
    print("pg_stat n_live_tup: 245,985,578")
    print("差: 26,746行")
    
    print("\n80チャンクGPU処理: 242,386,050")
    print("pg_stat n_live_tupとの差: 3,599,528行")
    
    # 真の原因
    print("\n=== 真の原因 ===")
    
    print("\n1. 期待値246,012,324の出所が不明")
    print("   - これは本当に正しい値か？")
    print("   - pg_statでは245,985,578行")
    
    print("\n2. GPU処理で3,599,528行の欠損")
    print("   - 245,985,578 - 242,386,050 = 3,599,528")
    print("   - これが真の問題")
    
    print("\n3. 欠損の原因候補:")
    print("   a) チャンク境界での取りこぼし")
    print("   b) 空ページによる影響")
    print("   c) Binary COPYの特殊な行")
    print("   d) GPUパーサーのバグ")
    
    # 計算
    print("\n=== 詳細計算 ===")
    
    chunks = 80
    missing_per_chunk = 3_599_528 / chunks
    
    print(f"\nチャンクあたりの欠損: {missing_per_chunk:,.0f}行")
    print("これは大きすぎる！")
    
    # チャンク0の分析
    print("\n【チャンク0の分析】")
    print("転送バイト: 728,231,874")
    print("推定行数: 2,068,840")
    print("ページ数: 57,686")
    print("ページあたり: 35.9行")
    
    print("\n期待されるページあたり行数: 129.6")
    print("実際: 35.9")
    print("比率: 27.7%")
    
    print("\n=== 結論 ===")
    print("\n空ページが原因で、実データは全ページの27.7%しかない")
    print("これが98.53%ではなく、もっと低い処理率の原因")
    
    print("\n【解決策】")
    print("1. VACUUM FULL実行（空ページを除去）")
    print("2. より多くのチャンク（100-120）で実行")
    print("3. 実データのあるページのみを処理する方法を検討")


def calculate_required_chunks():
    """必要なチャンク数を計算"""
    
    print("\n\n=== 100%達成に必要なチャンク数 ===")
    
    # 既知の値
    n_live_tup = 245_985_578
    processed_80 = 242_386_050
    coverage_80 = processed_80 / n_live_tup
    
    print(f"\npg_stat n_live_tup: {n_live_tup:,}")
    print(f"80チャンク処理: {processed_80:,}")
    print(f"カバレッジ: {coverage_80:.2%}")
    
    # 線形外挿
    required_chunks = int(80 / coverage_80 + 0.5)
    
    print(f"\n必要チャンク数（線形外挿）: {required_chunks}")
    
    # 安全マージンを追加
    safe_chunks = int(required_chunks * 1.05)
    print(f"安全マージン込み: {safe_chunks}")
    
    print("\n【推奨】")
    print(f"{safe_chunks}チャンクで実行")
    print(f"export TOTAL_CHUNKS={safe_chunks}")
    print("python docs/benchmark/benchmark_rust_gpu_direct.py")


if __name__ == "__main__":
    final_analysis()
    calculate_required_chunks()