#!/usr/bin/env python3
"""
正しい分析：count(*)が真実
"""

def correct_analysis():
    """正しい分析"""
    
    print("=== 訂正：正しい行数分析 ===\n")
    
    print("【正しい行数】")
    print("SELECT COUNT(*) FROM lineorder: 246,012,324行")
    print("pg_stat n_live_tup: 245,985,578行")
    print("差: 26,746行")
    
    print("\n【pg_statが少ない理由】")
    print("- pg_statは推定値（統計情報）")
    print("- COUNT(*)は実際の行数（真実）")
    print("- 統計情報の更新タイミングのずれ")
    
    print("\n【実際の欠損】")
    print("期待値（COUNT(*)）: 246,012,324行")
    print("80チャンク処理: 242,386,050行")
    print("欠損: 3,626,274行（1.47%）")
    
    print("\n=== 元の問題に戻る ===")
    
    print("\n32チャンクで500行差分は正確だった可能性が高い")
    print("80チャンクで3,626,274行差分に増加")
    
    # 差分の増加率
    diff_32 = 500
    diff_80 = 3_626_274
    increase_factor = diff_80 / diff_32
    
    print(f"\n差分の増加: {increase_factor:.0f}倍")
    
    print("\n【原因】")
    print("1. チャンク数増加による累積エラー")
    print("2. 空ページ問題の顕在化")
    print("3. チャンク境界での取りこぼし")
    
    # 必要なチャンク数の再計算
    print("\n=== 100%達成への計算 ===")
    
    total_correct = 246_012_324
    processed_80 = 242_386_050
    coverage = processed_80 / total_correct
    
    print(f"\n正しいカバレッジ: {coverage:.4%}")
    
    # 線形外挿
    required_chunks = 80 / coverage
    print(f"必要チャンク数: {required_chunks:.1f}")
    print(f"切り上げ: {int(required_chunks + 1)}")
    
    print("\n【推奨アクション】")
    print("1. 32チャンクで再実行（ほぼ100%のはず）")
    print("2. または82チャンクで実行")
    print("3. 根本解決にはVACUUM FULL")


if __name__ == "__main__":
    correct_analysis()