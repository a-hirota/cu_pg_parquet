#!/usr/bin/env python3
"""
32チャンクと80チャンクの比較分析
"""

def compare_chunks():
    """チャンク数による違いを分析"""
    
    print("=== 32チャンクと80チャンクの比較 ===\n")
    
    # テスト結果
    print("【チャンク0の転送バイト数】")
    print("32チャンク: 1,820,590,200 bytes (1.70 GB)")
    print("80チャンク: 728,231,874 bytes (0.68 GB)")
    
    # 推定行数（1行352バイト）
    rows_32 = 1_820_590_200 // 352
    rows_80 = 728_231_874 // 352
    
    print(f"\n【チャンク0の推定行数】")
    print(f"32チャンク: {rows_32:,}行")
    print(f"80チャンク: {rows_80:,}行")
    print(f"比率: {rows_32 / rows_80:.1f}倍")
    
    # 全体の推定
    total_32 = rows_32 * 32
    total_80 = rows_80 * 80
    
    print(f"\n【全体の推定行数（単純計算）】")
    print(f"32チャンク: {total_32:,}行")
    print(f"80チャンク: {total_80:,}行")
    
    # 期待値との比較
    expected = 246_012_324
    print(f"\n【期待値との比較】")
    print(f"期待値: {expected:,}行")
    print(f"32チャンク推定との差: {expected - total_32:,}行")
    print(f"80チャンク推定との差: {expected - total_80:,}行")
    
    # 重要な発見
    print("\n=== 重要な発見 ===")
    print("\n32チャンクの場合:")
    print("- 単純計算で165,508,192行（期待値の67.3%）")
    print("- 空ページの影響で実際にはもっと少ない可能性")
    
    print("\n80チャンクの場合:")
    print("- 単純計算で165,507,200行（期待値の67.3%）")
    print("- ほぼ同じ行数！")
    
    print("\n【結論】")
    print("1. チャンク数に関わらず、処理される行数がほぼ同じ")
    print("2. これは空ページ問題が主要因ではない証拠")
    print("3. 別の要因（GPUパーサーの問題）が存在する")


def analyze_real_issue():
    """真の問題を分析"""
    
    print("\n\n=== 真の問題の分析 ===")
    
    # 実際の80チャンク結果
    actual_80 = 242_386_050
    expected = 246_012_324
    
    print(f"【実際の80チャンク結果】")
    print(f"処理行数: {actual_80:,}")
    print(f"期待値: {expected:,}")
    print(f"達成率: {actual_80 / expected * 100:.2f}%")
    
    # 単純推定との差
    simple_estimate = 165_507_200
    print(f"\n【単純推定との差】")
    print(f"単純推定: {simple_estimate:,}行（67.3%）")
    print(f"実際: {actual_80:,}行（98.5%）")
    print(f"差: {actual_80 - simple_estimate:,}行")
    
    print("\n【この差（76,878,850行）はどこから？】")
    print("1. 後半のチャンクが前半より多い？")
    print("2. GPUパーサーが何か追加で処理している？")
    print("3. 重複カウント？")
    
    # チャンクあたりの平均
    avg_per_chunk = actual_80 / 80
    print(f"\n【実際の平均行数/チャンク】")
    print(f"{avg_per_chunk:,.0f}行")
    print(f"チャンク0の実測: {2_068_840:,}行")
    print(f"差: {avg_per_chunk - 2_068_840:,.0f}行")
    
    print("\n=== 仮説 ===")
    print("\n【仮説1: チャンク分布の不均一】")
    print("- 前半のチャンクは空ページが多い")
    print("- 後半のチャンクはデータが密")
    print("- 平均すると98.5%")
    
    print("\n【仮説2: GPUパーサーの境界処理】")
    print("- 各チャンクで境界付近の行を重複処理？")
    print("- または欠損している？")
    
    print("\n【検証方法】")
    print("1. 各チャンクの実際の処理行数を記録")
    print("2. 合計が期待値と一致するか確認")
    print("3. 重複や欠損を特定")


if __name__ == "__main__":
    compare_chunks()
    analyze_real_issue()