#!/usr/bin/env python3
"""
元の問題を再分析：32チャンクで500行差分だった
"""

def reanalyze_issue():
    """元の問題を正確に再分析"""
    
    print("=== 問題の再分析 ===\n")
    
    print("【元の状況】")
    print("- 32チャンクで実行")
    print("- 期待: 246,012,324行")
    print("- 実際: 246,011,824行（推定）")
    print("- 差分: 約500行")
    
    print("\n【現在の状況】")
    print("- 80チャンクで実行")
    print("- 期待: 246,012,324行")  
    print("- 実際: 242,386,050行")
    print("- 差分: 3,626,274行")
    
    print("\n【重要な発見】")
    print("32チャンク → 80チャンク で差分が大幅に増加")
    print("500行 → 3,626,274行（7252倍！）")
    
    print("\n=== これは別の問題を示唆 ===")
    
    print("\n【可能性1: チャンク分割の問題】")
    print("- 32チャンクではほぼ正確に分割できていた")
    print("- 80チャンクで何か問題が発生している")
    print("- ページ境界での取りこぼし？")
    
    print("\n【可能性2: GPU処理の問題】")
    print("- チャンク数が増えると累積エラーが増加")
    print("- 各チャンクで小さなエラーが蓄積")
    
    print("\n【可能性3: 空ページの影響】")
    print("- 32チャンクでは空ページの影響が少ない")
    print("- 80チャンクでは空ページによる影響が顕著")
    
    # 計算
    print("\n=== 数値的分析 ===")
    
    # 32チャンクの場合
    chunks_32 = 32
    pages_per_chunk_32 = 4_614_912 // chunks_32  # 144,216ページ/チャンク
    
    # 80チャンクの場合
    chunks_80 = 80
    pages_per_chunk_80 = 4_614_912 // chunks_80  # 57,686ページ/チャンク
    
    print(f"\n32チャンク:")
    print(f"  ページ/チャンク: {pages_per_chunk_32:,}")
    print(f"  差分/チャンク: {500 / chunks_32:.1f}行")
    
    print(f"\n80チャンク:")
    print(f"  ページ/チャンク: {pages_per_chunk_80:,}")
    print(f"  差分/チャンク: {3_626_274 / chunks_80:.1f}行")
    
    # 差分の比率
    error_rate_32 = 500 / 246_012_324
    error_rate_80 = 3_626_274 / 246_012_324
    
    print(f"\nエラー率:")
    print(f"  32チャンク: {error_rate_32:.6%}")
    print(f"  80チャンク: {error_rate_80:.6%}")
    
    print("\n=== 真の原因 ===")
    print("\n空ページ問題ではなく、別の要因がある可能性が高い：")
    print("1. GPUパーサーでの行カウントエラー")
    print("2. Binary COPYのヘッダー処理")
    print("3. チャンク境界での重複または欠損")


def check_gpu_parser_issue():
    """GPUパーサー側の問題を確認"""
    
    print("\n\n=== GPUパーサー側の問題確認 ===")
    
    print("\n【Binary COPYフォーマット】")
    print("- ヘッダー: 11バイト（PGCOPY\\n\\377\\r\\n\\0）")
    print("- 各行: 行長 + フィールド数 + 各フィールド")
    print("- フッター: -1（2バイト）")
    
    print("\n【チャンクごとのBinary COPY】")
    print("- 各チャンクに独立したヘッダー/フッター")
    print("- 80チャンク = 80個のヘッダー/フッター")
    
    print("\n【GPU処理での問題】")
    print("1. ヘッダー/フッターを行としてカウント？")
    print("2. 境界でのオフセット計算エラー？")
    print("3. 無効行の除外処理？")
    
    # 計算
    print("\n【数値的検証】")
    chunks = 80
    header_size = 11
    footer_size = 2
    overhead_per_chunk = header_size + footer_size
    total_overhead = overhead_per_chunk * chunks
    
    print(f"チャンク数: {chunks}")
    print(f"オーバーヘッド/チャンク: {overhead_per_chunk}バイト")
    print(f"総オーバーヘッド: {total_overhead}バイト")
    
    # 1行あたり352バイトとして
    row_size = 352
    overhead_as_rows = total_overhead / row_size
    print(f"\n行数換算: {overhead_as_rows:.1f}行相当")
    print("これだけでは3,626,274行の差分を説明できない")


def propose_solution():
    """真の解決策を提案"""
    
    print("\n\n=== 真の解決策 ===")
    
    print("\n【調査すべきポイント】")
    print("1. 32チャンクでの正確な処理行数を再確認")
    print("2. GPUパーサーのデバッグログを詳細に確認")
    print("3. 各チャンクの実際の処理行数を記録")
    
    print("\n【実験提案】")
    print("1. 32チャンクで再実行して500行差分を確認")
    print("2. 1チャンクだけで全データを処理（比較用）")
    print("3. チャンク境界での重複/欠損をチェック")
    
    print("\n【コード確認箇所】")
    print("- postgres_binary_parser.py: parse_rows_and_fields_lite")
    print("- valid_rows のカウント方法")
    print("- 境界スレッドでの処理")
    
    print("\n【結論】")
    print("空ページ問題は副次的。主要因は別にある。")
    print("32チャンクで500行差分なら、それが本来の問題。")


if __name__ == "__main__":
    reanalyze_issue()
    check_gpu_parser_issue()
    propose_solution()