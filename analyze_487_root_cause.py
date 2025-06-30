#!/usr/bin/env python3
"""
487行欠損の真の原因を調査
"""

def analyze_487_root_cause():
    """487行欠損の根本原因を特定"""
    
    print("=== 487行欠損の詳細分析 ===\n")
    
    # データ
    pg_rows = 246_012_324  # PostgreSQL
    gpu_rows = 246_011_837  # GPU Parser
    missing = 487
    
    print(f"PostgreSQL行数: {pg_rows:,}")
    print(f"GPU処理行数: {gpu_rows:,}")
    print(f"欠損行数: {missing}")
    print(f"欠損率: {missing/pg_rows*100:.6f}%")
    
    # 32チャンクでの分析
    chunks = 32
    workers_per_chunk = 16
    
    print(f"\n処理構成:")
    print(f"- チャンク数: {chunks}")
    print(f"- ワーカー/チャンク: {workers_per_chunk}")
    print(f"- 総ワーカー数: {chunks * workers_per_chunk}")
    
    # 可能性のある原因
    print("\n=== 可能性のある原因 ===")
    
    print("\n1. PostgreSQL COPY BINARYの終端処理")
    print("- COPY BINARYフォーマットは最後に0xFFFF（-1）を含む")
    print("- このマーカーを行として誤カウントしていないか？")
    print("- 各チャンクの最後に終端マーカーがある可能性")
    
    print("\n2. チャンク境界での行分割")
    print("- Rustが8GBチャンクに分割する際、行の途中で切れる")
    print("- 分割された行が正しく処理されていない可能性")
    print("- 16ワーカー × 32チャンク = 512箇所の境界")
    print(f"- 487/512 = {487/512:.2f} ≈ 各境界で約1行欠損？")
    
    print("\n3. 最後のスレッドの処理")
    print("- 各ワーカーの最後のスレッドがデータ終端を正しく処理していない")
    print("- thread_strideの計算で端数が切り捨てられている可能性")
    
    print("\n4. Rust側の行数カウント")
    print("- Rustのpg_fast_copy_single_chunkが行数を過大にカウント")
    print("- 無効な行やメタデータを行としてカウントしている可能性")
    
    print("\n=== 調査すべき箇所 ===")
    
    print("\n1. データ終端の処理（parse_rows_and_fields_lite）:")
    print("- candidate_pos <= -2でbreakしているが、最後の行を処理し忘れていないか")
    print("- FFFFマーカーの扱い")
    
    print("\n2. チャンクサイズとデータサイズの関係:")
    print("- 8GB固定チャンクサイズ vs 実際のデータサイズ")
    print("- 最後のチャンクは8GB未満の可能性")
    
    print("\n3. Rust側の処理:")
    print("- pg_fast_copy_single_chunkの行カウント方法")
    print("- チャンク分割時の境界処理")
    
    # 487の特徴
    print("\n=== 487という数字の特徴 ===")
    print("- 487は素数")
    print("- 512（総ワーカー数）に近い")
    print("- 各処理単位で約1行ずつ欠損している可能性が高い")

if __name__ == "__main__":
    analyze_487_root_cause()