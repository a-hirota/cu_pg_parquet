#!/usr/bin/env python3
"""
16チャンクで40%の行が欠損する原因を分析
"""

def main():
    # データ
    postgres_total = 246_012_324
    rust_estimated = 249_259_856  # 16チャンク * 15,578,741行
    gpu_processed = 148_915_673
    
    print("=== 16チャンク処理の分析 ===")
    print(f"PostgreSQL総行数: {postgres_total:,}")
    print(f"Rust推定総行数: {rust_estimated:,} ({rust_estimated/postgres_total*100:.1f}%)")
    print(f"GPU処理行数: {gpu_processed:,} ({gpu_processed/postgres_total*100:.1f}%)")
    print(f"\n欠損行数: {postgres_total - gpu_processed:,} ({(postgres_total - gpu_processed)/postgres_total*100:.1f}%)")
    
    # 各チャンクの処理率を計算
    print(f"\n=== チャンクごとの処理効率 ===")
    # 実際のチャンクごとの処理行数
    chunk_rows = [
        9_328_653, 9_275_254, 9_275_243, 9_275_248, 9_275_274,
        9_275_248, 9_275_266, 9_275_268, 9_275_237, 9_275_235,
        9_329_423, 9_356_070, 9_356_048, 9_356_079, 9_356_075, 9_356_052
    ]
    
    rust_per_chunk = 15_578_741
    
    for i, rows in enumerate(chunk_rows):
        efficiency = rows / rust_per_chunk * 100
        print(f"チャンク{i:2d}: {rows:,} 行 (推定の {efficiency:.1f}%)")
    
    avg_efficiency = sum(chunk_rows) / len(chunk_rows) / rust_per_chunk * 100
    print(f"\n平均処理効率: {avg_efficiency:.1f}%")
    
    print("\n=== 考えられる原因 ===")
    print("1. GPUパーサーが一部の行を検出できない")
    print("2. チャンクの境界で行が欠落")
    print("3. 特定のデータパターンで検出失敗")
    print("4. max_rows制限による早期終了")

if __name__ == "__main__":
    main()