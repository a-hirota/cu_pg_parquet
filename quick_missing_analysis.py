#!/usr/bin/env python3
"""
欠損行の原因を素早く特定
"""
import os

def quick_analysis():
    """素早い分析"""
    
    print("=== 欠損行3,626,274の原因分析 ===\n")
    
    # 既知の情報
    total_expected = 246_012_324
    processed_80chunks = 242_386_050
    missing = total_expected - processed_80chunks
    
    print(f"期待される行数: {total_expected:,}")
    print(f"80チャンクで処理: {processed_80chunks:,}")
    print(f"欠損: {missing:,} ({missing/total_expected*100:.2f}%)")
    
    # 重要な発見
    print("\n【重要な発見】")
    print("1. 最大ctidは(4614887,2)で、80チャンクの範囲内")
    print("2. チャンク79は欠損なし（調査済み）")
    print("3. pg_relation_sizeは4,614,888ページを返す")
    
    # 最も可能性の高い原因
    print("\n【最も可能性の高い原因】")
    print("GPUパーサー側での行数カウントの問題:")
    print("- Binary COPYのヘッダー/フッター処理")
    print("- 無効な行のスキップ")
    print("- parse_rows_and_fields_liteでの境界条件")
    
    # test_80chunk.logから情報を抽出
    print("\n【80チャンクログの分析】")
    print("各チャンクで「NO SORT」処理された行数を確認...")
    
    # 実際の処理行数を計算
    chunks_processed = []
    total_from_log = 0
    
    # 手動で確認した値（test_80chunk.logから）
    sample_chunks = [
        (1, 3_172_730),
        (2, 3_172_730),
        (3, 3_172_730),
        # ... 他のチャンクも同様
    ]
    
    print("\nサンプルチャンクの処理行数:")
    for chunk_id, rows in sample_chunks[:5]:
        print(f"  チャンク{chunk_id}: {rows:,}行")
    
    # 推定
    avg_per_chunk = 3_172_730
    estimated_total = avg_per_chunk * 80
    print(f"\n推定総処理行数（3,172,730 × 80）: {estimated_total:,}")
    print(f"実際の処理行数: {processed_80chunks:,}")
    print(f"差分: {estimated_total - processed_80chunks:,}")
    
    # 解決策
    print("\n=== 100%達成への確実な道 ===")
    
    print("\n【即効性のある解決策】")
    print("1. 82チャンクで実行")
    print("   export TOTAL_CHUNKS=82")
    print("   確実に全ページをカバー")
    
    print("\n【根本的な解決策】")
    print("2. 最終チャンクの拡張実装")
    print("   main_single_chunk.rs の修正:")
    print("   ```rust")
    print("   let chunk_end_page = if chunk_id == total_chunks - 1 {")
    print("       u32::MAX  // 最終チャンクは無限大まで")
    print("   } else {")
    print("       (chunk_id + 1) as u32 * pages_per_chunk")
    print("   };")
    print("   ```")
    
    print("\n【検証方法】")
    print("3. Parquetファイルの行数を合計")
    print("   ```python")
    print("   import pyarrow.parquet as pq")
    print("   total = sum(pq.read_metadata(f'output_{i}.parquet').num_rows for i in range(80))")
    print("   ```")


if __name__ == "__main__":
    quick_analysis()