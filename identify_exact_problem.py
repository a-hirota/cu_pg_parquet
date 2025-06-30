#!/usr/bin/env python3
"""
32チャンクでも80チャンクでも同じ行数になる根本原因を特定
"""

def identify_exact_problem():
    """正確な問題を特定"""
    
    print("=== 根本原因の特定 ===\n")
    
    print("【重要な発見】")
    print("32チャンク: 242,386,031行")
    print("80チャンク: 242,386,050行")
    print("差: わずか19行！")
    
    print("\n目標: 246,012,324行")
    print("欠損: 3,626,293行（1.47%）")
    
    print("\n=== この事実が示すこと ===")
    print("チャンク数に関係なく、ほぼ同じ行数が処理される")
    print("→ チャンク分割の問題ではない")
    print("→ データ自体の問題")
    
    # 処理率の一貫性
    print("\n【処理率の一貫性】")
    coverage = 242_386_031 / 246_012_324
    print(f"処理率: {coverage:.4%}")
    print("常に98.53%前後")
    
    print("\n=== 可能性のある原因 ===")
    
    print("\n1. PostgreSQL COPY BINARYの仕様")
    print("   - 特定の行が出力されない条件がある？")
    print("   - システムカラムや特殊なデータ？")
    
    print("\n2. データの物理的な問題")
    print("   - 破損したページ")
    print("   - アクセスできないデータ")
    print("   - 特殊な行の存在")
    
    print("\n3. ctidベースの読み取りの限界")
    print("   - ctidでアクセスできない行が存在？")
    print("   - TOASTテーブルのデータ？")
    
    # 検証方法
    print("\n=== 検証方法 ===")
    
    print("\n【方法1: 全データ一括取得】")
    print("```bash")
    print("# 1チャンクで全データを取得")
    print("export TOTAL_CHUNKS=1")
    print("export CHUNK_ID=0")
    print("./rust_bench_optimized/target/release/pg_fast_copy_single_chunk")
    print("```")
    
    print("\n【方法2: 別の方法でデータ取得】")
    print("```sql")
    print("-- ctid制限なしでCOPY")
    print("COPY lineorder TO '/tmp/lineorder_full.bin' (FORMAT BINARY);")
    print("```")
    
    print("\n【方法3: 欠損行の特定】")
    print("```sql")
    print("-- Parquetに存在しない行を見つける")
    print("-- （実装が必要）")
    print("```")


def analyze_pattern():
    """パターンを分析"""
    
    print("\n\n=== 32チャンクのパターン分析 ===")
    
    # 32チャンクのデータ
    chunks_32 = [
        (0, 7_774_761),
        (1, 7_540_881),
        (2, 7_444_509),
        # ... 中略
        (21, 7_774_760),
        (22, 7_774_729),
        # ... 最後まで
    ]
    
    print("\n【チャンク分布の特徴】")
    print("前半（0-20）: 主に7,444,xxx行")
    print("後半（21-31）: 主に7,774,xxx行")
    print("\n→ 後半のチャンクの方が行数が多い")
    print("→ データ分布の偏り")
    
    # 1行あたりのバイト数
    print("\n【重要な観察】")
    print("実際のバイト/行: 234.2")
    print("想定バイト/行: 352")
    print("比率: 66.5%")
    
    print("\n→ 実際の行サイズが想定より小さい")
    print("→ または何かが欠けている")


def propose_solution():
    """解決策を提案"""
    
    print("\n\n=== 100%達成への解決策 ===")
    
    print("\n【即座に試すべきこと】")
    
    print("\n1. ctid制限を外す")
    print("```rust")
    print("// main_single_chunk.rs を修正")
    print("let copy_query = format!(")
    print('    "COPY {} TO STDOUT (FORMAT BINARY)",')
    print("    table_name")
    print(");")
    print("```")
    
    print("\n2. 1チャンクで全データ取得")
    print("```bash")
    print("export TOTAL_CHUNKS=1")
    print("python cu_pg_parquet.py --table lineorder --chunks 1")
    print("```")
    
    print("\n3. 欠損行の特性を調査")
    print("```sql")
    print("-- 最初と最後のctidを確認")
    print("SELECT MIN(ctid), MAX(ctid) FROM lineorder;")
    print("")
    print("-- ページごとの行数分布")
    print("SELECT (ctid::text::point)[0]::int as page,")
    print("       COUNT(*) as rows")
    print("FROM lineorder")
    print("GROUP BY page")
    print("ORDER BY page")
    print("LIMIT 10;")
    print("```")
    
    print("\n【根本的な解決】")
    print("ctidベースではなく、通常のCOPYを使用")
    print("これで確実に100%取得できる")


if __name__ == "__main__":
    identify_exact_problem()
    analyze_pattern()
    propose_solution()