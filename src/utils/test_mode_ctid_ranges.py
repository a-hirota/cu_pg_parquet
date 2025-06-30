#!/usr/bin/env python3
"""
Rustプログラムの--testモードを実装する提案
各チャンクのCTID範囲と実際の行数を検証
"""

def generate_test_mode_code():
    """Rust側に追加すべきテストモードのコード例"""
    
    rust_code = '''
// main_single_chunk.rs に追加する提案コード

// コマンドライン引数の処理部分に追加
let is_test_mode = std::env::var("GPUPGPARSER_TEST_MODE").unwrap_or_default() == "1";

// チャンク範囲計算後に追加（138行目付近）
if is_test_mode {
    println!("=== テストモード: CTID範囲検証 ===");
    println!("チャンク{}: ページ範囲 {} - {} ({}ページ)", 
        chunk_id, chunk_start_page, chunk_end_page - 1, chunk_end_page - chunk_start_page);
    
    // 実際の行数を事前に確認
    let count_row = client.query_one(
        &format!("SELECT COUNT(*) FROM {} WHERE ctid >= '({},1)'::tid AND ctid < '({},1)'::tid", 
            table_name, chunk_start_page, chunk_end_page),
        &[]
    ).await?;
    let expected_rows: i64 = count_row.get(0);
    println!("  推定行数: {}", expected_rows);
    
    // 最初と最後のCTIDを確認
    let ctid_row = client.query_one(
        &format!("SELECT MIN(ctid::text), MAX(ctid::text) FROM {} WHERE ctid >= '({},1)'::tid AND ctid < '({},1)'::tid", 
            table_name, chunk_start_page, chunk_end_page),
        &[]
    ).await?;
    let min_ctid: String = ctid_row.get(0);
    let max_ctid: String = ctid_row.get(1);
    println!("  CTID範囲: {} - {}", min_ctid, max_ctid);
    
    // 最後のチャンクの場合、実際の最大ページを確認
    if chunk_id == total_chunks - 1 {
        let max_page_row = client.query_one(
            &format!("SELECT MAX((ctid::text::point)[0]::int) FROM {}", table_name),
            &[]
        ).await?;
        let actual_max_page: i32 = max_page_row.get(0);
        println!("  実際の最大ページ番号: {} (計算上: {})", actual_max_page, chunk_end_page - 1);
    }
    
    println!();
}

// JSONレスポンスに行数情報を追加
#[derive(Serialize, Deserialize, Debug)]
struct ChunkResult {
    columns: Vec<ColumnMeta>,
    chunk_id: usize,
    chunk_file: String,
    workers: Vec<WorkerMeta>,
    total_bytes: u64,
    elapsed_seconds: f64,
    row_count: Option<u64>,  // 新規追加
    page_range: Option<(u32, u32)>,  // 新規追加
}
'''
    
    return rust_code

def generate_python_test_script():
    """Pythonスクリプトでテストモードを使用する例"""
    
    python_code = '''
#!/usr/bin/env python3
"""
CTID範囲の検証テスト
"""

import os
import subprocess
import json

def test_ctid_ranges():
    """各チャンクのCTID範囲をテスト"""
    
    # テストモードを有効化
    env = os.environ.copy()
    env["GPUPGPARSER_TEST_MODE"] = "1"
    env["TABLE_NAME"] = "lineorder"
    env["TOTAL_CHUNKS"] = "32"
    
    print("=== CTID範囲テスト (32チャンク) ===\\n")
    
    total_rows = 0
    chunk_infos = []
    
    for chunk_id in range(32):
        env["CHUNK_ID"] = str(chunk_id)
        
        # Rustプログラムを実行
        result = subprocess.run(
            ["/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"],
            env=env,
            capture_output=True,
            text=True
        )
        
        # 出力からテスト情報を抽出
        lines = result.stdout.split("\\n")
        for line in lines:
            if "推定行数:" in line:
                rows = int(line.split(":")[1].strip().replace(",", ""))
                total_rows += rows
                chunk_infos.append({
                    "chunk_id": chunk_id,
                    "rows": rows
                })
                print(f"チャンク {chunk_id:2d}: {rows:,} 行")
    
    print(f"\\n合計行数: {total_rows:,}")
    
    # PostgreSQLの実際の行数と比較
    import psycopg
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM lineorder")
    actual_rows = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    
    print(f"実際の行数: {actual_rows:,}")
    print(f"差分: {total_rows - actual_rows:,} ({(total_rows - actual_rows) / actual_rows * 100:.4f}%)")

if __name__ == "__main__":
    test_ctid_ranges()
'''
    
    return python_code

def main():
    print("=== Rustプログラムへのテストモード実装提案 ===\n")
    
    print("1. Rust側の修正案:")
    print("-" * 60)
    print(generate_test_mode_code())
    print("-" * 60)
    
    print("\n2. Pythonテストスクリプト例:")
    print("-" * 60)
    print(generate_python_test_script())
    print("-" * 60)
    
    print("\n3. 使用方法:")
    print("   export GPUPGPARSER_TEST_MODE=1")
    print("   python test_ctid_ranges.py")
    
    print("\n4. 期待される効果:")
    print("   - 各チャンクの正確な行数を事前に確認")
    print("   - CTID範囲の重複や漏れを検出")
    print("   - 実際のページ番号と計算上のページ番号の差異を可視化")

if __name__ == "__main__":
    main()