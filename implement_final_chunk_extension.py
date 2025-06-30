#!/usr/bin/env python3
"""
最終チャンク拡張の実装例
"""

def show_rust_modification():
    """Rustコードの修正案を表示"""
    
    print("=== 最終チャンク拡張の実装 ===\n")
    
    print("【修正箇所: main_single_chunk.rs】")
    print("行134-139を以下のように修正:\n")
    
    rust_code = '''let chunk_end_page = if chunk_id == total_chunks - 1 {
    // 最後のチャンクは最大ページ+1まで読む（全データ確実にカバー）
    max_page + 1
} else {
    (chunk_id + 1) as u32 * pages_per_chunk
};'''
    
    print(rust_code)
    
    print("\n\n【修正箇所: COPY文も調整（行291-294）】")
    
    copy_code = '''// COPY開始（最終チャンクは+1ページまで含める）
let copy_query = if chunk_id == total_chunks - 1 && end_page == max_page + 1 {
    // 最終チャンクの特別処理
    format!(
        "COPY (SELECT * FROM {} WHERE ctid >= '({},0)'::tid) TO STDOUT (FORMAT BINARY)",
        table_name, start_page
    )
} else {
    format!(
        "COPY (SELECT * FROM {} WHERE ctid >= '({},0)'::tid AND ctid < '({},0)'::tid) TO STDOUT (FORMAT BINARY)",
        table_name, start_page, end_page
    )
};'''
    
    print(copy_code)
    
    print("\n\n【動作確認】")
    print("1. コンパイル:")
    print("   cd rust_bench_optimized")
    print("   cargo build --release")
    print("\n2. 80チャンクでテスト:")
    print("   export TOTAL_CHUNKS=80")
    print("   python benchmark/benchmark_rust_gpu_direct.py")
    print("\n3. 結果確認:")
    print("   最終チャンクが追加の行を処理するか確認")


def show_alternative_sweep():
    """代替案：追加スイープの実装"""
    
    print("\n\n=== 代替案：追加スイープチャンク ===\n")
    
    print("【Python側で追加処理】")
    
    python_code = '''# benchmark_rust_gpu_direct.pyに追加

def run_final_sweep(dsn, table_name, start_page=4604888):
    """最終スイープで残りのデータを取得"""
    
    print(f"\\n最終スイープ実行: ページ{start_page}以降")
    
    env = os.environ.copy()
    env.update({
        'CHUNK_ID': '80',  # 追加チャンク
        'TOTAL_CHUNKS': '81',  # 仮想的に81チャンク目
        'GPUPASER_PG_DSN': dsn,
        'TABLE_NAME': table_name,
        'RUST_PARALLEL_CONNECTIONS': '4',  # 少なめに
    })
    
    # 特別なRustプログラムを呼ぶか、既存のものを流用
    result = subprocess.run(
        ['./rust_bench_optimized/target/release/pg_fast_copy_single_chunk'],
        env=env,
        capture_output=True,
        text=True
    )
    
    # 結果をパース
    if "===CHUNK_RESULT_JSON===" in result.stdout:
        # JSONを抽出して処理
        pass
'''
    
    print(python_code)
    
    print("\n【メリット】")
    print("- 既存の80チャンク処理を変更不要")
    print("- 欠損データを確実に回収")
    print("- 実装が簡単")


if __name__ == "__main__":
    show_rust_modification()
    show_alternative_sweep()