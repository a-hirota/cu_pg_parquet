#!/usr/bin/env python3
"""
Rustのチャンク分割ロジックをデバッグ
"""

import os
import subprocess
import json

def main():
    # テストモードを有効化
    env = os.environ.copy()
    env['GPUPGPARSER_TEST_MODE'] = '1'
    env['CHUNK_ID'] = '0'
    env['TOTAL_CHUNKS'] = '16'
    env['TABLE_NAME'] = 'lineorder'
    
    print("=== Rustのチャンク分割ロジックをデバッグ ===")
    print("チャンク0の処理を実行して、ページ範囲と行数を確認します...\n")
    
    rust_binary = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
    
    # Rustプログラムを実行
    result = subprocess.run(
        [rust_binary],
        capture_output=True,
        text=True,
        env=env
    )
    
    print("【Rust出力】")
    print(result.stdout)
    
    if result.stderr:
        print("\n【エラー出力】")
        print(result.stderr)
    
    # JSON結果を解析
    output = result.stdout
    json_start = output.find("===CHUNK_RESULT_JSON===")
    json_end = output.find("===END_CHUNK_RESULT_JSON===")
    
    if json_start != -1 and json_end != -1:
        json_str = output[json_start + len("===CHUNK_RESULT_JSON==="):json_end].strip()
        chunk_data = json.loads(json_str)
        
        print("\n【チャンク情報】")
        print(f"チャンクID: {chunk_data['chunk_id']}")
        print(f"ファイルサイズ: {chunk_data['total_bytes'] / 1024**3:.2f} GB")
        print(f"転送時間: {chunk_data['elapsed_seconds']:.2f}秒")

if __name__ == "__main__":
    main()