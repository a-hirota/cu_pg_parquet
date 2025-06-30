#!/usr/bin/env python3
"""
最終チャンク拡張のテスト
"""
import subprocess
import os
import json

def test_final_chunk():
    """最終チャンク（チャンク79）だけをテスト"""
    
    print("=== 最終チャンク拡張のテスト ===\n")
    
    # 環境変数の設定
    env = os.environ.copy()
    env.update({
        'CHUNK_ID': '79',  # 最終チャンク
        'TOTAL_CHUNKS': '80',
        'GPUPASER_PG_DSN': os.environ.get('GPUPASER_PG_DSN'),
        'TABLE_NAME': 'lineorder',
        'RUST_PARALLEL_CONNECTIONS': '16',
        'GPUPGPARSER_TEST_MODE': '1',  # デバッグモード
    })
    
    print("チャンク79（最終チャンク）を実行中...")
    print("期待: ctid >= '(4557194,0)'::tid のみの条件（上限なし）")
    
    # Rustプログラムを実行
    result = subprocess.run(
        ['./rust_bench_optimized/target/release/pg_fast_copy_single_chunk'],
        env=env,
        capture_output=True,
        text=True
    )
    
    # 出力を表示
    print("\n=== 標準出力 ===")
    print(result.stdout)
    
    print("\n=== 標準エラー出力 ===")
    print(result.stderr[:1000])  # 最初の1000文字
    
    # JSON結果を解析
    if "===CHUNK_RESULT_JSON===" in result.stdout:
        json_start = result.stdout.find("===CHUNK_RESULT_JSON===") + len("===CHUNK_RESULT_JSON===")
        json_end = result.stdout.find("===END_CHUNK_RESULT_JSON===")
        if json_start > 0 and json_end > 0:
            json_str = result.stdout[json_start:json_end].strip()
            try:
                chunk_result = json.loads(json_str)
                total_bytes = chunk_result['total_bytes']
                
                print(f"\n=== 結果 ===")
                print(f"転送されたバイト数: {total_bytes:,} ({total_bytes/1024/1024/1024:.2f} GB)")
                
                # 推定行数（lineorderの場合、1行約352バイト）
                estimated_rows = total_bytes // 352
                print(f"推定行数: {estimated_rows:,}")
                
                # 以前のチャンク79の結果と比較
                print(f"\n以前のチャンク79: 3,173,117行")
                print(f"差分: {estimated_rows - 3_173_117:,}行")
                
            except json.JSONDecodeError as e:
                print(f"JSONパースエラー: {e}")
    
    return result.returncode == 0


def test_chunk_0():
    """比較のため、チャンク0もテスト"""
    
    print("\n\n=== チャンク0のテスト（比較用） ===\n")
    
    env = os.environ.copy()
    env.update({
        'CHUNK_ID': '0',
        'TOTAL_CHUNKS': '80',
        'GPUPASER_PG_DSN': os.environ.get('GPUPASER_PG_DSN'),
        'TABLE_NAME': 'lineorder',
        'RUST_PARALLEL_CONNECTIONS': '16',
    })
    
    result = subprocess.run(
        ['./rust_bench_optimized/target/release/pg_fast_copy_single_chunk'],
        env=env,
        capture_output=True,
        text=True
    )
    
    if "===CHUNK_RESULT_JSON===" in result.stdout:
        json_start = result.stdout.find("===CHUNK_RESULT_JSON===") + len("===CHUNK_RESULT_JSON===")
        json_end = result.stdout.find("===END_CHUNK_RESULT_JSON===")
        if json_start > 0 and json_end > 0:
            json_str = result.stdout[json_start:json_end].strip()
            try:
                chunk_result = json.loads(json_str)
                total_bytes = chunk_result['total_bytes']
                estimated_rows = total_bytes // 352
                print(f"チャンク0の推定行数: {estimated_rows:,}")
            except:
                pass


if __name__ == "__main__":
    success = test_final_chunk()
    if success:
        test_chunk_0()
        print("\n✅ 最終チャンク拡張が正常に動作しています")
    else:
        print("\n❌ エラーが発生しました")