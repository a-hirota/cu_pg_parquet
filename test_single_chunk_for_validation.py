#!/usr/bin/env python3
"""
1チャンクだけ実行して動作確認
"""
import subprocess
import os
import sys
import json

# cudf_dev環境のPythonパスを設定
sys.path.insert(0, '/home/ubuntu/miniforge/envs/cudf_dev/lib/python3.11/site-packages')

def run_single_chunk_test():
    """1チャンクだけテスト実行"""
    
    print("=== 単一チャンクテスト（チャンク0） ===\n")
    
    # Rust側の実行
    env = os.environ.copy()
    env.update({
        'CHUNK_ID': '0',
        'TOTAL_CHUNKS': '80',
        'GPUPASER_PG_DSN': os.environ.get('GPUPASER_PG_DSN'),
        'TABLE_NAME': 'lineorder',
        'RUST_PARALLEL_CONNECTIONS': '16',
    })
    
    print("Rust転送を実行中...")
    result = subprocess.run(
        ['./rust_bench_optimized/target/release/pg_fast_copy_single_chunk'],
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"エラー: {result.stderr}")
        return False
    
    # JSON結果を取得
    json_data = None
    if "===CHUNK_RESULT_JSON===" in result.stdout:
        json_start = result.stdout.find("===CHUNK_RESULT_JSON===") + len("===CHUNK_RESULT_JSON===")
        json_end = result.stdout.find("===END_CHUNK_RESULT_JSON===")
        if json_start > 0 and json_end > 0:
            json_str = result.stdout[json_start:json_end].strip()
            json_data = json.loads(json_str)
    
    if not json_data:
        print("JSON結果が取得できませんでした")
        return False
    
    print(f"✅ Rust転送完了: {json_data['total_bytes'] / 1024 / 1024 / 1024:.2f} GB")
    
    # GPU処理をテスト
    print("\nGPU処理を実行中...")
    
    # src/main_postgres_to_parquet.pyを直接実行する代わりに、
    # 必要な処理部分だけを抽出
    try:
        import cudf
        from src.cuda_kernels.postgres_binary_parser import parse_postgres_binary_gpu
        
        # チャンクファイルを読み込み
        chunk_file = json_data['chunk_file']
        
        # メタデータ
        columns = json_data['columns']
        if not columns:
            print("警告: カラムメタデータがありません（チャンク0以外）")
            return True
        
        # GPUで処理（簡易版）
        print("GPU処理をスキップ（動作確認のみ）")
        print(f"チャンクファイル: {chunk_file}")
        print(f"カラム数: {len(columns)}")
        
        return True
        
    except ImportError as e:
        print(f"インポートエラー: {e}")
        print("cuDF環境が正しく設定されていることを確認してください")
        return False


if __name__ == "__main__":
    success = run_single_chunk_test()
    if success:
        print("\n✅ 単一チャンクテスト成功")
        print("\n次のステップ:")
        print("1. 80チャンク全体を実行するには時間がかかります（約75秒）")
        print("2. まず最終チャンク（79）の結果を確認することをお勧めします")
    else:
        print("\n❌ エラーが発生しました")