#!/usr/bin/env python3
"""
Grid境界スレッドデバッグ情報テスト
=================================

テストモードでGPU処理を実行し、Grid境界スレッドの情報を確認
"""

import os
import sys
import subprocess
import json
import glob

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_test():
    """小規模データでテストモードを実行"""
    
    # 環境設定
    env = os.environ.copy()
    env["GPUPASER_PG_DSN"] = "dbname=postgres user=postgres host=localhost port=5432"
    env["GPUPGPARSER_TEST_MODE"] = "1"  # テストモード有効化
    env["RUST_PARALLEL_CONNECTIONS"] = "1"
    
    # conda環境の設定
    conda_setup = "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate cudf_dev"
    
    # テスト用の小さなテーブルを使用（lineorderの一部）
    # まず行数を確認
    count_cmd = f'{conda_setup} && python -c "import psycopg2; conn = psycopg2.connect(\'dbname=postgres user=postgres host=localhost port=5432\'); cur = conn.cursor(); cur.execute(\'SELECT count(*) FROM lineorder LIMIT 10000\'); print(\'Test rows:\', cur.fetchone()[0]); cur.close(); conn.close()"'
    
    print("=== Grid境界スレッドデバッグテスト ===")
    print("行数確認中...")
    subprocess.run(count_cmd, shell=True, executable='/bin/bash', env=env)
    
    # GPU処理実行（小規模データ）
    print("\nGPU処理実行中（テストモード）...")
    gpu_cmd = f'{conda_setup} && python cu_pg_parquet.py --test --table lineorder --parallel 1 --chunks 1 --output /tmp/test_grid_boundary'
    
    result = subprocess.run(
        gpu_cmd,
        shell=True,
        executable='/bin/bash',
        env=env,
        capture_output=True,
        text=True,
        cwd="/home/ubuntu/gpupgparser"
    )
    
    print("\n=== 実行結果 ===")
    print("リターンコード:", result.returncode)
    
    if result.stdout:
        print("\n標準出力:")
        # Grid境界情報のみ抽出
        lines = result.stdout.split('\n')
        in_grid_section = False
        for line in lines:
            if "Grid境界スレッドデバッグ情報" in line:
                in_grid_section = True
            if in_grid_section:
                print(line)
            if "デバッグ情報をJSONファイルに保存" in line:
                in_grid_section = False
                # JSONファイルパスを抽出
                if "_grid_debug.json" in line:
                    json_path = line.split(": ")[-1].strip()
                    if os.path.exists(json_path):
                        print(f"\nJSONファイル内容:")
                        with open(json_path, 'r') as f:
                            debug_data = json.load(f)
                            print(json.dumps(debug_data, indent=2))
                        # クリーンアップ
                        os.remove(json_path)
    
    if result.stderr:
        print("\n標準エラー:")
        print(result.stderr[:1000])  # 最初の1000文字のみ
    
    # 生成されたParquetファイルの確認
    parquet_files = glob.glob("/tmp/test_grid_boundary/*.parquet")
    if parquet_files:
        print(f"\n生成されたParquetファイル: {len(parquet_files)}個")
        for f in parquet_files:
            os.remove(f)  # クリーンアップ
    
    # 出力ディレクトリのクリーンアップ
    import shutil
    if os.path.exists("/tmp/test_grid_boundary"):
        shutil.rmtree("/tmp/test_grid_boundary")


if __name__ == "__main__":
    run_test()