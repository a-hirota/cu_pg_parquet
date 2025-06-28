#!/usr/bin/env python3
"""
改善されたデバッグ情報の確認テスト
================================
"""

import os
import sys
import subprocess

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_test():
    """改善されたテストモードを実行"""
    
    # 環境設定
    env = os.environ.copy()
    env["GPUPASER_PG_DSN"] = "dbname=postgres user=postgres host=localhost port=5432"
    env["GPUPGPARSER_TEST_MODE"] = "1"  # テストモード有効化
    env["RUST_PARALLEL_CONNECTIONS"] = "2"
    
    # conda環境の設定
    conda_setup = "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate cudf_dev"
    
    print("=== 改善されたGrid境界デバッグ情報テスト ===")
    
    # GPU処理実行（小規模データ）
    print("\nGPU処理実行中（テストモード）...")
    gpu_cmd = f'{conda_setup} && python cu_pg_parquet.py --test --table lineorder --parallel 2 --chunks 2 --output /tmp/test_improved_debug 2>&1 | grep -A 100 "Grid境界スレッド"'
    
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
    if result.stdout:
        lines = result.stdout.split('\n')
        
        # 最初のGrid境界スレッドの情報を表示
        print_section = False
        line_count = 0
        
        for line in lines:
            if "Grid境界スレッド 1 ---" in line:
                print_section = True
            
            if print_section:
                print(line)
                line_count += 1
                
                # 次のスレッド情報が始まったら終了
                if line_count > 1 and "Grid境界スレッド 2 ---" in line:
                    print("\n[以降のスレッド情報は省略]")
                    break
    
    if result.stderr:
        print("\n標準エラー:")
        print(result.stderr[:500])


if __name__ == "__main__":
    run_test()