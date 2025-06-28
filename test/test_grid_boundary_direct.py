#!/usr/bin/env python3
"""
Grid境界スレッドデバッグ情報テスト（直接実行版）
==========================================

テストモードでGPU処理を直接実行し、Grid境界スレッドの情報を確認
"""

import os
import sys
import tempfile
import shutil

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 環境変数設定
os.environ["GPUPASER_PG_DSN"] = "dbname=postgres user=postgres host=localhost port=5432"
os.environ["GPUPGPARSER_TEST_MODE"] = "1"  # テストモード有効化
os.environ["RUST_PARALLEL_CONNECTIONS"] = "1"


def test_small_data():
    """小規模データでテスト"""
    print("=== Grid境界スレッドデバッグテスト（直接実行） ===")
    
    # 出力ディレクトリ作成
    output_dir = tempfile.mkdtemp(prefix="test_grid_")
    print(f"出力ディレクトリ: {output_dir}")
    
    try:
        # ベンチマークモジュールをインポート
        sys.path.append("/home/ubuntu/gpupgparser/docs/benchmark")
        from benchmark_rust_gpu_direct import main as benchmark_main
        
        # sys.argvを設定
        original_argv = sys.argv
        sys.argv = [
            "test_grid_boundary_direct.py",
            "--table", "lineorder",
            "--parallel", "1",
            "--chunks", "1"
        ]
        
        # 直接実行
        print("\nGPU処理実行中...")
        benchmark_main()
        
        print("\nテスト完了")
        
    except Exception as e:
        print(f"\nエラー発生: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sys.argv = original_argv
        
        # クリーンアップ
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"\n出力ディレクトリを削除しました: {output_dir}")


if __name__ == "__main__":
    test_small_data()