#!/usr/bin/env python3
"""
cu_pg_parquet - PostgreSQL to GPU-accelerated Parquet converter

高性能なGPUアクセラレーションを活用したPostgreSQLバイナリデータパーサー。
PostgreSQL COPY BINARYデータを直接GPU上で解析し、cuDF DataFrameやParquetファイルに変換します。

使用例:
    # 基本的な使い方（デフォルト: 16並列×8チャンク）
    python cu_pg_parquet.py --table lineorder
    
    # カスタム設定
    python cu_pg_parquet.py --table lineorder --parallel 32 --chunks 16
    
環境変数:
    GPUPASER_PG_DSN: PostgreSQL接続情報
    RUST_PARALLEL_CONNECTIONS: Rust側の並列接続数（デフォルト: --parallelの値）
"""

import os
import sys
import argparse
import time
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# benchmarkディレクトリからインポート
sys.path.append(str(Path(__file__).parent / "docs" / "benchmark"))
from benchmark_rust_gpu_direct import main as benchmark_main


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="PostgreSQL → GPU-accelerated Parquet converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
環境設定例:
  export GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"
  export RUST_PARALLEL_CONNECTIONS=16
  
詳細はCLAUDE.mdまたはREADME.mdを参照してください。
        """
    )
    
    parser.add_argument(
        "--table", 
        type=str, 
        required=True,
        help="対象テーブル名"
    )
    parser.add_argument(
        "--parallel", 
        type=int, 
        default=16,
        help="並列接続数（デフォルト: 16）"
    )
    parser.add_argument(
        "--chunks", 
        type=int, 
        default=8,
        help="チャンク数（デフォルト: 8）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="出力ディレクトリ（デフォルト: ./output）"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="テストモード（Grid境界スレッド情報を出力）"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="出力ディレクトリ（--output-dirのエイリアス）"
    )
    
    args = parser.parse_args()
    
    # 環境変数の設定
    if "RUST_PARALLEL_CONNECTIONS" not in os.environ:
        os.environ["RUST_PARALLEL_CONNECTIONS"] = str(args.parallel)
    
    # PostgreSQL接続情報の確認
    if "GPUPASER_PG_DSN" not in os.environ:
        print("警告: GPUPASER_PG_DSN環境変数が設定されていません。")
        print("例: export GPUPASER_PG_DSN=\"dbname=postgres user=postgres host=localhost port=5432\"")
        return 1
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output or args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # テストモードの環境変数設定
    if args.test:
        os.environ["GPUPGPARSER_TEST_MODE"] = "1"
    
    # テーブル名を環境変数に設定（Rust側でも使用）
    os.environ["TABLE_NAME"] = args.table
    
    print(f"GPUPGParser - PostgreSQL to GPU-accelerated Parquet converter")
    print(f"============================================================")
    print(f"テーブル: {args.table}")
    print(f"並列数: {args.parallel}")
    print(f"チャンク数: {args.chunks}")
    print(f"総タスク数: {args.parallel * args.chunks}")
    print(f"出力先: {output_dir}")
    print()
    
    start_time = time.time()
    
    # ベンチマーク関数を直接呼び出し
    # sys.argvを一時的に書き換え
    original_argv = sys.argv
    sys.argv = [
        "cu_pg_parquet.py",
        "--table", args.table,
        "--parallel", str(args.parallel),
        "--chunks", str(args.chunks)
    ]
    
    try:
        # benchmark_rust_gpu_direct.pyのmain()を実行
        benchmark_main(total_chunks=args.chunks, table_name=args.table)
        elapsed_time = time.time() - start_time
        print(f"\n処理完了: {elapsed_time:.2f}秒")
        return 0
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        return 1
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    exit(main())