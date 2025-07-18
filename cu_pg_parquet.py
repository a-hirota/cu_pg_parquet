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

# processorsディレクトリからインポート
sys.path.append(str(Path(__file__).parent / "processors"))
from gpu_pipeline_processor import main as pipeline_main


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
        "-y", "--yes",
        action="store_true",
        help="既存のParquetファイルを確認なしで削除"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="出力ディレクトリ（--output-dirのエイリアス）"
    )
    parser.add_argument(
        "--test-duplicate-keys",
        type=str,
        default=None,
        help="テストモード時の重複チェックに使用する列名（カンマ区切り）。例: lo_orderkey,lo_linenumber"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["snappy", "gzip", "lz4", "brotli", "zstd", "none"],
        help="Parquet圧縮方式（デフォルト: zstd）"
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
    
    # outputフォルダ内の.parquetファイルチェック
    parquet_files = list(output_dir.glob("*.parquet"))
    if parquet_files:
        print(f"\n⚠️  outputフォルダ内に{len(parquet_files)}個の.parquetファイルが見つかりました。")
        
        # -yオプションが指定されている場合は確認なしで削除
        if args.yes:
            for file in parquet_files:
                file.unlink()
            print(f"✅ {len(parquet_files)}個のファイルを削除しました。")
        else:
            response = input("削除してもよろしいですか？ [y/n]: ")
            if response.lower() == 'y':
                for file in parquet_files:
                    file.unlink()
                print(f"✅ {len(parquet_files)}個のファイルを削除しました。")
            else:
                print("処理を中断しました。")
                return 1
    
    # テストモードの環境変数設定
    if args.test:
        os.environ["GPUPGPARSER_TEST_MODE"] = "1"
    
    # --test-duplicate-keysが指定されているが--testがない場合は警告
    if args.test_duplicate_keys and not args.test:
        print("⚠️  警告: --test-duplicate-keysは--testと併用する必要があります。")
        print("   --test-duplicate-keysオプションは無視されます。")
    
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
        # gpu_pipeline_processor.pyのmain()を実行
        pipeline_main(
            total_chunks=args.chunks, 
            table_name=args.table, 
            test_mode=args.test,
            test_duplicate_keys=args.test_duplicate_keys if args.test else None,
            compression=args.compression
        )
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