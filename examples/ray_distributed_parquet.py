#!/usr/bin/env python
"""
マルチGPUでPostgreSQLデータを並列処理し、Parquetファイルに出力するRayスクリプト
"""

import ray
import argparse
import os
import time
import glob
from typing import Dict, List, Optional, Tuple

from gpupaser.main import PgGpuProcessor

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='PostgreSQL GPU Parser - Ray分散処理版')
    parser.add_argument('--table', required=True, help='処理するテーブル名')
    parser.add_argument('--total_rows', type=int, required=True, help='処理する合計行数')
    parser.add_argument('--num_gpus', type=int, default=None, help='使用するGPU数（指定しない場合は利用可能なすべてのGPUを使用）')
    parser.add_argument('--gpu_ids', help='使用するGPU IDのカンマ区切りリスト (例: "0,1,2")')
    parser.add_argument('--output_dir', default='./parquet_output', help='Parquet出力ディレクトリ')
    parser.add_argument('--chunk_size', type=int, default=None, help='1チャンクあたりの行数（指定しない場合は自動計算）')
    parser.add_argument('--db_name', default='postgres', help='データベース名')
    parser.add_argument('--db_user', default='postgres', help='データベースユーザー')
    parser.add_argument('--db_password', default='postgres', help='データベースパスワード')
    parser.add_argument('--db_host', default='localhost', help='データベースホスト')
    return parser.parse_args()


@ray.remote(num_gpus=1, num_cpus=4)
def process_chunk(table_name: str, chunk_size: int, offset: int, output_file: str,
                  db_name: str = 'postgres', db_user: str = 'postgres',
                  db_password: str = 'postgres', db_host: str = 'localhost',
                  gpu_id: int = None):
    """1つのGPUで1チャンクを処理

    Args:
        table_name: 処理するテーブル名
        chunk_size: 処理する行数
        offset: 開始行オフセット
        output_file: Parquet出力ファイルパス
        db_name: データベース名
        db_user: データベースユーザー
        db_password: データベースパスワード
        db_host: データベースホスト
        gpu_id: 使用するGPU ID（Noneの場合はRayが自動割り当て）

    Returns:
        処理結果情報
    """
    # 特定のGPUを指定する場合
    if gpu_id is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"GPU #{gpu_id} で処理を実行")

    print(f"GPU処理開始: {table_name}テーブル {offset}～{offset+chunk_size}行")
    start_time = time.time()

    # GPUプロセッサの初期化（Parquet出力設定）
    processor = PgGpuProcessor(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        parquet_output=output_file
    )

    try:
        # 指定範囲のみを処理
        result = processor.process_table_chunk(table_name, chunk_size, offset, output_file)

        processing_time = time.time() - start_time
        print(f"GPU処理完了: {output_file} 処理時間: {processing_time:.3f}秒")

        return {
            "output_file": output_file,
            "rows_processed": chunk_size,
            "processing_time": processing_time,
            "offset": offset,
            "gpu_id": gpu_id
        }

    except Exception as e:
        print(f"チャンク処理エラー ({offset}～{offset+chunk_size}): {e}")
        raise
    finally:
        # リソース解放
        processor.close()


def main():
    """メイン処理"""
    # 引数解析
    args = parse_args()

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # Ray初期化
    ray.init()

    # 利用可能なGPU数の確認
    available_gpus = int(ray.available_resources().get('GPU', 0))
    if available_gpus == 0:
        raise RuntimeError("利用可能なGPUがありません。Ray環境でCUDAが正しく設定されているか確認してください。")

    # 使用するGPU IDの特定
    gpu_ids = []
    if args.gpu_ids:
        gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
        num_gpus = len(gpu_ids)
    else:
        # 使用するGPU数の決定
        num_gpus = args.num_gpus if args.num_gpus is not None else available_gpus
        num_gpus = min(num_gpus, available_gpus)  # 利用可能数を超えないように
        gpu_ids = list(range(num_gpus))  # 0からnum_gpus-1までの連番

    print(f"Ray初期化完了: 利用可能なGPU: {available_gpus} 使用するGPU: {num_gpus}")
    print(f"使用するGPU ID: {gpu_ids}")

    # チャンクサイズと数の計算
    if args.chunk_size:
        chunk_size = args.chunk_size
        num_chunks = (args.total_rows + chunk_size - 1) // chunk_size  # 切り上げ
    else:
        # 各GPUに均等に分配（最低でも1チャンク）
        num_chunks = max(num_gpus, 1)
        chunk_size = (args.total_rows + num_chunks - 1) // num_chunks

    print(f"チャンク設定: サイズ={chunk_size}行 数={num_chunks}個")

    # タスク実行
    results = []
    start_time = time.time()

    # GPUごとのチャンク割り当て計画を作成（前半・後半チャンクを分離）
    gpu_assignments = []
    num_chunks_per_half = (num_chunks + 1) // 2  # 半分に分割（切り上げ）
    num_gpus_per_half = max(1, len(gpu_ids) // 2)  # 半分のGPU数（最低1）
    
    for i in range(num_chunks):
        if i < num_chunks_per_half:
            # 前半部分 - 前半のGPUに割り当て
            gpu_idx = i % num_gpus_per_half
        else:
            # 後半部分 - 後半のGPUに割り当て
            remaining_gpus = len(gpu_ids) - num_gpus_per_half
            if remaining_gpus <= 0:
                # GPUが1つしかない場合は同じGPUを使用
                gpu_idx = 0
            else:
                # 後半のGPUセットからラウンドロビンで割り当て
                gpu_idx = num_gpus_per_half + ((i - num_chunks_per_half) % remaining_gpus)
        
        # GPU IDの範囲チェック
        gpu_idx = min(gpu_idx, len(gpu_ids) - 1)
        gpu_assignments.append(gpu_ids[gpu_idx])
        
    print(f"チャンク割り当て計画: {gpu_assignments}")

    print(f"{len(results)}個のチャンクを処理中...")
    for i in range(num_chunks):
        offset = i * chunk_size

        # 最後のチャンクは残りすべて（total_rowsを超えないように）
        current_chunk_size = min(chunk_size, args.total_rows - offset)
        if current_chunk_size <= 0:
            break

        output_file = os.path.join(args.output_dir, f"{args.table}_chunk_{i}.parquet")
        gpu_id = gpu_assignments[i]

        # 非同期実行（特定のGPU IDを指定）
        result = process_chunk.remote(
            args.table,
            current_chunk_size,
            offset,
            output_file,
            args.db_name,
            args.db_user,
            args.db_password,
            args.db_host,
            gpu_id
        )
        results.append(result)

    # すべての結果を待機
    output_info = ray.get(results)

    # 処理終了時間を記録（ファイルリスト表示やcuDF検証など前）
    processing_end_time = time.time()
    processing_time = processing_end_time - start_time
    total_rows = sum(info["rows_processed"] for info in output_info)

    print("\n=== 処理結果 ===")
    print(f"PostgreSQLからParquet出力までの処理時間: {processing_time:.3f}秒")
    print(f"処理行数: {total_rows}行")
    print(f"スループット: {total_rows/processing_time:.1f}行/秒")

    # GPU別の処理統計
    gpu_stats = {}
    for info in output_info:
        gpu_id = info.get("gpu_id", "unknown")
        if gpu_id not in gpu_stats:
            gpu_stats[gpu_id] = {"count": 0, "time": 0, "rows": 0}
        gpu_stats[gpu_id]["count"] += 1
        gpu_stats[gpu_id]["time"] += info["processing_time"]
        gpu_stats[gpu_id]["rows"] += info["rows_processed"]

    print("\n=== GPU別統計 ===")
    for gpu_id, stats in gpu_stats.items():
        print(f"GPU #{gpu_id}: {stats['count']}チャンク処理 {stats['rows']}行 {stats['time']:.3f}秒 " +
              f"({stats['rows']/stats['time']:.1f}行/秒)")

    print("\n--- 以下は参考情報（処理時間には含まれていません）---")
    
    # 出力ファイル一覧
    output_files = [info["output_file"] for info in output_info]
    print(f"\n出力ファイル ({len(output_files)}個):")
    for f in output_files:
        file_size = os.path.getsize(f) / (1024 * 1024)  # MBに変換
        print(f"  {f} ({file_size:.2f} MB)")

    # 検証用サンプル読み込み（オプション）
    verify_sample = os.environ.get("VERIFY_SAMPLE", "0").lower() in ["1", "true", "yes"]
    if verify_sample:
        try:
            if output_files:
                import cudf
                sample_file = output_files[0]
                print(f"\n検証: {sample_file}をcuDFで読み込み")
                df = cudf.read_parquet(sample_file)
                print(f"サンプルファイル行数: {len(df)}")
                print("\n最初の5行:")
                print(df.head(5))
        except ImportError:
            print("\ncuDFがインストールされていないため、検証をスキップします")
        except Exception as e:
            print(f"\nParquetファイル検証中にエラー: {e}")
    else:
        print("\nサンプル検証は無効です（有効にするには VERIFY_SAMPLE=1 を設定）")

    ray.shutdown()
    print("\n処理完了!")


if __name__ == "__main__":
    main()
