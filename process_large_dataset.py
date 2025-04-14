#!/usr/bin/env python
"""
大規模データセット処理のサンプルスクリプト

このスクリプトはPostgreSQLの大きなテーブルからデータを取得し、
GPUメモリの使用量を最適化しながら複数チャンクに分けて処理します。
"""

import time
import argparse
import numpy as np
import sys
import os
import concurrent.futures
from typing import Dict, List, Optional, Tuple
import glob

# Rayによる並列処理のためのインポート（--parallelフラグ時のみ使用）
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("警告: Rayがインストールされていません。マルチGPU並列処理には必要です。")

# モジュールパスの追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from numba import cuda
import psycopg2

# gpuPaserモジュールをインポート
from gpupaser.main import PgGpuProcessor, load_table_optimized
from gpupaser.memory_manager import GPUMemoryManager
from gpupaser.utils import ColumnInfo, ChunkConfig
from gpupaser.pg_connector import connect_to_postgres, get_table_info, get_table_row_count


# Rayでの分散処理用の関数定義
@ray.remote(num_gpus=1)
def process_chunk_ray(table_name: str, chunk_size: int, offset: int, output_file: str = None,
                      dbname: str = 'postgres', user: str = 'postgres',
                      password: str = 'postgres', host: str = 'localhost'):
    """Rayリモートタスクとして1チャンクを処理
    
    Args:
        table_name: 処理するテーブル名
        chunk_size: 処理する行数
        offset: 開始行オフセット
        output_file: Parquet出力ファイルパス（Noneの場合はメモリ内処理のみ）
        dbname, user, password, host: DB接続情報
    
    Returns:
        処理結果（チャンク情報と出力ファイルパス）
    """
    print(f"GPU処理開始: {table_name}テーブル {offset}～{offset+chunk_size}行")
    start_time = time.time()
    
    # GPUプロセッサの初期化
    processor = PgGpuProcessor(
        dbname=dbname, 
        user=user, 
        password=password, 
        host=host,
        parquet_output=output_file
    )
    
    try:
        # 指定範囲のデータを処理
        result = processor.process_table_chunk(table_name, chunk_size, offset, output_file)
        
        processing_time = time.time() - start_time
        print(f"GPU処理完了: オフセット={offset} 行数={chunk_size} 時間={processing_time:.3f}秒")
        
        return {
            "output_file": output_file,
            "rows_processed": chunk_size,
            "processing_time": processing_time,
            "offset": offset,
            "result": result if output_file is None else None  # メモリモードの場合のみ結果を返す
        }
    except Exception as e:
        print(f"チャンク処理エラー ({offset}～{offset+chunk_size}): {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # リソース解放
        processor.close()


def process_in_chunks(table_name, total_rows, chunk_size=None):
    """
    データを複数チャンクに分割して処理
    
    Args:
        table_name: 処理するテーブル名
        total_rows: 処理する合計行数
        chunk_size: チャンクサイズ（指定しない場合は自動計算）
    
    Returns:
        処理結果の辞書
    """
    print(f"=== テーブル {table_name} から {total_rows}行を処理 ===")
    
    # 接続とメモリマネージャーの初期化
    conn = connect_to_postgres()
    memory_manager = GPUMemoryManager()
    
    # テーブル情報の取得
    columns = get_table_info(conn, table_name)
    
    # チャンクサイズの決定
    if chunk_size is None:
        # 利用可能なGPUメモリから最適なチャンクサイズを計算
        chunk_size = memory_manager.calculate_optimal_chunk_size(columns, total_rows)
    
    # 安全のため、最大値を制限
    chunk_size = min(chunk_size, 65000)  # CUDA制限に余裕を持たせる
    
    print(f"GPUメモリに基づく最適チャンクサイズ: {chunk_size}行")
    
    # チャンク数の計算
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    print(f"処理チャンク数: {num_chunks}")
    
    # 結果の集約用辞書
    all_results = {}
    processed_rows = 0
    
    # 全体処理時間の測定
    total_start_time = time.time()
    
    # 各チャンクを処理
    for chunk_idx in range(num_chunks):
        start_row = chunk_idx * chunk_size
        current_chunk_size = min(chunk_size, total_rows - start_row)
        
        print(f"\n== チャンク {chunk_idx+1}/{num_chunks}: 行 {start_row+1}～{start_row+current_chunk_size} ==")
        chunk_start_time = time.time()
        
        # プロセッサ初期化
        processor = PgGpuProcessor()
        
        try:
            # チャンク処理
            chunk_results = processor.process_table_chunk(table_name, current_chunk_size, start_row)
            chunk_time = time.time() - chunk_start_time
            print(f"チャンク {chunk_idx+1} 処理時間: {chunk_time:.3f}秒")
            
            # 結果の集約
            if not all_results:
                # 初回はそのまま結果を保存
                all_results = chunk_results
            else:
                # 2回目以降は結果を連結
                for col_name, data in chunk_results.items():
                    if isinstance(data, np.ndarray):
                        # NumPy配列の場合はconcatenate
                        all_results[col_name] = np.concatenate([all_results[col_name], data])
                    elif isinstance(data, list):
                        # リストの場合はextend
                        all_results[col_name].extend(data)
            
            # 処理済み行数を更新
            processed_rows += current_chunk_size
            
        except Exception as e:
            print(f"チャンク {chunk_idx+1} の処理中にエラーが発生: {e}")
            # エラー発生時は現状の結果を返す
            break
            
        finally:
            # リソースの解放
            processor.close()
    
    total_time = time.time() - total_start_time
    print(f"\n=== 全体の処理完了 ===")
    print(f"処理した行数: {processed_rows}/{total_rows}")
    print(f"総処理時間: {total_time:.3f}秒")
    print(f"平均処理速度: {processed_rows/total_time:.1f}行/秒")
    
    # GPUメモリ使用状況の最終確認
    try:
        mem_info = cuda.current_context().get_memory_info()
        free_memory = mem_info[0]
        total_memory = mem_info[1]
        used_memory = total_memory - free_memory
        print(f"GPUメモリ使用状況: {used_memory/(1024**2):.2f}MB使用 / {total_memory/(1024**2):.2f}MB合計")
    except:
        pass
    
    return all_results


def process_parallel_ray(table_name, total_rows, gpus=2, chunk_size=None, output_dir=None, output_format='memory'):
    """Rayを使用した並列処理の実行
    
    Args:
        table_name: 処理するテーブル名
        total_rows: 処理する合計行数
        gpus: 使用するGPU数（0=利用可能な全て）
        chunk_size: チャンクサイズ（指定しない場合は自動計算）
        output_dir: 出力ディレクトリ（Parquet形式のみ）
        output_format: 出力形式（'parquet'または'memory'）
        
    Returns:
        結果辞書（メモリ形式）またはファイルパスのリスト（Parquet形式）
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Rayがインストールされていません。pip install rayでインストールしてください。")
    
    # Ray初期化
    ray.init()
    
    try:
        # 利用可能なGPU数の確認
        available_gpus = int(ray.available_resources().get('GPU', 0))
        if available_gpus == 0:
            raise RuntimeError("利用可能なGPUがありません。Ray環境でCUDAが正しく設定されているか確認してください。")
            
        # 使用するGPU数の決定
        if gpus == 0:
            num_gpus = available_gpus  # 0の場合は全GPU使用
        else:
            num_gpus = min(gpus, available_gpus)  # 利用可能数を超えないように
        
        print(f"Ray初期化完了: 利用可能なGPU: {available_gpus} 使用するGPU: {num_gpus}")
        
        # データベース接続して情報取得
        conn = connect_to_postgres()
        columns = get_table_info(conn, table_name)
        conn.close()
        
        # チャンクサイズとチャンク数の計算
        if chunk_size is None:
            # 最適なチャンクサイズを計算
            memory_manager = GPUMemoryManager()
            chunk_size = memory_manager.calculate_optimal_chunk_size(columns, total_rows)
            chunk_size = min(chunk_size, 65000)  # 安全のため制限
        
        # チャンク数の計算（最低でもGPU数分）
        num_chunks = max(num_gpus, (total_rows + chunk_size - 1) // chunk_size)
        
        # GPUあたりの最適なチャンク数を決定
        chunks_per_gpu = (num_chunks + num_gpus - 1) // num_gpus
        
        # チャンクサイズを再計算（チャンク数が決定した後）
        chunk_size = (total_rows + num_chunks - 1) // num_chunks
        
        print(f"並列処理設定: {num_gpus}GPU使用, チャンクサイズ={chunk_size}行, チャンク数={num_chunks}個")
        print(f"GPU当たりチャンク数: {chunks_per_gpu}")
        
        # 出力ディレクトリの設定（Parquet形式の場合）
        if output_format == 'parquet':
            if output_dir is None:
                output_dir = f"{table_name}_output"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Parquet出力ディレクトリ: {output_dir}")
        
        # タスク実行
        tasks = []
        start_time = time.time()
        
        for chunk_idx in range(num_chunks):
            offset = chunk_idx * chunk_size
            # 最後のチャンクは残りすべて
            current_chunk_size = min(chunk_size, total_rows - offset)
            if current_chunk_size <= 0:
                break
                
            # Parquet出力用のファイルパス設定
            output_file = None
            if output_format == 'parquet':
                output_file = os.path.join(output_dir, f"{table_name}_chunk_{chunk_idx}.parquet")
            
            # Rayタスクとして実行
            task = process_chunk_ray.remote(
                table_name, 
                current_chunk_size, 
                offset, 
                output_file
            )
            tasks.append(task)
        
        # すべての結果を待機
        print(f"全{len(tasks)}タスクを実行中...")
        results = ray.get(tasks)
        
        # 全体の処理時間を計算
        total_time = time.time() - start_time
        total_rows_processed = sum(r["rows_processed"] for r in results)
        
        print(f"\n=== 並列処理完了 ===")
        print(f"処理行数: {total_rows_processed}")
        print(f"総処理時間: {total_time:.3f}秒")
        print(f"スループット: {total_rows_processed/total_time:.1f}行/秒")
        
        # GPU別の処理統計
        gpu_stats = {}
        for i, result in enumerate(results):
            gpu_id = i % num_gpus  # 簡易的にGPU IDを割り当て
            if gpu_id not in gpu_stats:
                gpu_stats[gpu_id] = {"count": 0, "time": 0, "rows": 0}
            gpu_stats[gpu_id]["count"] += 1
            gpu_stats[gpu_id]["time"] += result["processing_time"]
            gpu_stats[gpu_id]["rows"] += result["rows_processed"]
        
        print("\n=== GPU別統計 ===")
        for gpu_id, stats in gpu_stats.items():
            print(f"GPU #{gpu_id}: {stats['count']}チャンク処理 {stats['rows']}行 {stats['time']:.3f}秒 " +
                  f"({stats['rows']/stats['time']:.1f}行/秒)")
        
        # 出力形式に基づいた結果の返却
        if output_format == 'parquet':
            # Parquet出力の場合はファイルパスのリストを返す
            output_files = [r["output_file"] for r in results if r["output_file"] is not None]
            print(f"\n出力ファイル ({len(output_files)}個):")
            for f in output_files:
                if os.path.exists(f):
                    file_size = os.path.getsize(f) / (1024 * 1024)  # MBに変換
                    print(f"  {f} ({file_size:.2f} MB)")
            return output_files
        else:
            # メモリ出力の場合は結果を統合
            all_results = {}
            
            # 結果のマージ（オフセット順）
            sorted_results = sorted(results, key=lambda r: r["offset"])
            
            for result_info in sorted_results:
                chunk_result = result_info.get("result", {})
                if not chunk_result:
                    continue
                    
                if not all_results:
                    # 初回はそのまま結果を保存
                    all_results = chunk_result
                else:
                    # 2回目以降は結果を連結
                    for col_name, data in chunk_result.items():
                        if isinstance(data, np.ndarray):
                            # NumPy配列の場合はconcatenate
                            all_results[col_name] = np.concatenate([all_results[col_name], data])
                        elif isinstance(data, list):
                            # リストの場合はextend
                            all_results[col_name].extend(data)
            
            return all_results
    finally:
        # Ray終了
        ray.shutdown()


def analyze_table(table_name):
    """テーブル情報の分析と最適な処理方法の提案"""
    print(f"=== テーブル {table_name} の分析 ===")
    
    # 接続
    conn = connect_to_postgres()
    
    # テーブル情報の取得
    columns = get_table_info(conn, table_name)
    row_count = get_table_row_count(conn, table_name)
    
    # カラム情報表示
    print(f"\nカラム情報:")
    for col in columns:
        print(f"  {col.name}: {col.type}" + (f" (長さ: {col.length})" if col.length else ""))
    
    # 行サイズ計算
    row_size = 0
    for col in columns:
        from gpupaser.utils import get_column_type, get_column_length
        if get_column_type(col.type) <= 1:  # 数値型
            row_size += 8
        else:  # 文字列型
            row_size += get_column_length(col.type, col.length)
    
    # メモリマネージャーの初期化
    memory_manager = GPUMemoryManager()
    
    # 最適チャンクサイズ計算
    optimal_chunk_size = memory_manager.calculate_optimal_chunk_size(columns, row_count)
    
    print(f"\n===== 分析結果 =====")
    print(f"テーブル行数: {row_count:,}行")
    print(f"1行あたりサイズ: {row_size}バイト")
    print(f"推定テーブル全体サイズ: {(row_size * row_count) / (1024**2):.2f}MB")
    print(f"最適チャンクサイズ: {optimal_chunk_size:,}行")
    print(f"推奨チャンク数: {(row_count + optimal_chunk_size - 1) // optimal_chunk_size}")
    
    # 処理時間の見積もり（1秒あたり10,000行と仮定）
    estimated_time = row_count / 10000
    if estimated_time < 60:
        print(f"推定処理時間: {estimated_time:.1f}秒")
    else:
        minutes = int(estimated_time / 60)
        seconds = estimated_time % 60
        print(f"推定処理時間: {minutes}分 {seconds:.1f}秒")
    
    conn.close()
    return {
        "row_count": row_count,
        "row_size": row_size,
        "optimal_chunk_size": optimal_chunk_size
    }


def display_data_sample(results, sample_rows=5):
    """処理結果のサンプルデータを表示する
    
    Args:
        results: 処理結果の辞書（カラム名→データ配列）
        sample_rows: 表示する行数
    """
    if not results:
        print("\n=== サンプルデータ: 結果がありません ===")
        return
        
    # サンプル行数の調整
    rows_available = min(sample_rows, len(next(iter(results.values()))))
    if rows_available <= 0:
        print("\n=== サンプルデータ: 行がありません ===")
        return
    
    print(f"\n=== サンプルデータ (先頭{rows_available}行) ===")
    
    # カラム名の表示
    columns = list(results.keys())
    col_widths = {col: max(len(col), 15) for col in columns}
    
    # ヘッダー行
    header = " | ".join(f"{col:<{col_widths[col]}}" for col in columns)
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    
    # データ行
    for row_idx in range(rows_available):
        row_data = []
        for col in columns:
            data = results[col]
            if isinstance(data, np.ndarray):
                value = data[row_idx]
                if isinstance(value, (np.integer, np.floating)):
                    cell = f"{value:<{col_widths[col]}g}"
                else:
                    cell = f"{str(value):<{col_widths[col]}}"
            elif isinstance(data, list) and row_idx < len(data):
                cell = f"{str(data[row_idx]):<{col_widths[col]}}"
            else:
                cell = f"{'N/A':<{col_widths[col]}}"
            row_data.append(cell)
        print(" | ".join(row_data))
    
    print(separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PostgreSQL 大規模データ処理')
    parser.add_argument('--table', required=True, help='処理するテーブル名')
    parser.add_argument('--rows', type=int, default=None, help='処理する行数 (デフォルト: テーブル全体)')
    parser.add_argument('--chunk-size', type=int, default=None, help='チャンクサイズ (デフォルト: 自動計算)')
    parser.add_argument('--analyze', action='store_true', help='テーブル分析のみ実行')
    parser.add_argument('--parquet', help='Parquetファイルの出力先パス')
    parser.add_argument('--output-format', choices=['parquet', 'memory'], default='memory', 
                       help='出力形式（parquet: Parquet形式で保存、memory: メモリ内NumPy配列）')
    parser.add_argument('--sample-rows', type=int, default=5, 
                       help='メモリ出力時のサンプル表示行数')
    parser.add_argument('--no-debug-files', action='store_true',
                       help='デバッグ用一時ファイル出力を無効化（.binファイル等）')
    parser.add_argument('--gpus', type=int, default=2,
                       help='並列処理時に使用するGPU数（デフォルト: 2、利用可能なすべてを使用する場合は0）')
    parser.add_argument('--parallel', action='store_true',
                       help='Rayを使ったマルチGPU並列処理を有効化')
    
    args = parser.parse_args()
    
    if args.analyze:
        # テーブル分析のみ
        analyze_table(args.table)
    else:
        # 実際の処理
        start_time = time.time()
        
        # 行数を取得（指定がなければテーブル全体）
        if args.rows is None:
            conn = connect_to_postgres()
            total_rows = get_table_row_count(conn, args.table)
            conn.close()
        else:
            total_rows = args.rows
        
        # デバッグファイル制御オプションの設定
        os.environ['GPUPASER_DEBUG_FILES'] = '0' if args.no_debug_files else '1'
        
        # 並列処理かシングルGPU処理かを決定
        if args.parallel:
            # 並列処理モード（Rayを使用）
            print(f"マルチGPU並列処理モード (GPU数: {args.gpus})")
            
            output_dir = None
            if args.output_format == 'parquet':
                # 出力先の決定
                if args.parquet:
                    # パスが直接指定された場合はディレクトリとして使用
                    if os.path.isdir(args.parquet):
                        output_dir = args.parquet
                    else:
                        # ディレクトリパスを抽出
                        output_dir = os.path.dirname(args.parquet)
                        if not output_dir:
                            output_dir = '.'
                else:
                    # デフォルトのディレクトリ
                    output_dir = f"{args.table}_parallel_output"
            
            # 並列処理実行
            results = process_parallel_ray(
                args.table, 
                total_rows, 
                args.gpus, 
                args.chunk_size, 
                output_dir, 
                args.output_format
            )
            
            # 結果サマリー出力（メモリモードの場合）
            if args.output_format == 'memory' and results:
                print("\n=== 結果サマリー ===")
                for col_name, data in results.items():
                    print(f"{col_name}: {type(data).__name__} 長さ={len(data)}")
                
                # サンプルデータの表示
                display_data_sample(results, args.sample_rows)
        
        else:
            # 単一GPU処理モード
            print("シングルGPU処理モード")
            
            # 出力先の決定
            parquet_path = None
            if args.output_format == 'parquet':
                if args.parquet:
                    parquet_path = args.parquet
                else:
                    # デフォルトのパス
                    parquet_path = f"{args.table}_output.parquet"
                
                print(f"Parquet出力ファイル: {parquet_path}")
            
            # 処理の実行
            if parquet_path:
                # Parquet出力モード
                results = load_table_optimized(args.table, args.rows, parquet_path)
            else:
                # メモリ出力モード
                results = process_in_chunks(args.table, total_rows, args.chunk_size)
            
            # 結果サマリー
            if results:
                print("\n=== 結果サマリー ===")
                for col_name, data in results.items():
                    print(f"{col_name}: {type(data).__name__} 長さ={len(data)}")
                
                # メモリ出力モードの場合はサンプルデータを表示
                if args.output_format == 'memory':
                    display_data_sample(results, args.sample_rows)
