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
import re
import concurrent.futures
from typing import Dict, List, Optional, Tuple
import glob

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


def extract_table_from_sql(sql_query):
    """SQLクエリからテーブル名を抽出する
    
    Args:
        sql_query: SQLクエリ文字列
        
    Returns:
        抽出されたテーブル名、見つからない場合は"query_result"
    """
    # 正規表現でFROM句のテーブル名を探す
    # 基本的なSELECT文のパターン
    match = re.search(r'from\s+([^\s,\(\)]+)', sql_query.lower())
    
    if match:
        return match.group(1)
    
    # ALIASを使っているケース (e.g. "FROM table_name AS t")
    match = re.search(r'from\s+([^\s,\(\)]+)\s+as\s+', sql_query.lower())
    if match:
        return match.group(1)
    
    # 見つからなければデフォルト名
    return "query_result"


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
    parser.add_argument('--table', help='処理するテーブル名')
    parser.add_argument('--sql', help='カスタムSQLクエリ (--tableより優先)')
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
    parser.add_argument('--quiet', action='store_true',
                       help='出力を最小限にする（実行時間や結果のみ表示）')
    parser.add_argument('--gpuid', type=int, help='使用する特定のGPU ID（他の全てのGPUは無視されます）')
    
    args = parser.parse_args()
    
    # 特定のGPU IDが指定された場合、環境変数を設定
    if args.gpuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
        print(f"GPU {args.gpuid} を使用します (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    
    # SQLクエリが指定された場合、テーブル名を抽出
    if args.sql and not args.table:
        args.table = extract_table_from_sql(args.sql)
        print(f"SQLクエリからテーブル名を推測: {args.table}")
    
    # テーブル名とSQLクエリの整合性チェック
    if not args.table and not args.sql:
        parser.error("--table または --sql オプションを指定してください")
    
    # analyzeフラグがある場合はテーブル分析のみ実行
    if args.analyze:
        if not args.table:
            parser.error("テーブル分析には --table オプションが必要です")
        analyze_table(args.table)
    else:
        # 実際の処理
        start_time = time.time()
        
        # SQLクエリが指定された場合は行数の取得方法を変更
        if args.sql:
            # SQLクエリを使用して行数を推定
            # 行数が指定されていればそれを使用、なければ大きめの値を設定
            if args.rows is not None:
                total_rows = args.rows
            else:
                # SQLクエリの場合はデフォルト行数を大きめに設定
                total_rows = 1000000  # 100万行を想定
            print(f"SQLクエリを使用: {args.sql}")
            print(f"推定行数: {total_rows}行")
        else:
            # テーブル全体を処理
            if not args.table:
                parser.error("テーブル名またはSQLクエリを指定してください")
                
            # 行数を取得（指定がなければテーブル全体）
            if args.rows is None:
                conn = connect_to_postgres()
                total_rows = get_table_row_count(conn, args.table)
                conn.close()
            else:
                total_rows = args.rows
        
        # デバッグファイル制御オプションの設定
        os.environ['GPUPASER_DEBUG_FILES'] = '0' if args.no_debug_files else '1'
        
        # 出力制御オプションの設定
        if args.quiet:
            # 標準出力を一時的に抑制する設定
            os.environ['GPUPASER_QUIET'] = '1'
            # 最小限の情報のみ表示
            print("実行モード: GPU処理")
        else:
            os.environ['GPUPASER_QUIET'] = '0'
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
        if args.sql:
            # SQLクエリを使用した処理（カスタムクエリモード）
            processor = PgGpuProcessor(parquet_output=parquet_path)
            try:
                print(f"カスタムSQLクエリを実行: {args.sql}")
                # カスタムクエリで処理
                results = processor.process_custom_query(args.sql, parquet_path)
            finally:
                processor.close()
        else:
            # 通常のテーブル処理
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
