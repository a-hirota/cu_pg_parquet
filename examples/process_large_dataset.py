#!/usr/bin/env python
"""
大規模データセット処理のサンプルスクリプト

このスクリプトはPostgreSQLの大きなテーブルからデータを取得し、
GPUメモリの使用量を最適化しながら複数チャンクに分けて処理します。
"""

import time
import argparse
import numpy as np
from numba import cuda
import psycopg2

# gpuPaserモジュールをインポート
from gpupaser.main import PgGpuProcessor, load_table_optimized
from gpupaser.memory_manager import GPUMemoryManager
from gpupaser.utils import ColumnInfo, ChunkConfig
from gpupaser.pg_connector import connect_to_postgres, get_table_info, get_table_row_count

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
            chunk_results = processor.process_table(table_name, current_chunk_size, start_row)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PostgreSQL 大規模データ処理')
    parser.add_argument('--table', required=True, help='処理するテーブル名')
    parser.add_argument('--rows', type=int, default=None, help='処理する行数 (デフォルト: テーブル全体)')
    parser.add_argument('--chunk-size', type=int, default=None, help='チャンクサイズ (デフォルト: 自動計算)')
    parser.add_argument('--analyze', action='store_true', help='テーブル分析のみ実行')
    
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
        
        # 複数チャンクに分けて処理
        results = process_in_chunks(args.table, total_rows, args.chunk_size)
        
        # 結果サマリー
        if results:
            print("\n=== 結果サマリー ===")
            for col_name, data in results.items():
                print(f"{col_name}: {type(data)} 長さ={len(data)}")
