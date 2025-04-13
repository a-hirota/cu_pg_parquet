"""
PostgreSQL-GPU処理パイプライン メインモジュール
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any

from .pg_connector import connect_to_postgres, check_table_exists, get_table_info, get_table_row_count, get_binary_data
from .binary_parser import BinaryDataParser
from .memory_manager import GPUMemoryManager
from .gpu_decoder import GPUDecoder
from .output_handler import OutputHandler
from .utils import ChunkConfig, ColumnInfo

class PgGpuProcessor:
    """PostgreSQLデータGPU処理の統合クラス"""
    
    def __init__(self, dbname='postgres', user='postgres', password='postgres', host='localhost', parquet_output=None):
        """初期化"""
        self.conn = connect_to_postgres(dbname, user, password, host)
        self.memory_manager = GPUMemoryManager()
        self.parser = BinaryDataParser()
        self.gpu_decoder = GPUDecoder()
        self.output_handler = OutputHandler(parquet_output)
        self.parquet_output = parquet_output
    
    def process_table(self, table_name: str, limit: Optional[int] = None):
        """テーブル全体を処理（複数チャンク対応）"""
        # テーブルの存在確認
        if not check_table_exists(self.conn, table_name):
            raise ValueError(f"Table {table_name} does not exist")
        
        # テーブル情報と行数の取得
        columns = get_table_info(self.conn, table_name)
        if not columns:
            raise ValueError(f"No columns found in table {table_name}")
        
        total_rows = min(get_table_row_count(self.conn, table_name), limit if limit else float('inf'))
        
        # 最適なチャンクサイズを計算
        optimal_chunk_size = self.memory_manager.calculate_optimal_chunk_size(columns, total_rows)
        
        # チャンク設定の初期化（最適なチャンクサイズを使用）
        chunk_config = ChunkConfig(total_rows, optimal_chunk_size)
        print(f"GPUメモリに基づく最適チャンクサイズ: {chunk_config.rows_per_chunk}行")
        
        # バイナリデータの取得
        start_time = time.time()
        buffer_data, buffer = get_binary_data(self.conn, table_name, limit)
        
        # 複数チャンク処理のための変数初期化
        num_columns = len(columns)
        all_results = {}  # 全チャンクの結果を集約する辞書
        processed_rows = 0
        chunk_count = 0
        
        # 全行を複数チャンクに分けて処理
        while processed_rows < total_rows:
            # 現在のチャンクサイズを計算
            current_chunk_size = min(chunk_config.rows_per_chunk, total_rows - processed_rows)
            chunk_count += 1
            
            print(f"\n== チャンク {chunk_count}: {processed_rows+1}～{processed_rows+current_chunk_size}行目を処理 (全{total_rows}行中) ==")
            
            # GPUバッファの初期化（各チャンクごとに新しいバッファを作成）
            buffers = self.memory_manager.initialize_device_buffers(columns, current_chunk_size)
            
            try:
                # バイナリデータの解析（特定行からスタート）
                # 解析の際、常に全データを使用し、チャンクサイズに制限を設ける
                chunk_array, field_offsets, field_lengths, rows_in_chunk = self.parser.parse_chunk(
                    buffer_data, 
                    max_chunk_size=1024*1024,
                    num_columns=num_columns,
                    start_row=processed_rows,  # 開始行を指定
                    max_rows=current_chunk_size  # 現在のチャンクサイズで行数を制限
                )
                
                if rows_in_chunk == 0:
                    print("これ以上処理する行がありません")
                    break
                    
                print(f"処理中: {rows_in_chunk}行")
            except Exception as e:
                print(f"チャンク {chunk_count} のバイナリデータ解析でエラー: {e}")
                # 最初のチャンクでエラーなら致命的
                if chunk_count == 1:
                    raise
                # それ以外は処理済みの行だけを返す
                print(f"これまでに処理した {processed_rows} 行の結果を返します")
                break
            
            # GPUで解析
            d_col_types = None
            d_col_lengths = None
            d_chunk = None  # バイナリデータ用バッファ参照
            d_offsets = None  # オフセット用バッファ参照
            d_lengths = None  # 長さ用バッファ参照
            
            try:
                print(f"GPUデコード開始: チャンク {chunk_count}（{rows_in_chunk}行）")
                
                # チャンク間でのGCを促進
                import gc
                gc.collect()
                from numba import cuda
                cuda.synchronize()
                
                # デバイスに転送
                d_col_types = self.memory_manager.transfer_to_device(buffers["col_types"], np.int32)
                d_col_lengths = self.memory_manager.transfer_to_device(buffers["col_lengths"], np.int32)
                
                # デコード処理
                chunk_results = self.gpu_decoder.decode_chunk(
                    buffers, 
                    chunk_array, 
                    field_offsets, 
                    field_lengths, 
                    rows_in_chunk, 
                    columns
                )
                
                print(f"GPUデコード完了: チャンク {chunk_count}")
                
                # 結果の集約
                if not all_results:
                    # 初回は結果をそのまま格納
                    all_results = chunk_results
                else:
                    # 2回目以降は結果を連結
                    for col_name, data in chunk_results.items():
                        if isinstance(data, np.ndarray):
                            # NumPy配列はconcatenateで連結
                            all_results[col_name] = np.concatenate([all_results[col_name], data])
                        elif isinstance(data, list):
                            # リストはextendで連結
                            all_results[col_name].extend(data)
                
                # 処理済み行数の更新
                processed_rows += rows_in_chunk
                
            except Exception as e:
                print(f"チャンク {chunk_count} 処理中にエラー発生: {e}")
                raise
                
            finally:
                # GPUリソースのクリーンアップ
                cleanup_dict = {
                    "d_col_types": d_col_types, 
                    "d_col_lengths": d_col_lengths,
                    "d_chunk": d_chunk,
                    "d_offsets": d_offsets,
                    "d_lengths": d_lengths
                }
                self.memory_manager.cleanup_buffers(cleanup_dict)
                self.memory_manager.cleanup_buffers(buffers)
                
                # 明示的にGPU同期を取り、全リソースが解放されたことを確認
                from numba import cuda
                cuda.synchronize()
                
                # チャンク処理後のガベージコレクションを促進
                import gc
                gc.collect()
        
        # 全チャンクの処理完了
        print(f"\n全 {processed_rows} 行の処理が完了しました（全{total_rows}行中）")
        
        # 結果の処理
        self.output_handler.process_chunk_result(all_results)
        final_results = self.output_handler.print_summary()
        print(f"Processing completed in {time.time() - start_time:.3f}s")
        
        return final_results
    
    def close(self):
        """リソースの解放"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

def load_table_optimized(table_name: str, limit: Optional[int] = None, parquet_output: Optional[str] = None):
    """最適化されたGPU実装でテーブルを読み込む（コンビニエンス関数）"""
    processor = PgGpuProcessor(parquet_output=parquet_output)
    try:
        results = processor.process_table(table_name, limit)
        
        # Parquet出力が指定されている場合の検証
        if parquet_output:
            print(f"\nParquetファイルが保存されました: {parquet_output}")
            print("cuDFで読み込みテスト:")
            try:
                import cudf
                df = cudf.read_parquet(parquet_output)
                print("\n最初の5行:")
                print(df.head(5))
                print("\n最後の5行:")
                print(df.tail(5))
            except ImportError:
                print("cuDFがインストールされていないため、読み込みテストをスキップします")
            except Exception as e:
                print(f"Parquetファイル読み込み中にエラー: {e}")
                
        return results
    finally:
        processor.close()

if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='PostgreSQL GPU Parser')
    parser.add_argument('--table', required=True, help='Table name to process')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows')
    parser.add_argument('--parquet', help='Output path for Parquet file')
    args = parser.parse_args()
    
    # テーブル処理
    print(f"=== {args.table}テーブル ===")
    print("\n[最適化GPU実装]")
    start_time = time.time()
    
    try:
        results = load_table_optimized(args.table, args.limit, args.parquet)
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
    except Exception as e:
        print(f"Error processing {args.table} table: {e}")
