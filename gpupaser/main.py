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
    
    def __init__(self, dbname='postgres', user='postgres', password='postgres', host='localhost'):
        """初期化"""
        self.conn = connect_to_postgres(dbname, user, password, host)
        self.memory_manager = GPUMemoryManager()
        self.parser = BinaryDataParser()
        self.gpu_decoder = GPUDecoder()
        self.output_handler = OutputHandler()
    
    def process_table(self, table_name: str, limit: Optional[int] = None):
        """テーブル全体を処理"""
        # テーブルの存在確認
        if not check_table_exists(self.conn, table_name):
            raise ValueError(f"Table {table_name} does not exist")
        
        # テーブル情報と行数の取得
        columns = get_table_info(self.conn, table_name)
        if not columns:
            raise ValueError(f"No columns found in table {table_name}")
        
        row_count = min(get_table_row_count(self.conn, table_name), limit if limit else float('inf'))
        
        # チャンク設定の初期化
        chunk_config = ChunkConfig(row_count)
        
        # バイナリデータの取得
        start_time = time.time()
        buffer_data, buffer = get_binary_data(self.conn, table_name, limit)
        
        # GPUバッファの初期化
        buffers = self.memory_manager.initialize_device_buffers(columns, chunk_config.rows_per_chunk)
        
        # バイナリデータの解析
        chunk_array, field_offsets, field_lengths, rows_in_chunk = self.parser.parse_chunk(
            buffer_data, 
            max_chunk_size=1024*1024,
            num_columns=len(columns)
        )
        print(f"\nProcessing {rows_in_chunk} rows")
        
        # GPUで解析
        try:
            # デバイスに転送
            d_col_types = self.memory_manager.transfer_to_device(buffers["col_types"], np.int32)
            d_col_lengths = self.memory_manager.transfer_to_device(buffers["col_lengths"], np.int32)
            
            # デコード処理
            results = self.gpu_decoder.decode_chunk(
                buffers, 
                chunk_array, 
                field_offsets, 
                field_lengths, 
                rows_in_chunk, 
                columns
            )
            
            # 結果の処理
            self.output_handler.process_chunk_result(results)
            
        finally:
            # GPUリソースのクリーンアップ
            cleanup_dict = {"d_col_types": d_col_types, "d_col_lengths": d_col_lengths}
            self.memory_manager.cleanup_buffers(cleanup_dict)
            self.memory_manager.cleanup_buffers(buffers)
        
        # 最終結果の取得と表示
        final_results = self.output_handler.print_summary()
        print(f"Processing completed in {time.time() - start_time:.3f}s")
        
        return final_results
    
    def close(self):
        """リソースの解放"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

def load_table_optimized(table_name: str, limit: Optional[int] = None):
    """最適化されたGPU実装でテーブルを読み込む（コンビニエンス関数）"""
    processor = PgGpuProcessor()
    try:
        return processor.process_table(table_name, limit)
    finally:
        processor.close()

if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description='PostgreSQL GPU Parser')
    parser.add_argument('--table', required=True, help='Table name to process')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of rows')
    args = parser.parse_args()
    
    # テーブル処理
    print(f"=== {args.table}テーブル ===")
    print("\n[最適化GPU実装]")
    start_time = time.time()
    
    try:
        results = load_table_optimized(args.table, args.limit)
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
    except Exception as e:
        print(f"Error processing {args.table} table: {e}")
