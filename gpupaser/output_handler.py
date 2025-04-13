"""
デコード結果の処理と出力を管理するモジュール
"""

import numpy as np
from typing import Dict, List, Any

class ResultAggregator:
    """結果データの集約を管理するクラス"""
    
    def __init__(self):
        """初期化"""
        self.results = {}  # {column_name: data_list}
    
    def add_chunk_results(self, chunk_results: Dict[str, Any]):
        """チャンク結果を追加"""
        # 初期化
        if not self.results:
            self.results = {col_name: [] for col_name in chunk_results.keys()}
        
        # 結果を追加
        for col_name, data in chunk_results.items():
            if col_name in self.results:
                self.results[col_name].append(data)
    
    def get_aggregated_results(self):
        """集約された結果を取得"""
        final_results = {}
        
        for col_name, chunks in self.results.items():
            if not chunks:
                continue
                
            if isinstance(chunks[0], np.ndarray):
                # NumPy配列の場合は連結
                final_results[col_name] = np.concatenate(chunks)
            else:
                # リストの場合はフラット化
                final_results[col_name] = [item for chunk in chunks for item in chunk]
        
        return final_results
    
    def clear(self):
        """結果をクリア"""
        self.results = {}

class OutputHandler:
    """出力処理を管理するクラス"""
    
    def __init__(self):
        """初期化"""
        self.aggregator = ResultAggregator()
    
    def process_chunk_result(self, chunk_result: Dict[str, Any]):
        """チャンク結果を処理"""
        self.aggregator.add_chunk_results(chunk_result)
    
    def get_final_results(self):
        """最終結果を取得"""
        return self.aggregator.get_aggregated_results()
    
    def print_summary(self, limit=5):
        """結果の概要を表示"""
        results = self.get_final_results()
        
        print("\n=== 結果サマリー ===")
        for col_name, data in results.items():
            if len(data) > 0:
                print(f"{col_name}: {data[:limit]}")
                print(f"  型: {type(data[0])}, 行数: {len(data)}")
            else:
                print(f"{col_name}: [空]")
        
        return results

# 将来的にParquet出力を実装する予定
class ParquetWriter:
    """Parquet形式での出力を管理するクラス（将来実装）"""
    
    def __init__(self, output_path: str):
        """初期化"""
        self.output_path = output_path
        # PyArrowの初期化など
    
    def write_results(self, results: Dict[str, Any]):
        """結果をParquet形式で保存（スタブ）"""
        # 将来実装予定
        print(f"Results would be written to {self.output_path}")
