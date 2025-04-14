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
    
    def __init__(self, parquet_output=None):
        """初期化"""
        self.aggregator = ResultAggregator()
        self.parquet_writer = None
        
        # Parquet出力設定があれば初期化
        if parquet_output:
            self.parquet_writer = ParquetWriter(parquet_output)
    
    def process_chunk_result(self, chunk_result: Dict[str, Any]):
        """チャンク結果を処理"""
        # 空のチャンク結果はスキップ
        if not chunk_result:
            print("空のチャンク結果をスキップします")
            return
            
        self.aggregator.add_chunk_results(chunk_result)
        
        # Parquet出力が有効ならチャンクをParquetに書き込む
        if self.parquet_writer:
            self.parquet_writer.write_chunk(chunk_result)
    
    def get_final_results(self):
        """最終結果を取得"""
        return self.aggregator.get_aggregated_results()
    
    def close(self):
        """リソースを解放"""
        if self.parquet_writer:
            self.parquet_writer.close()
    
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
        
        # Parquet出力を閉じる
        self.close()
        
        return results

class ParquetWriter:
    """Parquet形式での出力を管理するクラス"""
    
    def __init__(self, output_path: str):
        """初期化"""
        self.output_path = output_path
        self.writer = None
        # 書き込み済みレコード数の追跡
        self.records_written = 0
    
    def initialize_writer(self, first_chunk_data):
        """最初のチャンクからParquetライターを初期化"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # 最初のチャンクからスキーマを構築
        self.schema = self._create_schema(first_chunk_data)
        # ライターを初期化
        self.writer = pq.ParquetWriter(self.output_path, self.schema)
    
    def _create_schema(self, data):
        """データからPyArrowスキーマを作成"""
        import pyarrow as pa
        
        # カラム名とデータ型からスキーマを構築
        fields = []
        for col_name, values in data.items():
            # データ型に基づいてPyArrow型を決定
            if isinstance(values, np.ndarray):
                if np.issubdtype(values.dtype, np.integer):
                    pa_type = pa.int32()
                elif np.issubdtype(values.dtype, np.floating):
                    pa_type = pa.float64()
                else:
                    pa_type = pa.string()
            elif isinstance(values, list):
                if len(values) > 0:
                    if isinstance(values[0], (int, np.integer)):
                        pa_type = pa.int32()
                    elif isinstance(values[0], (float, np.floating)):
                        pa_type = pa.float64()
                    else:
                        pa_type = pa.string()
                else:
                    pa_type = pa.string()  # デフォルト
            else:
                pa_type = pa.string()  # デフォルト
                
            fields.append(pa.field(col_name, pa_type))
            
        return pa.schema(fields)
    
    def write_chunk(self, chunk_data):
        """チャンクデータをParquetファイルに追加"""
        import pyarrow as pa
        
        # 空のデータチェック
        if not chunk_data or not any(len(values) > 0 for values in chunk_data.values()):
            print("書き込み可能なデータがありません")
            return
            
        # チャンクデータからPyArrowテーブルを作成
        arrays = []
        names = []
        
        for col_name, values in chunk_data.items():
            # 空の値配列はスキップ
            if values is None or (isinstance(values, (list, tuple)) and len(values) == 0) or \
               (isinstance(values, np.ndarray) and values.size == 0):
                continue
                
            # NumPy配列またはリストからPyArrow配列を作成
            try:
                if isinstance(values, np.ndarray):
                    arr = pa.array(values)
                elif isinstance(values, list):
                    # numeric型の特殊表現を処理
                    if any(isinstance(v, str) and '@' in v for v in values if v is not None):
                        # PostgreSQLのnumeric型特殊表現の処理
                        converted_values = []
                        for v in values:
                            if v is None:
                                converted_values.append(None)
                            elif isinstance(v, str) and '@' in v:
                                try:
                                    # 数値化を試みる（簡易的な処理）
                                    # 値が文字列形式 "[high_bits-low_bits]@scale" の場合
                                    # 最も単純な近似として最初の数値部分を抽出
                                    parts = v.split('@')
                                    if len(parts) == 2:
                                        scale = int(parts[1])
                                        number_part = parts[0].strip('[]')
                                        if '-' in number_part:
                                            # 大きな数値の場合は数値化が難しいため、
                                            # 最初の部分だけを使用
                                            first_part = number_part.split('-')[0]
                                            try:
                                                val = float(first_part) / (10 ** scale)
                                                converted_values.append(val)
                                            except ValueError:
                                                converted_values.append(0.0)  # デフォルト値
                                        else:
                                            try:
                                                val = float(number_part) / (10 ** scale)
                                                converted_values.append(val)
                                            except ValueError:
                                                converted_values.append(0.0)  # デフォルト値
                                    else:
                                        converted_values.append(0.0)  # デフォルト値
                                except Exception as e:
                                    print(f"数値変換エラー: {e} for value {v}")
                                    converted_values.append(0.0)  # デフォルト値
                            else:
                                converted_values.append(v)
                        arr = pa.array(converted_values)
                    else:
                        arr = pa.array(values)
                else:
                    continue  # 不明なデータ型はスキップ
            except Exception as e:
                print(f"配列変換エラー: {e} for column {col_name} with type {type(values)}")
                continue  # エラーが発生した場合はこのカラムをスキップ
                
            arrays.append(arr)
            names.append(col_name)
        
        # 配列が空ならスキップ
        if not arrays or not names:
            print("書き込み可能なカラムがありません")
            return
            
        # PyArrow Table作成
        table = pa.Table.from_arrays(arrays, names=names)
        
        # 最初のチャンクならライターを初期化
        if self.writer is None:
            self.initialize_writer(chunk_data)
        
        # チャンクを追加
        self.writer.write_table(table)
        self.records_written += len(arrays[0])  # 最初の配列の長さを使用
        
    def close(self):
        """ライターを閉じる"""
        if self.writer:
            self.writer.close()
            print(f"Parquetファイルが保存されました: {self.output_path}")
            print(f"合計 {self.records_written} 行が書き込まれました")
