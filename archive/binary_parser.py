"""
PostgreSQLバイナリデータの基本パース処理とパイプライン処理
"""

import numpy as np
from numba import njit
from typing import Tuple, Dict, List, Any
import concurrent.futures
from queue import Queue
import threading
import time

from gpupaser.gpu_decoder import GPUDecoder
from gpupaser.memory_manager import allocate_gpu_buffers
from gpupaser.utils import ColumnInfo
from gpupaser.pg_connector import PostgresConnector
from gpupaser.output_handler import OutputHandler

# 循環インポートを避けるため、main.pyからのインポートを削除
# from gpupaser.main import PgGpuProcessor

# Numbaによる高速化を適用
@njit
def parse_binary_chunk(chunk_array, header_expected=True):
    """バイナリチャンクのパース"""
    field_offsets = []
    field_lengths = []
    pos = np.int64(0)
    
    # デバッグ出力の追加
    print(f"パース開始: chunk_size={len(chunk_array)}, header_expected={header_expected}")
    
    # ヘッダーのスキップ（最初のチャンクのみ）
    if header_expected and len(chunk_array) >= 11:
        header_pattern = np.array([80,71,67,79,80,89,10,255,13,10,0], dtype=np.int32)
        header_match = True
        
        # ヘッダーパターンの比較
        for i in range(min(11, len(chunk_array))):
            if chunk_array[i] != header_pattern[i]:
                header_match = False
                break
        
        if header_match:
            print(f"ヘッダーパターンを検出: PGCOPY\\n\\377\\r\\n\\0")
            pos = np.int64(11)
            
            if len(chunk_array) >= pos + 8:
                # フラグとヘッダー拡張をスキップ
                flags = np.int32((chunk_array[pos] << 24) | (chunk_array[pos+1] << 16) | \
                       (chunk_array[pos+2] << 8) | chunk_array[pos+3])
                pos += np.int64(4)
                ext_len = np.int32((chunk_array[pos] << 24) | (chunk_array[pos+1] << 16) | \
                         (chunk_array[pos+2] << 8) | chunk_array[pos+3])
                pos += np.int64(4)
                
                print(f"ヘッダー情報: フラグ={flags}, 拡張長={ext_len}")
                
                if ext_len > 0 and len(chunk_array) >= pos + ext_len:
                    pos += np.int64(ext_len)
    
    print(f"パースポジション: {pos}/{len(chunk_array)}")
    
    # チャンク内の各タプルを処理
    row_count = 0
    while pos + 2 <= len(chunk_array):
        # タプルのフィールド数を読み取り
        num_fields = np.int16((chunk_array[pos] << 8) | chunk_array[pos + 1])
        
        # 値のチェックと詳細デバッグ（Numba対応のため書式指定子を削除）
        if pos < 100:  # 先頭部分のみ詳細表示
            print(f"位置 {pos}: フィールド数={num_fields}, バイト値=[{chunk_array[pos]},{chunk_array[pos+1]}]")
        
        if num_fields == -1:  # np.int16(0xFFFF) は -1 になる
            print(f"ファイル終端マーカー(0xFFFF)を検出: pos={pos}")
            break
            
        if num_fields <= 0 or num_fields > 200:  # 異常値チェック
            print(f"警告: 異常なフィールド数 {num_fields} (位置={pos})")
            break
            
        row_count += 1
        pos += np.int64(2)
        
        # 各フィールドを処理
        field_count_in_row = 0
        for field_idx in range(num_fields):
            if pos + 4 > len(chunk_array):
                print(f"警告: フィールド長読み取り時にバッファ終端に到達 (pos={pos}, len={len(chunk_array)})")
                break
                
            # フィールド長を読み取り
            b0 = chunk_array[pos]
            b1 = chunk_array[pos + 1]
            b2 = chunk_array[pos + 2]
            b3 = chunk_array[pos + 3]
            
            # ビッグエンディアンからリトルエンディアンに変換
            # 整数オーバーフローを防ぐため、int32を使用
            field_len = np.int32(0)
            field_len = field_len | np.int32(b0 & 0xFF) << 24
            field_len = field_len | np.int32(b1 & 0xFF) << 16
            field_len = field_len | np.int32(b2 & 0xFF) << 8
            field_len = field_len | np.int32(b3 & 0xFF)
            
            # field_lenは既にnp.int32なので、符号は自動的に処理される
            
            if field_idx < 2 and row_count <= 2:  # 最初の行の詳細のみ表示
                print(f"行 {row_count} フィールド {field_idx}: 長さ={field_len}, バイト値=[{b0},{b1},{b2},{b3}]")
            
            pos += np.int64(4)
            field_count_in_row += 1
            
            if field_len == -1:  # NULL値
                field_offsets.append(0)  # NULL値のオフセットは0
                field_lengths.append(-1)
            elif field_len < -1:  # 異常値
                print(f"警告: 無効なフィールド長 {field_len} (行={row_count}, フィールド={field_idx})")
                break
            else:
                if pos + field_len > len(chunk_array):
                    print(f"警告: フィールドデータがバッファ終端を超える (pos={pos}, len={field_len}, buffer_size={len(chunk_array)})")
                    # チャンク境界をまたぐ場合は中断、ここまでのデータを返す
                    return np.array(field_offsets, dtype=np.int32), np.array(field_lengths, dtype=np.int32)
                
                # 正常な値フィールド
                field_offsets.append(int(pos))
                field_lengths.append(int(field_len))
                pos += np.int64(field_len)
        
        # 1行の処理が完了
        if field_count_in_row != num_fields:
            print(f"警告: 期待されるフィールド数と実際のフィールド数が一致しません: 期待={num_fields}, 実際={field_count_in_row}")
            break
    
    print(f"パース完了: 処理行数={row_count}, 取得フィールド数={len(field_offsets)}")
    
    return np.array(field_offsets, dtype=np.int32), np.array(field_lengths, dtype=np.int32)

class BinaryParser:
    """PostgreSQLバイナリデータをパースしてカラムデータに変換するクラス"""
    
    def __init__(self, use_gpu=True):
        """初期化
        
        Args:
            use_gpu: GPUを使用するかどうか
        """
        self.gpu_decoder = GPUDecoder() if use_gpu else None
        self.data_parser = BinaryDataParser(use_gpu=use_gpu)
        self.use_gpu = use_gpu
    
    def parse_postgres_binary(self, binary_data: bytes, columns: List[ColumnInfo]) -> Dict[str, Any]:
        """PostgreSQLバイナリデータをパースして列データに変換
        
        Args:
            binary_data: PostgreSQLバイナリデータ
            columns: カラム情報のリスト
            
        Returns:
            Dict[str, Any]: カラム名をキー、値の配列を値とする辞書
        """
        print(f"PostgreSQLバイナリデータのパース開始: {len(binary_data)} バイト, {len(columns)} 列")
        
        # カラム数の取得
        num_columns = len(columns)
        
        # バイナリデータをパース（フィールドの位置と長さを取得）
        chunk_array, field_offsets, field_lengths, rows_in_chunk = self.data_parser.parse_chunk(
            binary_data, 
            num_columns=num_columns, 
            max_rows=65535
        )
        
        if rows_in_chunk == 0 or field_offsets is None or field_lengths is None:
            print("データのパースに失敗しました")
            return {}
        
        print(f"バイナリデータのパース完了: {rows_in_chunk}行")
        
        # カラムの型情報を生成
        col_types = np.zeros(num_columns, dtype=np.int32)
        col_lengths = np.zeros(num_columns, dtype=np.int32)
        
        for i, col in enumerate(columns):
            if 'int' in col.type:
                # 整数型
                col_types[i] = 0
                col_lengths[i] = 4
            elif 'numeric' in col.type:
                # 数値型
                col_types[i] = 1
                col_lengths[i] = 8
                print(f"DEBUG: Numeric型カラム検出: {col.name}, type={col.type}")
                # ここではデコード前のデータがないため、後段のデコード時にデバッグが必要
            else:
                # 文字列型
                col_types[i] = 2
                # 長さを設定（最大長が指定されている場合はそれを使用、それ以外は64）
                col_lengths[i] = col.length if col.length is not None else 64
        
        # GPUバッファの割り当て
        buffers = allocate_gpu_buffers(rows_in_chunk, num_columns, col_types, col_lengths)
        
        # GPUでのデコード処理
        decoded_data = self.gpu_decoder.decode_chunk(
            buffers, 
            chunk_array, 
            field_offsets, 
            field_lengths, 
            rows_in_chunk, 
            columns
        )
        
        if not decoded_data:
            print("GPUデコードに失敗しました")
            return {}
        
        print(f"デコード完了: {len(decoded_data)}列")
        return decoded_data


class PipelinedProcessor:
    """非同期パイプラインによるPostgreSQLデータ処理クラス
    
    データの流れ: PostgreSQL → GPU → Parquet
    
    * データ取得スレッド: PostgreSQLからデータを取得し、キューに入れる
    * GPU処理スレッド: キューからデータを取り出し、GPUで処理し、結果を出力
    * 複数GPUに対応: 各GPUで別々のスレッドで処理
    """
    
    def __init__(self, table_name, db_params, output_dir, chunk_size=100000, gpu_count=None):
        """初期化
        
        Args:
            table_name: 処理対象テーブル名
            db_params: PostgreSQL接続パラメータ
            output_dir: 出力ディレクトリ
            chunk_size: チャンクサイズ（行数）
            gpu_count: 使用するGPU数（Noneの場合は自動検出）
        """
        self.table_name = table_name
        self.db_params = db_params
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        
        # GPUの数を取得または設定
        if gpu_count is None:
            try:
                from numba import cuda
                self.gpu_count = len(cuda.gpus)
                print(f"利用可能なGPU数: {self.gpu_count}")
            except Exception as e:
                print(f"GPU情報取得エラー: {e}")
                self.gpu_count = 1
        else:
            self.gpu_count = gpu_count
            
        # パイプラインキュー（データ取得スレッドと処理スレッド間の通信）
        self.queue = Queue(maxsize=self.gpu_count * 2)  # キューサイズはGPU数の2倍
        
        # 状態管理用の変数
        self.total_rows = 0
        self.processed_rows = 0
        self.total_chunks = 0
        self.current_chunk = 0
        self.is_running = False
        self.error_occurred = False
        
        # スレッド管理用の変数
        self.data_thread = None
        self.gpu_threads = []
        
        # 結果保存用の変数
        self.chunk_results = {}
        
        # PostgreSQL接続の初期化
        self.pg_conn = PostgresConnector(**self.db_params)
        
        # テーブル情報の取得
        if not self.pg_conn.check_table_exists(table_name):
            raise ValueError(f"テーブル {table_name} が存在しません")
            
        self.columns = self.pg_conn.get_table_info(table_name)
        self.row_count = self.pg_conn.get_table_row_count(table_name)
        
        # 総チャンク数の計算
        self.total_chunks = (self.row_count + self.chunk_size - 1) // self.chunk_size
        
        print(f"パイプライン初期化完了: テーブル={table_name}, 行数={self.row_count}, チャンク数={self.total_chunks}, GPU数={self.gpu_count}")
        
    def data_fetch_worker(self):
        """データ取得スレッドの処理
        
        PostgreSQLからデータを取得し、キューに入れる
        """
        try:
            # 各チャンクを処理
            for chunk_idx in range(self.total_chunks):
                if self.error_occurred:
                    break
                    
                # チャンクの範囲設定
                offset = chunk_idx * self.chunk_size
                limit = min(self.chunk_size, self.row_count - offset)
                
                if limit <= 0:
                    break
                    
                print(f"データ取得開始: チャンク {chunk_idx + 1}/{self.total_chunks} (オフセット: {offset}, 行数: {limit})")
                
                # バイナリデータの取得
                start_time = time.time()
                try:
                    binary_data, _ = self.pg_conn.get_binary_data(self.table_name, limit=limit, offset=offset)
                    fetch_time = time.time() - start_time
                    print(f"データ取得完了: チャンク {chunk_idx}, {len(binary_data)} バイト, {fetch_time:.2f}秒")
                    
                    # キューにデータを入れる（処理スレッドに渡す）
                    self.queue.put((chunk_idx, binary_data, offset, limit))
                except Exception as e:
                    print(f"データ取得エラー: チャンク {chunk_idx}, {e}")
                    self.error_occurred = True
                    # エラー通知をキューに入れる
                    self.queue.put((chunk_idx, None, offset, limit))
            
            # 終了通知をキューに入れる（GPUごとに1つずつ）
            for _ in range(self.gpu_count):
                self.queue.put((None, None, None, None))
                
        except Exception as e:
            print(f"データ取得スレッドでエラー発生: {e}")
            self.error_occurred = True
            import traceback
            traceback.print_exc()
            
            # エラー通知をキューに入れる
            for _ in range(self.gpu_count):
                self.queue.put((None, None, None, None))
    
    def gpu_process_worker(self, gpu_id):
        """GPU処理スレッドの処理
        
        Args:
            gpu_id: 使用するGPUのID
        """
        try:
            # GPUを選択
            from numba import cuda
            cuda.select_device(gpu_id)
            print(f"GPU {gpu_id} を選択しました")
            
            # バイナリパーサーの初期化
            parser = BinaryParser(use_gpu=True)
            
            # 処理ループ
            while True:
                if self.error_occurred:
                    break
                    
                # キューからデータを取り出す
                chunk_idx, binary_data, offset, limit = self.queue.get()
                
                # 終了通知を受け取ったら終了
                if chunk_idx is None:
                    print(f"GPU {gpu_id} 処理終了")
                    break
                    
                # データがない場合はスキップ
                if binary_data is None:
                    print(f"チャンク {chunk_idx} のデータがありません (GPU {gpu_id})")
                    continue
                    
                # 出力設定
                output_path = f"{self.output_dir}/{self.table_name}_chunk_{chunk_idx}_gpu{gpu_id}.parquet"
                output_handler = OutputHandler(parquet_output=output_path)
                
                try:
                    # データの解析と変換
                    start_time = time.time()
                    print(f"チャンク {chunk_idx} の処理開始 (GPU {gpu_id})")
                    
                    # バイナリデータをパース
                    result = parser.parse_postgres_binary(binary_data, self.columns)
                    
                    if not result:
                        print(f"チャンク {chunk_idx} の解析に失敗しました (GPU {gpu_id})")
                        continue
                        
                    # 行数の確認
                    row_count = len(next(iter(result.values())))
                    print(f"チャンク {chunk_idx} 解析成功: {row_count}行 (GPU {gpu_id})")
                    
                    # 結果をParquetに出力
                    output_handler.process_chunk_result(result)
                    output_handler.close()
                    
                    # 処理時間の計算
                    total_time = time.time() - start_time
                    print(f"チャンク {chunk_idx} 処理完了: {total_time:.2f}秒, {row_count / total_time:.2f} rows/sec (GPU {gpu_id})")
                    
                    # 処理済み行数を更新
                    with threading.Lock():
                        self.processed_rows += row_count
                        self.chunk_results[chunk_idx] = {
                            "output_path": output_path,
                            "row_count": row_count,
                            "processing_time": total_time
                        }
                        
                except Exception as e:
                    print(f"チャンク {chunk_idx} の処理中にエラー: {e} (GPU {gpu_id})")
                    self.error_occurred = True
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"GPU {gpu_id} 処理スレッドでエラー発生: {e}")
            self.error_occurred = True
            import traceback
            traceback.print_exc()
    
    def run(self, max_rows=None):
        """パイプライン処理を実行
        
        Args:
            max_rows: 処理する最大行数（Noneの場合は全行）
            
        Returns:
            処理結果の辞書
        """
        # 出力ディレクトリの作成
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 処理行数の制限
        if max_rows is not None:
            self.row_count = min(max_rows, self.row_count)
            self.total_chunks = (self.row_count + self.chunk_size - 1) // self.chunk_size
            
        print(f"パイプライン処理開始: {self.row_count}行, {self.total_chunks}チャンク")
        
        # 状態初期化
        self.is_running = True
        self.error_occurred = False
        self.processed_rows = 0
        self.chunk_results = {}
        
        # 時間計測開始
        total_start_time = time.time()
        
        # データ取得スレッドの開始
        self.data_thread = threading.Thread(target=self.data_fetch_worker)
        self.data_thread.start()
        
        # GPU処理スレッドの開始
        self.gpu_threads = []
        for gpu_id in range(self.gpu_count):
            thread = threading.Thread(target=self.gpu_process_worker, args=(gpu_id,))
            thread.start()
            self.gpu_threads.append(thread)
            
        # すべてのスレッドの終了を待つ
        self.data_thread.join()
        for thread in self.gpu_threads:
            thread.join()
            
        # 処理時間の計算
        total_time = time.time() - total_start_time
        
        # 結果の表示
        print(f"\n=== 処理完了 ===")
        print(f"処理行数: {self.processed_rows} / {self.row_count}")
        print(f"処理チャンク数: {len(self.chunk_results)} / {self.total_chunks}")
        print(f"合計処理時間: {total_time:.2f}秒")
        print(f"処理速度: {self.processed_rows / total_time:.2f} rows/sec")
        
        # 状態更新
        self.is_running = False
        
        return {
            "processed_rows": self.processed_rows,
            "total_rows": self.row_count,
            "processed_chunks": len(self.chunk_results),
            "total_chunks": self.total_chunks,
            "processing_time": total_time,
            "throughput": self.processed_rows / total_time if total_time > 0 else 0,
            "chunk_results": self.chunk_results,
            "error_occurred": self.error_occurred
        }
    
    def combine_outputs(self, output_path=None):
        """処理結果のParquetファイルを結合
        
        Args:
            output_path: 結合後の出力パス（Noneの場合はデフォルト値）
            
        Returns:
            結合結果
        """
        if output_path is None:
            output_path = f"{self.output_dir}/{self.table_name}_combined.parquet"
            
        # 入力ファイルの検索
        chunk_files = [result["output_path"] for result in self.chunk_results.values()]
        
        if not chunk_files:
            print("結合するParquetファイルがありません")
            return False
            
        print(f"Parquetファイル結合: {len(chunk_files)}ファイル")
        
        try:
            # cuDFでファイルを読み込んで結合
            import cudf
            
            combined_df = None
            total_rows = 0
            
            for file_path in chunk_files:
                try:
                    df = cudf.read_parquet(file_path)
                    if combined_df is None:
                        combined_df = df
                    else:
                        combined_df = cudf.concat([combined_df, df], ignore_index=True)
                        
                    total_rows += len(df)
                    print(f"  ファイル読み込み: {file_path}, {len(df)}行")
                except Exception as e:
                    print(f"  ファイル読み込みエラー: {file_path}, {e}")
                    
            # 結合したデータフレームを保存
            if combined_df is not None:
                combined_df.to_parquet(output_path)
                print(f"結合ファイル保存: {output_path}, {total_rows}行")
                return {
                    "output_path": output_path,
                    "row_count": total_rows
                }
            else:
                print("結合に失敗しました")
                return False
                
        except Exception as e:
            print(f"結合処理中にエラー: {e}")
            import traceback
            traceback.print_exc()
            return False


class BinaryDataParser:
    """バイナリデータの低レベルパーサークラス"""
    
    def __init__(self, use_gpu=True):
        """初期化
        
        Args:
            use_gpu: GPUパーサーを使用するかどうか
        """
        self.header_expected = True
        self._remaining_data = None
        self.use_gpu = use_gpu
        self.gpu_decoder = GPUDecoder() if use_gpu else None
    
    def parse_chunk(self, chunk_data: bytes, max_chunk_size: int = 1024*1024, num_columns: int = None, start_row: int = 0, max_rows: int = 65535) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """チャンク単位でのパース処理
        
        Args:
            chunk_data: バイナリデータ
            max_chunk_size: 最大チャンクサイズ
            num_columns: カラム数（Noneの場合は動的に検出）
            start_row: 処理開始行インデックス（複数チャンク対応）
            max_rows: 処理する最大行数（デフォルト: 65535）
            
        Returns:
            (chunk_array, field_offsets, field_lengths, rows_count)
        """
        # 常に完全なバイナリデータを使用
        chunk_array = np.frombuffer(chunk_data, dtype=np.uint8)
        
        # バージョンエラー防止のため、カラム数をnp.int32型に統一
        if num_columns is not None:
            num_columns = np.int32(num_columns)
        
        # ヘッダーチェックとスキップ（最初のチャンクの場合のみ）
        header_pos = 0
        if start_row == 0 and len(chunk_array) >= 11:
            # PostgreSQLバイナリフォーマットヘッダーの検出
            if chunk_array[0] == 80 and chunk_array[1] == 71:  # 'P', 'G'
                # ヘッダー本体 (11バイト) をスキップ
                header_pos = 11
                # フラグフィールド (4バイト) をスキップ
                if len(chunk_array) >= header_pos + 4:
                    header_pos += 4
                    # 拡張ヘッダー長を読み取り (4バイト, ビッグエンディアン)
                    ext_len = int.from_bytes(chunk_array[header_pos:header_pos+4], byteorder='big', signed=False)
                    header_pos += 4
                    # 拡張データをスキップ
                    if ext_len > 0 and len(chunk_array) >= header_pos + ext_len:
                        header_pos += ext_len
                print(f"ヘッダーを検出: 長さ={header_pos}バイト")
        
        # フィールド数が不明の場合は検出を試みる
        if num_columns is None and len(chunk_array) >= header_pos + 2:
            # ヘッダー後の位置から列数を検出（重要）
            b0 = np.int32(chunk_array[header_pos])
            b1 = np.int32(chunk_array[header_pos + 1])
            detected_columns = np.int32((b0 << 8) | b1)
            
            # デバッグ情報を表示
            if header_pos < len(chunk_array) - 10:
                dump_bytes = [f"{b:02x}" for b in chunk_array[header_pos:header_pos+10]]
                print(f"ヘッダー後の先頭10バイト: {' '.join(dump_bytes)}")
            
            if 0 < detected_columns < 100:  # 妥当なカラム数範囲チェック
                num_columns = detected_columns
                print(f"検出したカラム数: {num_columns}")
            else:
                print(f"警告: 検出したカラム数 {detected_columns} が範囲外です。デフォルト値を使用します。")
                num_columns = np.int32(17)  # lineorderテーブルのデフォルト値
        elif num_columns is None:
            print(f"警告: カラム数を検出できません。デフォルト値を使用します。")
            num_columns = np.int32(17)  # lineorderテーブルのデフォルト値
        
        # 開始行と最大行数をnp.int32に変換
        start_row = np.int32(start_row)
        max_rows = np.int32(max_rows)
        
        # GPU実装を使用する場合、直接GPUでパース
        if self.use_gpu and self.gpu_decoder is not None:
            try:
                print(f"GPUを使用してバイナリデータをパース（{len(chunk_array)}バイト、予想行数:{max_rows}行）")
                # 予測行数を計算（バイト数と列数から概算）
                # このnp.int32キャストは重要なので削除しないでください
                est_rows_val = min(int(max_rows), len(chunk_array) // (int(num_columns) * 20))  # 1フィールド平均20バイトと仮定
                est_rows_val = max(est_rows_val, 1000)  # 最低1000行は確保
                est_rows = np.int32(est_rows_val)
                
                # GPUパーサーを呼び出し
                field_offsets, field_lengths, rows_in_chunk = self.gpu_decoder.parse_binary_data(
                    chunk_array, est_rows, num_columns
                )
                
                # 結果の検証
                if rows_in_chunk > 0:
                    print(f"GPU処理成功: {rows_in_chunk}行解析")
                    # 最大行数の制限
                    if rows_in_chunk > max_rows:
                        print(f"警告: 行数を {max_rows} に制限します (実際の行数: {rows_in_chunk})")
                        max_fields = int(max_rows) * int(num_columns)
                        field_offsets = field_offsets[:max_fields]
                        field_lengths = field_lengths[:max_fields]
                        rows_in_chunk = np.int32(max_rows)
                    
                    return chunk_array, field_offsets, field_lengths, rows_in_chunk
                else:
                    print("GPU処理が0行を返却。CPUパースにフォールバック")
            except Exception as e:
                print(f"GPUパース中にエラー発生、CPUにフォールバック: {e}")
                import traceback
                traceback.print_exc()
        
        # GPU処理が失敗した場合やGPUを使用しない場合はCPU処理
        print("CPUを使用してバイナリデータをパース")
        
        # ヘッダーフラグは最初のチャンク（start_row=0）のみTrueに設定
        self.header_expected = (start_row == 0)
        
        # 最大試行回数を設定
        max_attempts = np.int32(3)
        attempts = np.int32(0)
        
        # ヘッダースキップ処理（最初のチャンクの場合のみ）
        current_pos = np.int32(0)
        header_len = np.int32(0)
        
        if start_row == 0 and len(chunk_array) >= 11:
            # ヘッダーが "PGCOPY\n\377\r\n\0" (バイナリフォーマット識別子) で始まるか確認
            if np.int32(chunk_array[0]) == 80 and np.int32(chunk_array[1]) == 71:  # 'P', 'G'
                header_len = np.int32(11)  # ヘッダー部分の長さ
                
                # フラグフィールドとヘッダー拡張をスキップ
                if len(chunk_array) >= header_len + 8:
                    header_len += np.int32(8)  # フラグ(4バイト) + 拡張長(4バイト)
                    
                    # ヘッダーサイズ（ヘッダー拡張）を読み取り、拡張データをスキップ
                    b0 = np.int32(chunk_array[header_len-4])
                    b1 = np.int32(chunk_array[header_len-3])
                    b2 = np.int32(chunk_array[header_len-2])
                    b3 = np.int32(chunk_array[header_len-1])
                    
                    ext_len = np.int32(0)
                    ext_len = ext_len | ((b0 & 0xFF) << 24)
                    ext_len = ext_len | ((b1 & 0xFF) << 16)
                    ext_len = ext_len | ((b2 & 0xFF) << 8)
                    ext_len = ext_len | (b3 & 0xFF)
                    
                    if ext_len > 0 and len(chunk_array) >= header_len + ext_len:
                        header_len += ext_len
                
                print(f"ヘッダーパターン検出: 長さ={header_len}バイト")
        
        # データポジション計算（ヘッダー + スキップ行分）
        if start_row > 0:
            print(f"開始行 {start_row} からのパース開始: ヘッダー長={header_len}バイト")
            
            # 現在位置をヘッダーの後ろに設定
            current_pos = np.int32(header_len)
            skipped_rows = np.int32(0)
            row_positions = []  # 各行の開始位置を記録
            
            # パフォーマンス向上のため、行位置を事前スキャン
            print(f"行位置のスキャンを開始: 開始位置={current_pos}")
            
            # 指定行数をスキップ（行単位で移動）
            while skipped_rows < start_row and current_pos + 2 <= len(chunk_array):
                # 行の開始位置を記録
                row_positions.append(current_pos)
                
                # 行の最初にあるフィールド数（2バイト）を読み取り
                b0 = np.int32(chunk_array[current_pos])
                b1 = np.int32(chunk_array[current_pos + 1])
                num_fields = np.int32((b0 << 8) | b1)
                
                if num_fields == 0xFFFF:  # ファイル終端マーカー
                    print(f"ファイル終端に到達しました（行 {skipped_rows}）")
                    return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
                
                # フィールド数の妥当性チェック
                if num_fields <= 0 or num_fields > 1000:
                    print(f"警告: 異常なフィールド数 {num_fields} (位置={current_pos})")
                    break
                
                current_pos += np.int32(2)  # フィールド数フィールドをスキップ
                
                # 各フィールドをスキップ
                valid_row = True
                for field_idx in range(num_fields):
                    if current_pos + 4 > len(chunk_array):
                        print(f"データ境界を超えました（行 {skipped_rows}, フィールド {field_idx}, 位置 {current_pos}/{len(chunk_array)}）")
                        valid_row = False
                        break
                    
                    # フィールド長を読み取り（ビッグエンディアン）
                    # 整数オーバーフローを防ぐため、np.int32を使用
                    b0 = np.int32(chunk_array[current_pos])
                    b1 = np.int32(chunk_array[current_pos+1])
                    b2 = np.int32(chunk_array[current_pos+2])
                    b3 = np.int32(chunk_array[current_pos+3])
                    
                    field_len = np.int32(0)
                    field_len = field_len | ((b0 & 0xFF) << 24)
                    field_len = field_len | ((b1 & 0xFF) << 16)
                    field_len = field_len | ((b2 & 0xFF) << 8)
                    field_len = field_len | (b3 & 0xFF)
                    
                    current_pos += np.int32(4)  # フィールド長フィールドをスキップ
                    
                    if field_len == -1:  # NULLフィールド
                        continue
                    elif field_len < 0:
                        print(f"無効なフィールド長: {field_len} (行 {skipped_rows}, フィールド {field_idx})")
                        valid_row = False
                        break
                    
                    # バッファ境界チェック
                    if current_pos + field_len > len(chunk_array):
                        print(f"フィールドデータがバッファ境界を超えています（行 {skipped_rows}, フィールド {field_idx}, 長さ {field_len}, 位置 {current_pos}/{len(chunk_array)}）")
                        valid_row = False
                        break
                    
                    # フィールドデータをスキップ
                    current_pos += field_len
                
                # 行が正常に処理できた場合のみカウント
                if valid_row:
                    skipped_rows += 1
                else:
                    # 無効な行があった場合は処理を中止
                    break
            
            if skipped_rows < start_row:
                print(f"指定された開始行 {start_row} まで到達できませんでした（スキップした行数: {skipped_rows}）")
                return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
            
            # 開始行が見つかった場合、そこからデータを切り出す
            if len(row_positions) >= start_row:
                start_pos = row_positions[start_row - 1] if start_row > 0 else header_len
                print(f"行 {start_row} の開始位置: {start_pos}")
                
                # 現在位置以降のデータで新しい配列を作成
                if start_pos > 0:
                    chunk_array = chunk_array[start_pos:]
                    print(f"パース開始位置: {start_pos}, 残りデータ: {len(chunk_array)} バイト")
        
        # ヘッダー処理フラグの設定（start_row=0の時はヘッダーを期待する）
        self.header_expected = (start_row == 0)
        
        # デバッグ情報の出力
        print(f"ヘッダー処理フラグ: {self.header_expected}, 開始位置: {current_pos}")
            
        while attempts < max_attempts:
            field_offsets, field_lengths = parse_binary_chunk(chunk_array, self.header_expected)
            self.header_expected = False  # 2回目以降はヘッダーを期待しない

            # CPUデバッグ: lo_extendedprice列のndigitsとweightをログ出力
            if field_offsets is not None and len(field_offsets) >= num_columns:
                ext_idx = 9  # lo_extendedpriceカラムのインデックス (0-based)
                field_offset = int(field_offsets[ext_idx])
                header_pos = field_offset
                if header_pos >= 0 and header_pos + 4 <= len(chunk_array):
                    nd = (int(chunk_array[header_pos]) << 8) | int(chunk_array[header_pos+1])
                    wt = (int(chunk_array[header_pos+2]) << 8) | int(chunk_array[header_pos+3])
                    print(f"DEBUG CPU numeric header(lo_extendedprice): ndigits={nd}, weight={wt}")
                else:
                    print("DEBUG CPU numeric header: header_pos範囲外")
            else:
                print("DEBUG CPU numeric header: field_offsets不足またはNone")
            
            # 完全な行が得られた場合
            if len(field_offsets) > 0:
                last_field_end = field_offsets[-1] + max(0, field_lengths[-1])
                if last_field_end <= len(chunk_array):
                    # 次のチャンクのために残りデータを保存
                    if last_field_end < len(chunk_array):
                        self._remaining_data = chunk_array[last_field_end:]
                    break
            
            # 追加データが必要な場合（実際の実装では追加データを読み込む処理が入る）
            # この例では単純化のためbreak
            break
        
        # パース完了時の処理
        if len(field_offsets) == 0:
            return None, None, None, 0
            
        # フィールド数から行数を計算
        if num_columns is None:
            # フィールド数が不明の場合は、検出を試みる
            # チャンク配列の先頭から最初の行のフィールド数を読み込む
            if len(chunk_array) >= 2:
                detected_columns = ((chunk_array[0] << 8) | chunk_array[1])
                if 0 < detected_columns < 100:  # 妥当なカラム数範囲チェック
                    num_columns = detected_columns
                    print(f"検出したカラム数: {num_columns}")
                else:
                    print(f"警告: 検出したカラム数 {detected_columns} が範囲外です。デフォルト値を使用します。")
                    num_columns = 8  # customerテーブルのデフォルト値
            else:
                print(f"警告: カラム数を検出できません。デフォルト値を使用します。")
                num_columns = 8  # customerテーブルのデフォルト値
        
        if num_columns <= 0:
            num_columns = 1  # 最低でも1列あると仮定
        
        # 最大行数の制限を適用
        total_fields = len(field_offsets)
        
        # カラム数と総フィールド数から行数を計算
        if total_fields == 0:
            print("警告: フィールドが検出されませんでした")
            return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
            
        if total_fields % num_columns != 0:
            print(f"警告: 総フィールド数 {total_fields} がカラム数 {num_columns} の倍数ではありません")
            # 完全な行のみを処理するために、あふれたフィールドを除外
            rows_in_chunk = total_fields // num_columns
            total_fields = rows_in_chunk * num_columns
            field_offsets = field_offsets[:total_fields]
            field_lengths = field_lengths[:total_fields]
        else:
            rows_in_chunk = total_fields // num_columns
        
        print(f"検出した行数: {rows_in_chunk} (フィールド数: {total_fields}, カラム数: {num_columns})")
        
        if rows_in_chunk > max_rows:
            print(f"警告: 行数を {max_rows} に制限します (実際の行数: {rows_in_chunk})")
            # フィールド数と行数を制限
            max_fields = max_rows * num_columns
            field_offsets = field_offsets[:max_fields]
            field_lengths = field_lengths[:max_fields]
            rows_in_chunk = max_rows
        
        if rows_in_chunk > 0:
            print(f"処理完了: {rows_in_chunk}行 ({total_fields}フィールド)")
            return chunk_array, field_offsets, field_lengths, rows_in_chunk
        else:
            print("警告: 有効な行が検出されませんでした")
            return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
