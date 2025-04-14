"""
GPU上でのPostgreSQLバイナリデータデコーダー
"""

import numpy as np
from numba import cuda
import math
from typing import List, Dict, Any, Optional, Tuple

# 分割したモジュールをインポート
from gpupaser.cuda_kernels import parse_binary_format_kernel
from gpupaser.data_processors import decode_all_columns_kernel


class GPUDecoder:
    """GPU上でのデコード処理を管理するクラス"""
    
    def __init__(self):
        """初期化"""
        # CUDAコンテキストを初期化
        try:
            # 明示的なデバイス選択を削除
            # Rayが設定した環境変数により、割り当てられたGPUが自動的に選択される
            print("CUDA device initialized for GPUDecoder (using Ray-assigned GPU)")
        except Exception as e:
            print(f"CUDA初期化エラー: {e}")
    
    def parse_binary_data(self, chunk_array, rows_in_chunk, num_cols):
        """
        PostgreSQLバイナリデータをGPUで解析する
        
        Args:
            chunk_array: バイナリデータ配列 (numpyバイト配列)
            rows_in_chunk: 予想される行数 (エラー検出用)
            num_cols: 1行あたりのカラム数
            
        Returns:
            field_offsets: フィールド位置の配列
            field_lengths: フィールド長の配列
            actual_rows: 実際に解析できた行数
        """
        try:
            # 整数型に変換（重要）
            rows_in_chunk_val = int(rows_in_chunk)
            num_cols_val = int(num_cols)
            
            # バイト数は安全に取得
            array_size = int(chunk_array.size if hasattr(chunk_array, 'size') else len(chunk_array))
            
            print(f"GPUでバイナリデータ解析を開始: {array_size}バイト、予想行数={rows_in_chunk_val}、列数={num_cols_val}")
            
            # フィールドオフセットと長さの配列を確保（データ型はint32）
            total_fields = rows_in_chunk_val * num_cols_val
            d_field_offsets = cuda.device_array(total_fields, dtype=np.int32)
            d_field_lengths = cuda.device_array(total_fields, dtype=np.int32)
            
            # チャンク配列をGPUに転送
            d_chunk_array = cuda.to_device(chunk_array)
            
            # スレッド数とブロック数を最適化（Ampereアーキテクチャ向け）
            threads_per_block = 1024  # 最大スレッド数に設定
            
            # デバイスのプロパティを取得
            try:
                device_props = cuda.get_current_device().get_attributes()
                sm_count = device_props['MultiProcessorCount']
                # A100/A30等ではSMあたり2ブロックが最適
                blocks_per_sm = 2
                blocks = min(2048, sm_count * blocks_per_sm)
                print(f"GPU特性: SM数={sm_count}、使用ブロック数={blocks}")
            except:
                # 既定値でフォールバック
                blocks = min(2048, (array_size + threads_per_block - 1) // threads_per_block)
            
            print(f"最適化されたCUDA設定: スレッド/ブロック={threads_per_block}、ブロック数={blocks}")
            
            # 共有メモリの確保（ヘッダーサイズと行位置情報用）
            header_shared_size = 2 + 100  # ヘッダーサイズ + 総行数 + 最大100行の位置
            d_header_shared = cuda.device_array(header_shared_size, dtype=np.int32)
            
            # 結果を格納する変数を前もって初期化
            field_offsets = np.array([], dtype=np.int32)
            field_lengths = np.array([], dtype=np.int32)
            header_shared = np.zeros(header_shared_size, dtype=np.int32)
            
            try:
                # パースカーネルを起動
                parse_binary_format_kernel[blocks, threads_per_block](
                    d_chunk_array, d_field_offsets, d_field_lengths, 
                    np.int32(num_cols_val), 
                    d_header_shared
                )
                cuda.synchronize()  # カーネル完了を待つ
                
                # 結果をホストにコピー
                field_offsets = d_field_offsets.copy_to_host()
                field_lengths = d_field_lengths.copy_to_host()
                
                # 共有メモリから検出された行数を取得
                header_shared = d_header_shared.copy_to_host()
                
            except Exception as kernel_err:
                print(f"カーネル実行中のエラー: {kernel_err}")
                import traceback
                traceback.print_exc()
                # 例外が発生しても続行（空の配列を返す準備はできている）
            finally:
                # リソース解放（try ブロックの外で必ず実行）
                # 各変数が存在するか確認してから削除
                if 'd_chunk_array' in locals():
                    del d_chunk_array
                if 'd_field_offsets' in locals():
                    del d_field_offsets
                if 'd_field_lengths' in locals():
                    del d_field_lengths
                if 'd_header_shared' in locals():
                    del d_header_shared
                cuda.synchronize()
            
            # 有効なフィールドを検出
            # フィールド数からユニークな行数を計算
            actual_rows = min(rows_in_chunk_val, total_fields // num_cols_val)
            detected_rows = int(header_shared[1])
            
            if detected_rows == 0:
                print(f"有効な行が検出されませんでした")
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
            
            # 検出行数が実際のバッファサイズを超えていないか確認
            max_possible_rows = min(detected_rows, len(field_offsets) // num_cols_val)
            valid_rows = max_possible_rows
            
            # 有効なフィールド数をカウント (デバッグ用)
            valid_fields = np.sum(field_offsets[:valid_rows * num_cols_val] > 0)
            
            print(f"バイナリデータ解析完了: 有効行数={valid_rows}, フィールド数={valid_fields}")
            
            # 有効なデータが含まれる部分のみを返す
            return field_offsets[:valid_rows * num_cols_val], field_lengths[:valid_rows * num_cols_val], valid_rows
            
        except Exception as e:
            print(f"GPUバイナリ解析中にエラー: {e}")
            import traceback
            traceback.print_exc()
            # エラー時は空の配列を返す
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
    
    def decode_chunk(self, buffers, chunk_array, field_offsets, field_lengths, rows_in_chunk, columns):
        """チャンクをデコードする
        
        Args:
            buffers: GPUバッファ辞書
            chunk_array: バイナリデータ配列
            field_offsets: フィールドオフセット配列
            field_lengths: フィールド長配列
            rows_in_chunk: チャンク内の行数
            columns: カラム情報リスト
            
        Returns:
            デコード結果
        """
        # 型変換を保証
        num_cols = np.int32(len(columns))
        rows_in_chunk = np.int32(rows_in_chunk)
        
        # 転送前にすべての変数を初期化（解放処理のため）
        d_chunk = None
        d_offsets = None
        d_lengths = None
        d_col_types = None
        d_col_lengths = None
        
        try:
            # メモリ転送前のGPUメモリ状態を確認
            try:
                mem_info = cuda.current_context().get_memory_info()
                print(f"メモリ使用量 (転送前): {mem_info[0]} / {mem_info[1]} バイト")
            except Exception as e:
                print(f"メモリ情報取得エラー: {e}")
            
            # デバイスに転送（通常はmemory_managerで行うが、ここではシンプルに直接処理）
            d_chunk = cuda.to_device(chunk_array)
            d_offsets = cuda.to_device(field_offsets)
            d_lengths = cuda.to_device(field_lengths)
            cuda.synchronize()  # メモリ転送の完了を待つ
            
            # メモリ転送後のGPUメモリ状態を確認
            try:
                mem_info = cuda.current_context().get_memory_info()
                print(f"メモリ使用量 (転送後): {mem_info[0]} / {mem_info[1]} バイト")
            except Exception as e:
                print(f"メモリ情報取得エラー: {e}")
            
            # バッファの取得
            int_buffer = buffers["int_buffer"]
            num_hi_output = buffers["num_hi_output"]
            num_lo_output = buffers["num_lo_output"]
            num_scale_output = buffers["num_scale_output"]
            str_buffer = buffers["str_buffer"]
            str_null_pos = buffers["str_null_pos"]
            d_str_offsets = buffers["d_str_offsets"]
            col_types = buffers["col_types"]
            col_lengths = buffers["col_lengths"]
            
            # デバイスに転送
            d_col_types = cuda.to_device(col_types)
            d_col_lengths = cuda.to_device(col_lengths)
            
            # スレッド数とブロック数の調整
            threads_per_block = 256  # 固定スレッド数
            blocks = min(
                1024,  # ブロック数の制限を緩和
                max(128, (rows_in_chunk + threads_per_block - 1) // threads_per_block)  # 最小128ブロック
            )
            print(f"Using {blocks} blocks with {threads_per_block} threads per block")
                
            # 統合カーネルの起動
            decode_all_columns_kernel[blocks, threads_per_block](
                d_chunk, d_offsets, d_lengths,
                int_buffer, num_hi_output, num_lo_output, num_scale_output,
                str_buffer, str_null_pos, d_str_offsets,
                d_col_types, d_col_lengths,
                rows_in_chunk, num_cols
            )
            
            # 結果を取得するための一時辞書
            results = {}
            
            # 結果の回収
            int_col_idx = 0
            str_col_idx = 0
            
            for i, col in enumerate(columns):
                col_type = col_types[i]
                if col_type <= 1:  # 数値型（integer or numeric）
                    if col_type == 0:  # integer
                        if int_buffer is not None:
                            host_array = np.empty(rows_in_chunk, dtype=np.int32)
                            int_buffer[int_col_idx * rows_in_chunk:(int_col_idx + 1) * rows_in_chunk].copy_to_host(host_array)
                            results[col.name] = host_array
                            int_col_idx += 1
                    else:  # numeric
                        if num_hi_output is not None and num_lo_output is not None:
                            host_hi = np.empty(rows_in_chunk, dtype=np.int64)
                            host_lo = np.empty(rows_in_chunk, dtype=np.int64)
                            host_scale = np.empty(1, dtype=np.int32)
                            
                            num_hi_output[int_col_idx * rows_in_chunk:(int_col_idx + 1) * rows_in_chunk].copy_to_host(host_hi)
                            num_lo_output[int_col_idx * rows_in_chunk:(int_col_idx + 1) * rows_in_chunk].copy_to_host(host_lo)
                            num_scale_output[int_col_idx:int_col_idx + 1].copy_to_host(host_scale)
                            
                            values = []
                            for j in range(rows_in_chunk):
                                hi = host_hi[j]
                                lo = host_lo[j]
                                scale = host_scale[0]
                                
                                if hi == 0 and lo >= 0:
                                    value = float(lo) / (10 ** scale) if scale > 0 else float(lo)
                                elif hi == -1 and lo < 0:
                                    value = float(lo) / (10 ** scale) if scale > 0 else float(lo)
                                else:
                                    value = f"[{hi},{lo}]@{scale}"
                                
                                values.append(value)
                            
                            results[col.name] = values
                            int_col_idx += 1
                else:  # 文字列型
                    if str_buffer is not None and str_null_pos is not None:
                        length = col_lengths[i]
                        
                        # ホスト側でオフセット値を取得
                        # まず配列サイズを確認（numba.cudadevicearray.DeviceNDArrayのlen問題を回避）
                        try:
                            if hasattr(d_str_offsets, 'size'):
                                d_str_offsets_size = d_str_offsets.size
                            else:
                                # サイズが不明の場合は文字列カラム数+1と仮定
                                d_str_offsets_size = str_col_idx + 1
                            host_offsets = np.empty(d_str_offsets_size, dtype=np.int32)
                            d_str_offsets.copy_to_host(host_offsets)
                        except Exception as e:
                            print(f"オフセット取得エラー: {e}")
                            # エラー発生時は文字列カラムの数+1分の要素を持つ配列を生成
                            num_str_cols = sum(1 for t in col_types if t > 1)
                            host_offsets = np.zeros(num_str_cols + 1, dtype=np.int32)
                            # 各文字列カラムのオフセットを計算（バッファサイズを均等配分）
                            if num_str_cols > 0:
                                col_lens = [col_lengths[i] for i, t in enumerate(col_types) if t > 1]
                                total_len = sum(col_lens)
                                cum_len = 0
                                str_idx = 0
                                for i, t in enumerate(col_types):
                                    if t > 1:  # 文字列型
                                        host_offsets[str_idx] = int(cum_len * rows_in_chunk)
                                        cum_len += (col_lengths[i] / total_len) if total_len > 0 else 0
                                        str_idx += 1
                        
                        str_start = host_offsets[str_col_idx]
                        str_end = str_start + (rows_in_chunk * length)
                        
                        # 文字列バッファ部分のコピー
                        host_str_array = np.empty(rows_in_chunk * length, dtype=np.uint8)
                        str_buffer[str_start:str_end].copy_to_host(host_str_array)
                        
                        # NULL位置の配列を取得
                        host_null_pos = np.empty(rows_in_chunk, dtype=np.int32)
                        str_null_pos[str_col_idx * rows_in_chunk:(str_col_idx + 1) * rows_in_chunk].copy_to_host(host_null_pos)
                        
                        # 文字列リストに変換
                        strings = []
                        for j in range(rows_in_chunk):
                            null_pos = host_null_pos[j]
                            if null_pos == 0:  # NULL値
                                strings.append(None)
                            else:
                                start_idx = j * length
                                end_idx = start_idx + null_pos
                                
                                # バイト配列を取得してUTF-8でデコード
                                # エラー処理を追加（不正なUTF-8シーケンスを置換）
                                byte_data = host_str_array[start_idx:end_idx].tobytes()
                                
                                try:
                                    # PostgreSQLバイナリデータを直接デコード
                                    if byte_data:
                                        # NULL終端の検索や除去を行わない
                                        # PostgreSQLは長さフィールドで文字列長を管理しているため
                                        string_val = byte_data.decode('utf-8', errors='replace')
                                    else:
                                        string_val = ""  # 空のバイト配列の場合
                                    
                                    strings.append(string_val)
                                except Exception as e:
                                    print(f"文字列デコードエラー: {e}, バイトデータ: {byte_data}")
                                    # デコードに失敗した場合は空文字列
                                    strings.append("")
                        
                        results[col.name] = strings
                        str_col_idx += 1
            
            # リソース解放
            if d_chunk is not None:
                del d_chunk
            if d_offsets is not None:
                del d_offsets
            if d_lengths is not None:
                del d_lengths
            if d_col_types is not None:
                del d_col_types
            if d_col_lengths is not None:
                del d_col_lengths
            cuda.synchronize()
            
            return results
            
        except Exception as e:
            print(f"GPUデコード中にエラー: {e}")
            # エラー時はリソース解放
            if d_chunk is not None:
                del d_chunk
            if d_offsets is not None:
                del d_offsets
            if d_lengths is not None:
                del d_lengths
            if d_col_types is not None:
                del d_col_types
            if d_col_lengths is not None:
                del d_col_lengths
            cuda.synchronize()
            
            # 空の結果を返す
            return {}
