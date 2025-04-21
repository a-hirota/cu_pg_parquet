"""
GPU上でのPostgreSQLバイナリデータデコーダー
"""

import numpy as np
from numba import cuda
import math
import time
from typing import List, Dict, Any, Optional, Tuple

# 分割したモジュールをインポート
from gpupaser.cuda_kernels import parse_binary_format_kernel, parse_binary_format_kernel_one_row
from gpupaser.data_processors import decode_all_columns_kernel

class GPUDecoder:
    """GPU上でのデコード処理を管理するクラス"""
    
    def __init__(self):
        """初期化"""
        try:
            # Rayが設定した環境変数により、割り当てられたGPUが自動的に選択される
            print("CUDA device initialized for GPUDecoder (using Ray-assigned GPU)")
        except Exception as e:
            print(f"CUDA初期化エラー: {e}")
    
    def parse_binary_data(self, chunk_array, rows_in_chunk, num_cols, use_one_row_per_thread=True):
        """
        PostgreSQLバイナリデータをGPUで解析する
        
        Args:
            chunk_array: バイナリデータ配列 (numpyバイト配列)
            rows_in_chunk: 予想される行数 (エラー検出用)
            num_cols: 1行あたりのカラム数（データと不一致の場合は動的検出）
            use_one_row_per_thread: 1スレッド1行の最適化カーネルを使用するか
            
        Returns:
            field_offsets: フィールド位置の配列
            field_lengths: フィールド長の配列
            actual_rows: 実際に解析できた行数
        """
        # 全体の開始時間計測
        total_start_time = time.perf_counter()
        try:
            rows_in_chunk_val = int(rows_in_chunk)
            num_cols_val = int(num_cols)
            
            # バイト数の取得
            array_size = int(chunk_array.size if hasattr(chunk_array, "size") else len(chunk_array))
            
            # 動的列数検出
            detected_cols = self._detect_column_count_from_data(chunk_array)
            if detected_cols > 0 and detected_cols != num_cols_val:
                print(f"データから検出した列数: {detected_cols}（指定された列数と異なる: {num_cols_val}）")
                num_cols_val = detected_cols
            
            print(f"GPUでバイナリデータ解析を開始: {array_size}バイト、予想行数={rows_in_chunk_val}、列数={num_cols_val}", flush=True)
            import binascii
            print("DEBUG: First 128 bytes of binary data:", flush=True)
            print(binascii.hexlify(chunk_array[:128]), flush=True)
            
            # メモリ確保開始時間計測
            alloc_start = time.perf_counter()
            
            print("=== GPUメモリ確保開始 ===", flush=True)
            total_fields = rows_in_chunk_val * num_cols_val
            d_field_offsets = cuda.device_array(total_fields, dtype=np.int32)
            d_field_lengths = cuda.device_array(total_fields, dtype=np.int32)
            
            # メモリ確保時間計測
            alloc_time = time.perf_counter() - alloc_start
            print(f"[TIMING] GPUメモリ確保時間: {alloc_time*1000:.2f} ms", flush=True)
            
            # データ転送開始時間計測
            transfer_start = time.perf_counter()
            print("=== GPU転送開始 ===", flush=True)
            
            d_chunk_array = cuda.to_device(chunk_array)
            
            # データ転送時間計測
            transfer_time = time.perf_counter() - transfer_start
            transfer_size = chunk_array.nbytes / (1024*1024)  # MB単位
            print(f"[TIMING] データ転送時間 (CPU→GPU): {transfer_time*1000:.2f} ms", flush=True)
            print(f"[TIMING] 転送速度: {transfer_size/transfer_time:.2f} MB/s", flush=True)
            
            # スレッド・ブロック設定
            import os
            
            # 環境変数から設定を読み取り (優先的に使用)
            threads_per_block = int(os.environ.get("GPUPASER_THREAD_COUNT", "1024"))
            block_size = int(os.environ.get("GPUPASER_BLOCK_SIZE", "256"))
            
            # GPUプロパティの取得を試行 (補助情報として使用)
            try:
                device_props = cuda.get_current_device().get_attributes()
                sm_count = device_props["MultiProcessorCount"]
                max_threads_per_block = device_props.get("MaxThreadsPerBlock", 1024)
                print(f"GPU特性: SM数={sm_count}、最大スレッド数={max_threads_per_block}")
                
                # 環境変数の設定がない場合のみGPU特性に基づく計算を使用
                if "GPUPASER_BLOCK_SIZE" not in os.environ:
                    blocks_per_sm = 2
                    blocks = min(2048, sm_count * blocks_per_sm)
                else:
                    # 環境変数から取得した値を使用
                    blocks = block_size
            except Exception as e:
                print(f"GPUプロパティ取得エラー: {e}、設定値を使用")
                # 環境変数の値またはデフォルト値を使用
                blocks = block_size
            
            print(f"最適化されたCUDA設定: スレッド/ブロック={threads_per_block}、ブロック数={blocks}", flush=True)
            print("=== カーネル実行開始 ===", flush=True)
            
            # 共有メモリの確保
            header_shared_size = 2 + 100  # ヘッダーサイズ + 行数情報 + 追加スペース
            d_header_shared = cuda.device_array(header_shared_size, dtype=np.int32)
            
            # 初期化用ホスト変数
            field_offsets = np.array([], dtype=np.int32)
            field_lengths = np.array([], dtype=np.int32)
            header_shared = np.zeros(header_shared_size, dtype=np.int32)
            
            try:
                # カーネル実行時間計測用イベント
                start_event = cuda.event()
                end_event = cuda.event()
                
                # カーネル実行開始時間計測
                kernel_start = time.perf_counter()
                
                # イベント記録開始
                start_event.record()
                
                # カーネルタイプによって異なるカーネルを使用
                if use_one_row_per_thread:
                    print("=== 1スレッド1行モードで実行 ===", flush=True)
                    # ヘッダーサイズ取得（ヘッダー検出から取得）
                    header_size = self._detect_header_size(chunk_array)
                    
                    # 1スレッド1行カーネル用にスレッド数を行数に合わせて調整
                    rows_threads = rows_in_chunk_val
                    blocks_needed = (rows_threads + threads_per_block - 1) // threads_per_block
                    blocks = min(blocks, blocks_needed * 2)  # 余裕を持たせる
                    
                    # カーネル起動（1スレッド1行モード）
                    # 空の行位置配列を作成（Noneの代わり）
                    empty_row_positions = cuda.device_array(0, dtype=np.int32)
                    
                    # GPU実行に必要なすべてのパラメータを設定
                    parse_binary_format_kernel_one_row[blocks, threads_per_block](
                        d_chunk_array, d_field_offsets, d_field_lengths, 
                        np.int32(num_cols_val), np.int32(header_size), empty_row_positions
                    )
                else:
                    print("=== 従来のモードで実行 ===", flush=True)
                    # 従来の（複数行担当）カーネル起動
                    parse_binary_format_kernel[blocks, threads_per_block](
                        d_chunk_array, d_field_offsets, d_field_lengths, 
                        np.int32(num_cols_val), d_header_shared
                    )
                
                # イベント記録終了
                end_event.record()
                end_event.synchronize()
                
                # カーネル実行時間（CUDA イベントベース）
                kernel_time_ms = cuda.event_elapsed_time(start_event, end_event)
                print(f"[TIMING] カーネル実行時間 (CUDA計測): {kernel_time_ms:.2f} ms", flush=True)
                
                cuda.synchronize()
                
                # カーネル実行時間（Python計測）
                kernel_time = time.perf_counter() - kernel_start
                print(f"[TIMING] カーネル実行時間 (Python計測): {kernel_time*1000:.2f} ms", flush=True)
                
                # 結果転送開始時間計測
                result_transfer_start = time.perf_counter()
                print("=== 結果転送開始 ===", flush=True)
                
                # 結果をホストにコピー
                field_offsets = d_field_offsets.copy_to_host()
                field_lengths = d_field_lengths.copy_to_host()
                header_shared = d_header_shared.copy_to_host()
                
                # 結果転送時間計測
                result_transfer_time = time.perf_counter() - result_transfer_start
                result_size = (field_offsets.nbytes + field_lengths.nbytes + header_shared.nbytes) / (1024*1024)  # MB単位
                print(f"[TIMING] 結果転送時間 (GPU→CPU): {result_transfer_time*1000:.2f} ms", flush=True)
                print(f"[TIMING] 結果転送速度: {result_size/result_transfer_time:.2f} MB/s", flush=True)
            except Exception as kernel_err:
                print(f"カーネル実行中のエラー: {kernel_err}")
                import traceback
                traceback.print_exc()
            finally:
                if "d_chunk_array" in locals():
                    del d_chunk_array
                if "d_field_offsets" in locals():
                    del d_field_offsets
                if "d_field_lengths" in locals():
                    del d_field_lengths
                if "d_header_shared" in locals():
                    del d_header_shared
                cuda.synchronize()
            
            actual_rows = min(rows_in_chunk_val, total_fields // num_cols_val)
            detected_rows = int(header_shared[1])
            
            if detected_rows == 0:
                print("有効な行が検出されませんでした")
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
            
            max_possible_rows = min(detected_rows, len(field_offsets) // num_cols_val)
            valid_rows = max_possible_rows
            valid_fields = np.sum(field_offsets[:valid_rows * num_cols_val] > 0)
            
            # 全体処理時間計測
            total_time = time.perf_counter() - total_start_time
            
            print(f"バイナリデータ解析完了: 有効行数={valid_rows}, フィールド数={valid_fields}")
            print(f"[TIMING] 全体処理時間: {total_time*1000:.2f} ms")
            print(f"[TIMING] 処理速度: {valid_rows/total_time:.2f} rows/sec")
            
            return field_offsets[:valid_rows * num_cols_val], field_lengths[:valid_rows * num_cols_val], valid_rows
        except Exception as e:
            print(f"GPUデータ解析中にエラー: {e}")
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
    
    def _detect_header_size(self, data):
        """
        バイナリデータからPostgreSQLヘッダーサイズを検出する
        
        Args:
            data: バイナリデータ配列
            
        Returns:
            ヘッダーサイズ（バイト数）
        """
        try:
            if len(data) < 11:
                return 0
                
            # PostgreSQLバイナリフォーマットのヘッダー検出
            header_size = 11
            
            # フラグと拡張ヘッダー長を処理
            if len(data) >= header_size + 8:
                header_size += 4  # フラグ
                ext_len = (data[header_size] << 24) | (data[header_size + 1] << 16) | \
                          (data[header_size + 2] << 8) | data[header_size + 3]
                header_size += 4  # 拡張長
                
                if ext_len > 0 and len(data) >= header_size + ext_len:
                    header_size += ext_len
            
            print(f"検出したヘッダーサイズ: {header_size}バイト")
            return header_size
        except Exception as e:
            print(f"ヘッダーサイズ検出中にエラー: {e}")
            return 19  # PostgreSQLのデフォルトヘッダーサイズ（基本+フラグ+拡張長）
    
    def _detect_column_count_from_data(self, data):
        """
        データから列数を検出する
        
        Args:
            data: バイナリデータ配列
            
        Returns:
            検出された列数（検出できない場合は0）
        """
        try:
            # ヘッダーサイズを動的に計算
            if len(data) < 11:
                return 0
                
            # PostgreSQLバイナリフォーマットのヘッダー検出
            header_size = 11
            
            # フラグと拡張ヘッダー長を処理
            if len(data) >= header_size + 8:
                header_size += 4  # フラグ
                ext_len = (data[header_size] << 24) | (data[header_size + 1] << 16) | \
                          (data[header_size + 2] << 8) | data[header_size + 3]
                header_size += 4  # 拡張長
                
                if ext_len > 0 and len(data) >= header_size + ext_len:
                    header_size += ext_len
            
            # ヘッダーの後から有効な列数を検索
            pos = header_size
            max_scan = min(len(data) - 2, header_size + 1000)  # 最大1000バイトまでスキャン
            
            # データのダンプ（デバッグ用）
            if pos < len(data) - 10:
                dump_bytes = [f"{b:02x}" for b in data[pos:pos+10]]
                print(f"ヘッダー後の先頭10バイト: {' '.join(dump_bytes)}")
            
            # 最初の有効な行を探す
            while pos < max_scan:
                field_count = (data[pos] << 8) | data[pos + 1]
                
                if field_count == 0xFFFF:  # 終端マーカー
                    break
                    
                if field_count > 0 and field_count < 100:
                    print(f"データから有効な列数を検出: {field_count}")
                    return field_count
                
                pos += 1
                
            # 列数を検出できなかった場合のフォールバック
            print(f"有効な列数を検出できませんでした（ヘッダーサイズ: {header_size}）")
            return 0
        except Exception as e:
            print(f"列数検出中にエラー: {e}")
            return 0
    
    def decode_chunk(self, buffers, chunk_array, field_offsets, field_lengths, rows_in_chunk, columns):
        """
        チャンクをデコードする
        
        性能計測付きのデコード処理
        
        Args:
            buffers: GPUバッファ辞書
            chunk_array: バイナリデータ配列
            field_offsets: フィールドオフセット配列
            field_lengths: フィールド長配列
            rows_in_chunk: チャンク内の行数
            columns: カラム情報リスト
            
        Returns:
            デコード結果の辞書
        """
        num_cols = np.int32(len(columns))
        rows_in_chunk = np.int32(rows_in_chunk)
        
        d_chunk = None
        d_offsets = None
        d_lengths = None
        d_col_types = None
        d_col_lengths = None
        
        stream1 = cuda.stream()
        stream2 = cuda.stream()
        
        try:
            # データ転送開始時間計測
            transfer_start = time.perf_counter()
            
            try:
                mem_info = cuda.current_context().get_memory_info()
                print(f"メモリ使用量 (転送前): {mem_info[0]} / {mem_info[1]} バイト")
            except Exception as e:
                print(f"メモリ情報取得エラー: {e}")
            
            d_chunk = cuda.to_device(chunk_array, stream=stream1)
            d_offsets = cuda.to_device(field_offsets, stream=stream1)
            d_lengths = cuda.to_device(field_lengths, stream=stream1)
            
            # データ転送時間計測
            transfer_time = time.perf_counter() - transfer_start
            transfer_size = (chunk_array.nbytes + field_offsets.nbytes + field_lengths.nbytes) / (1024*1024)
            print(f"[TIMING] データ転送時間 (CPU→GPU): {transfer_time*1000:.2f} ms 【{transfer_size:.2f} MB, {transfer_size/transfer_time:.2f} MB/s】")
            
            int_buffer = buffers["int_buffer"]
            num_hi_output = buffers["num_hi_output"]
            num_lo_output = buffers["num_lo_output"]
            num_scale_output = buffers["num_scale_output"]
            str_buffer = buffers["str_buffer"]
            str_null_pos = buffers["str_null_pos"]
            d_str_offsets = buffers["d_str_offsets"]
            col_types = buffers["col_types"]
            col_lengths = buffers["col_lengths"]
            
            d_col_types = cuda.to_device(col_types, stream=stream1)
            d_col_lengths = cuda.to_device(col_lengths, stream=stream1)
            
            stream1.synchronize()
            
            try:
                mem_info = cuda.current_context().get_memory_info()
                print(f"メモリ使用量 (転送後): {mem_info[0]} / {mem_info[1]} バイト")
            except Exception as e:
                print(f"メモリ情報取得エラー: {e}")
            
            # 環境変数から設定を読み取り (優先的に使用)
            import os
            threads_per_block = int(os.environ.get("GPUPASER_THREAD_COUNT", "1024"))
            block_size = int(os.environ.get("GPUPASER_BLOCK_SIZE", "256"))
            
            try:
                device_props = cuda.get_current_device().get_attributes()
                sm_count = device_props["MultiProcessorCount"]
                max_threads_per_block = device_props.get("MaxThreadsPerBlock", 1024)
                print(f"GPU特性: SM数={sm_count}, 最大スレッド数={max_threads_per_block}")
                
                # 環境変数の設定がない場合のみGPU特性に基づく計算を使用
                if "GPUPASER_BLOCK_SIZE" not in os.environ:
                    blocks_per_sm = 16
                    blocks = min(8192, sm_count * blocks_per_sm)
                    min_blocks = max(128, (rows_in_chunk + threads_per_block - 1) // threads_per_block)
                    blocks = max(min_blocks, blocks)
                else:
                    # 環境変数から取得した値を使用
                    blocks = block_size
                print(f"最適化設定: スレッド数={threads_per_block}, ブロック数={blocks}")
            except Exception as e:
                print(f"GPUプロパティ取得エラー: {e}, 設定値を使用")
                blocks = block_size
            
            print(f"Using {blocks} blocks with {threads_per_block} threads per block")
            
            # カーネル実行時間計測用イベント
            start_event = cuda.event()
            end_event = cuda.event()
            
            # カーネル実行開始時間計測
            kernel_start = time.perf_counter()
            
            # イベント記録開始
            start_event.record()
            
            decode_all_columns_kernel[blocks, threads_per_block, stream2](
                d_chunk, d_offsets, d_lengths,
                int_buffer, num_hi_output, num_lo_output, num_scale_output,
                str_buffer, str_null_pos, d_str_offsets,
                d_col_types, d_col_lengths,
                rows_in_chunk, num_cols
            )
            
            # イベント記録終了
            end_event.record()
            end_event.synchronize()
            
            # カーネル実行時間（CUDA イベントベース）
            kernel_time_ms = cuda.event_elapsed_time(start_event, end_event)
            
            stream2.synchronize()
            
            # カーネル実行時間（Python計測）
            kernel_time = time.perf_counter() - kernel_start
            print(f"[TIMING] デコードカーネル実行時間: {kernel_time*1000:.2f} ms (CUDA計測: {kernel_time_ms:.2f} ms)")
            
            results = {}
            int_col_idx = 0
            str_col_idx = 0
            
            for i, col in enumerate(columns):
                col_type = col_types[i]
                if col_type <= 1:
                    if col_type == 0:
                        if int_buffer is not None:
                            host_array = np.empty(rows_in_chunk, dtype=np.int32)
                            int_buffer[int_col_idx * rows_in_chunk:(int_col_idx + 1) * rows_in_chunk].copy_to_host(host_array)
                            results[col.name] = host_array
                            int_col_idx += 1
                    else:
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
                else:
                    if str_buffer is not None and str_null_pos is not None:
                        length = col_lengths[i]
                        try:
                            if hasattr(d_str_offsets, "size"):
                                d_str_offsets_size = d_str_offsets.size
                            else:
                                d_str_offsets_size = str_col_idx + 1
                            host_offsets = np.empty(d_str_offsets_size, dtype=np.int32)
                            d_str_offsets.copy_to_host(host_offsets)
                        except Exception as e:
                            print(f"オフセット取得エラー: {e}")
                            num_str_cols = sum(1 for t in col_types if t > 1)
                            host_offsets = np.zeros(num_str_cols + 1, dtype=np.int32)
                            if num_str_cols > 0:
                                col_lens = [col_lengths[i] for i, t in enumerate(col_types) if t > 1]
                                total_len = sum(col_lens)
                                cum_len = 0
                                str_idx = 0
                                for i, t in enumerate(col_types):
                                    if t > 1:
                                        host_offsets[str_idx] = int(cum_len * rows_in_chunk)
                                        cum_len += (col_lengths[i] / total_len) if total_len > 0 else 0
                                        str_idx += 1
                        str_start = host_offsets[str_col_idx]
                        str_end = str_start + (rows_in_chunk * length)
                        
                        host_str_array = np.empty(rows_in_chunk * length, dtype=np.uint8)
                        str_buffer[str_start:str_end].copy_to_host(host_str_array)
                        
                        host_null_pos = np.empty(rows_in_chunk, dtype=np.int32)
                        str_null_pos[str_col_idx * rows_in_chunk:(str_col_idx + 1) * rows_in_chunk].copy_to_host(host_null_pos)
                        
                        strings = []
                        for j in range(rows_in_chunk):
                            null_pos = host_null_pos[j]
                            if null_pos == 0:
                                strings.append(None)
                            else:
                                start_idx = j * length
                                end_idx = start_idx + null_pos
                                byte_data = host_str_array[start_idx:end_idx].tobytes()
                                try:
                                    if byte_data:
                                        string_val = byte_data.decode("utf-8", errors="replace")
                                    else:
                                        string_val = ""
                                    strings.append(string_val)
                                except Exception as e:
                                    print(f"文字列デコードエラー: {e}, バイトデータ: {byte_data}")
                                    strings.append("")
                        results[col.name] = strings
                        str_col_idx += 1
            
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
            return {}
