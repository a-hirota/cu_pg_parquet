"""
CUDAカーネル定義とGPUデコード処理モジュール
"""

import numpy as np
from numba import cuda, njit
from typing import List, Dict, Any, Optional, Tuple

@cuda.jit(device=True)
def check_bounds(data, pos, size):
    """境界チェック"""
    return pos >= 0 and pos + size <= len(data)

@cuda.jit(device=True)
def decode_int16(data, pos):
    """2バイト整数のデコード（ビッグエンディアン）"""
    if not check_bounds(data, pos, 2):
        return 0
    
    # バイトを取得
    b0 = data[pos]
    b1 = data[pos + 1]
    
    # ビッグエンディアンからリトルエンディアンに変換
    val = ((b0 & 0xFF) << 8) | (b1 & 0xFF)
    
    # 符号付き16ビット整数に変換
    if val & 0x8000:  # 最上位ビットが1なら負の数
        val = -(((~val) + 1) & 0xFFFF)
    
    return val

@cuda.jit(device=True)
def decode_int32(data, pos):
    """4バイト整数のデコード（ビッグエンディアン）"""
    if not check_bounds(data, pos, 4):
        return 0
    
    # バイトを取得
    b0 = data[pos]
    b1 = data[pos + 1]
    b2 = data[pos + 2]
    b3 = data[pos + 3]
    
    # ビッグエンディアンからリトルエンディアンに変換
    val = ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)
    
    # 符号付き32ビット整数に変換
    if val & 0x80000000:  # 最上位ビットが1なら負の数
        val = -(((~val) + 1) & 0xFFFFFFFF)
    
    return val

@cuda.jit(device=True)
def decode_numeric_postgres(data, pos, hi_out, lo_out, scale_out, row_idx):
    """PostgreSQLのnumeric型を128ビット固定小数点数に変換"""
    if not check_bounds(data, pos, 8):  # 少なくともヘッダー部分があるか
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # ヘッダー情報の取得
    ndigits = decode_int16(data, pos)
    weight = decode_int16(data, pos + 2)
    sign = decode_int16(data, pos + 4)
    dscale = decode_int16(data, pos + 6)
    
    # データの妥当性チェック
    if ndigits < 0 or ndigits > 100 or dscale < 0 or dscale > 100:
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # 必要なバイト数をチェック
    if not check_bounds(data, pos + 8, ndigits * 2):
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # 128ビット整数に変換
    hi = 0
    lo = 0
    
    # 各桁を処理
    digit_pos = pos + 8
    scale = 0
    
    for i in range(ndigits):
        digit = decode_int16(data, digit_pos + i * 2)
        if digit < 0 or digit > 9999:  # 不正な桁
            continue
            
        # 既存の値を10000倍して新しい桁を加算
        hi_old = hi
        lo_old = lo
        
        # 10000倍
        lo = lo_old * 10000
        hi = hi_old * 10000 + (lo_old >> 32) * 10000
        
        # 桁を加算
        lo += digit
        if lo < lo_old:  # 桁上がり
            hi += 1
        
        # スケールの更新
        scale = max(scale, dscale)
    
    # 符号の適用
    if sign == 0x4000:  # 負の数
        if lo == 0:
            hi = -hi
        else:
            lo = ~lo + 1
            hi = ~hi
            if lo == 0:
                hi += 1
    
    # 結果の格納
    scale_out[0] = scale
    hi_out[row_idx] = hi
    lo_out[row_idx] = lo

@cuda.jit(device=True)
def bulk_copy_64bytes(src, src_pos, dst, dst_pos, size):
    """64バイト単位でのバルクコピー"""
    if size > 64:
        size = 64
    
    # 8バイトずつコピー
    for i in range(0, size, 8):
        if i + 8 <= size:
            # 8バイトを一度に読み書き
            val = 0
            for j in range(8):
                val = (val << 8) | src[src_pos + i + j]
            
            # 8バイトを一度に書き込み
            for j in range(8):
                dst[dst_pos + i + j] = (val >> ((7-j) * 8)) & 0xFF
        else:
            # 残りのバイトを1バイトずつコピー
            for j in range(size - i):
                dst[dst_pos + i + j] = src[src_pos + i + j]

@cuda.jit
def decode_all_columns_kernel(raw_data, field_offsets, field_lengths,
                            int_outputs, num_hi_output, num_lo_output, num_scale_output,
                            str_outputs, str_null_pos, str_offsets,
                            col_types, col_lengths, chunk_size, num_cols):
    """全カラムを一度に処理する統合カーネル（累積オフセット方式）"""
    # スレッドインデックスの計算を改善
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    
    # グリッド内の絶対位置を計算
    row = block_id * block_size + thread_id
    
    # ストライド処理を追加
    stride = block_size * grid_size
    while row < chunk_size:
        for col in range(num_cols):
            # フィールドオフセットとデータ長を取得
            field_idx = row * num_cols + col
            if field_idx >= len(field_offsets):
                return
                
            pos = field_offsets[field_idx]
            length = field_lengths[field_idx]
            
            if col_types[col] <= 1:  # 数値型
                if length == -1:  # NULL値
                    int_outputs[col * chunk_size + row] = 0
                else:
                    if col_types[col] == 0:  # integer
                        val = decode_int32(raw_data, pos)
                        int_outputs[col * chunk_size + row] = val
                    else:  # numeric/decimal
                        # PostgreSQLのnumeric型を128ビット固定小数点数として処理
                        decode_numeric_postgres(raw_data, pos,
                                             num_hi_output[col * chunk_size:],
                                             num_lo_output[col * chunk_size:],
                                             num_scale_output[col:col + 1],
                                             row)
            else:  # 文字列型
                max_length = col_lengths[col]
                
                # 文字列カラムのインデックスを計算
                str_col_idx = 0
                for i in range(col):
                    if col_types[i] == 2:  # 文字列型
                        str_col_idx += 1
                
                # 文字列バッファの位置を計算（累積オフセット方式）
                buffer_offset = str_offsets[str_col_idx]
                dst_pos = buffer_offset + row * max_length
                
                if length == -1:  # NULL値
                    str_null_pos[str_col_idx * chunk_size + row] = 0
                    continue
                    
                # 文字列データのコピー
                valid_length = min(length, max_length)
                
                # バルクコピーを使用
                for i in range(0, valid_length, 64):
                    copy_size = min(64, valid_length - i)
                    bulk_copy_64bytes(raw_data, pos + i, str_outputs, dst_pos + i, copy_size)
                
                # 残りのバイトをゼロクリア
                for i in range(valid_length, max_length):
                    str_outputs[dst_pos + i] = 0
                
                # 文字列の有効範囲を設定（カラムごとに独立した位置）
                str_null_pos[str_col_idx * chunk_size + row] = valid_length
        
        # 次の行へ
        row += stride

class GPUDecoder:
    """GPU上でのデコード処理を管理するクラス"""
    
    def __init__(self):
        """初期化"""
        pass
    
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
        num_cols = len(columns)
        
        # デバイスに転送（通常はmemory_managerで行うが、ここではシンプルに直接処理）
        d_chunk = cuda.to_device(chunk_array)
        d_offsets = cuda.to_device(field_offsets)
        d_lengths = cuda.to_device(field_lengths)
        cuda.synchronize()  # メモリ転送の完了を待つ
        
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
                    host_offsets = np.empty(len(d_str_offsets), dtype=np.int32)
                    d_str_offsets.copy_to_host(host_offsets)
                    
                    str_start = host_offsets[str_col_idx]
                    str_end = str_start + (rows_in_chunk * length)
                    
                    host_array = np.empty(rows_in_chunk * length, dtype=np.uint8)
                    str_buffer[str_start:str_end].copy_to_host(host_array)
                    
                    # 文字列長の取得
                    null_positions = np.empty(rows_in_chunk, dtype=np.int32)
                    str_null_pos[str_col_idx * rows_in_chunk:(str_col_idx + 1) * rows_in_chunk].copy_to_host(null_positions)
                    
                    data = host_array.reshape(rows_in_chunk, length)
                    strings = []
                    
                    for row in range(rows_in_chunk):
                        str_length = null_positions[row]
                        if str_length == 0:  # NULL値
                            strings.append('')
                        else:
                            # 文字列データを直接デコード
                            row_data = data[row, :str_length]
                            try:
                                # バイト列を文字列にデコード（空白は保持）
                                s = bytes(row_data).decode('utf-8', errors='replace').rstrip()
                                strings.append(s)
                            except Exception as e:
                                strings.append('')  # デコードエラー
                    
                    # 結果を格納
                    results[col.name] = strings
                    str_col_idx += 1
        
        # 一時的なGPUメモリの解放
        cuda.synchronize()  # 処理完了を待つ
        del d_chunk
        del d_offsets
        del d_lengths
        del d_col_types
        del d_col_lengths
        cuda.synchronize()  # メモリ解放完了を待つ
        
        return results
