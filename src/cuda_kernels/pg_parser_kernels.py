"""
PostgreSQLバイナリデータ用のCUDAパーサーカーネル
"""

import os
import numpy as np
from numba import cuda
import math
import time
from numba import int64, int32

# 環境変数から最適化パラメータを取得
MAX_THREADS = int(os.environ.get('GPUPASER_MAX_THREADS', '1024'))
MAX_BLOCKS = int(os.environ.get('GPUPASER_MAX_BLOCKS', '2048'))

GPUPGPARSER_DEBUG_KERNELS = os.environ.get('GPUPGPARSER_DEBUG_KERNELS', '0').lower() in ('1', 'true')
DEBUG_ARRAY_SIZE = 1024 if GPUPGPARSER_DEBUG_KERNELS else 0

@cuda.jit(device=True, inline=True)
def decode_int32_be(data, pos):
    """
    4バイト整数のビッグエンディアンからのデコード
    
    Args:
        data: バイナリデータ配列
        pos: 読み取り位置
        
    Returns:
        デコードされた32ビット整数値 (NULLは -1)
    """
    # バイトを取得して直接ビット演算（NumPy API使用せず）
    b0 = (data[pos] & 0xFF)
    b1 = (data[pos + 1] & 0xFF)
    b2 = (data[pos + 2] & 0xFF)
    b3 = (data[pos + 3] & 0xFF)
    
    # ビッグエンディアンからリトルエンディアンに変換し、符号付き int32 として返す
    val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    # 0xFFFFFFFF を -1 に変換
    if val == 0xFFFFFFFF:
        return -1
    # 他の負の値はそのまま（エラーの可能性）、正の値はそのまま
    elif val >= 0x80000000:
         # This case should ideally not happen for valid positive lengths
         # but handles potential negative values other than -1 if they occur.
         # Reinterpret as signed int32
         return val - 0x100000000 
    return val

@cuda.jit
def calculate_row_lengths_and_null_flags_gpu(raw_data, max_rows, num_fields,
                                            row_starts, row_lengths, null_flags):
    """
    GPU kernel to parse PostgreSQL COPY BINARY data and compute each row's byte length 
    and NULL flags matrix. Each thread processes one row.
    
    Args:
        raw_data: uint8 device array containing the binary COPY data (big-endian format).
        max_rows: int32, maximum number of rows to process (upper bound of rows in data).
        num_fields: int32, number of columns (fields) per row (fixed schema).
        row_starts: int32 device array of length >= max_rows, containing the byte offset 
                    where each row starts in raw_data.
        row_lengths: int32 device array (length >= max_rows) to store the output byte length of each row.
        null_flags: int8 device 2D array of shape (max_rows, num_fields) to store NULL flags 
                    (1 if the field is NULL, 0 if not) for each row and column.
    """
    rid = cuda.grid(1)  # global thread index (row index)
    if rid >= max_rows:
        return  # thread is out of range of actual rows
    
    # Get start position of this row in the raw data
    start_pos = row_starts[rid]
    data_len = raw_data.shape[0]
    # Validate start position and ensure we have at least 2 bytes for the field count
    if start_pos < 0 or start_pos + 2 > data_len:
        # Invalid offset or no room for field count – mark this row as empty/invalid
        row_lengths[rid] = -1
        for c in range(num_fields):
            null_flags[rid, c] = 1
        return
    
    # Read the number of fields (uint16 big-endian) at the start of the row
    num_fields_in_row = read_uint16_be(raw_data, start_pos)
    if num_fields_in_row == 0xFFFF:
        # End-of-data marker encountered at this row's start – no actual row data
        row_lengths[rid] = -1
        for c in range(num_fields):
            null_flags[rid, c] = 1
        return
    
    # Determine how many fields to process (should equal num_fields for valid rows)
    fields_to_process = num_fields_in_row
    if fields_to_process > num_fields:
        fields_to_process = num_fields  # limit to expected number of columns
    
    pos = start_pos + 2  # position now at the first field-length entry
    # Parse each field in this row
    for c in range(fields_to_process):
        # Ensure there's enough data to read the 4-byte field length
        if pos + 4 > data_len:
            # Not enough bytes to read the next field length – treat remaining fields as NULL
            row_lengths[rid] = -1 # Mark row as problematic
            null_flags[rid, c] = 1
            for rem_c in range(c + 1, num_fields): # Null out remaining expected fields
                null_flags[rid, rem_c] = 1
            return
        
        field_len = decode_int32_be(raw_data, pos)
        pos += 4  # advance past the length field itself
        
        if field_len < 0: # Handles NULL (-1)
            # NULL field: mark flag and no data bytes to skip
            null_flags[rid, c] = 1
        else:
            # Non-NULL field: mark flag and skip the field data bytes
            null_flags[rid, c] = 0
            if pos + field_len > data_len:
                # Not enough data for the declared field length – treat as error
                row_lengths[rid] = -1 # Mark row as problematic
                # Mark this and remaining fields as NULL (data truncated)
                null_flags[rid, c] = 1
                for rem_c in range(c + 1, num_fields): # Null out remaining expected fields
                    null_flags[rid, rem_c] = 1
                return
            pos += field_len  # skip over the actual field data bytes
    
    # Mark any remaining expected fields as NULL if the row had fewer fields than expected
    for c in range(fields_to_process, num_fields):
        null_flags[rid, c] = 1
    
    # If the data row contains more fields than expected (fields_to_process was limited),
    # skip over the extra fields to reach the true end of the row.
    if num_fields_in_row > num_fields:
        for _ in range(num_fields, num_fields_in_row):
            if pos + 4 > data_len: # Check before reading length
                break 
            extra_field_len = decode_int32_be(raw_data, pos)
            pos += 4 # Advance past the length
            if extra_field_len >= 0: # If not NULL
                if pos + extra_field_len > data_len:
                    break # data ended unexpectedly
                pos += extra_field_len
            # If extra_field_len is < 0 (NULL), pos is already advanced past length, no data to skip
    
    # Calculate the total byte length of this row
    row_lengths[rid] = pos - start_pos

@cuda.jit(device=True)
def find_next_row_start(data, start_pos, end_pos, num_cols, valid_cols=0):
    """
    次の有効な行の先頭位置を見つける
    
    Args:
        data: バイナリデータ配列
        start_pos: 探索開始位置
        end_pos: 探索終了位置
        num_cols: 期待されるカラム数
        valid_cols: 検出済みの有効列数（0の場合は未検出）
        
    Returns:
        有効な行の先頭位置、見つからない場合は-1
    """
    pos = start_pos
    array_size = data.shape[0]  # len()の代わりにshape[0]を使用
    while pos < end_pos - 1:
        # 残りのバイト数チェック
        if pos + 2 > array_size:
            return -1
            
        # タプルのフィールド数を確認
        num_fields = ((data[pos] & 0xFF) << 8) | (data[pos + 1] & 0xFF)
        
        # 終端マーカーをチェック
        if num_fields == 0xFFFF:
            return -1
            
        # 有効なフィールド数をチェック - 動的検出対応
        if valid_cols > 0:
            # 検出済み有効列数と比較
            if num_fields == valid_cols:
                return pos
        else:
            # より柔軟なチェック：
            # 1. num_colsと一致する場合
            # 2. 妥当な範囲内（0 < 列数 < 100）で17（lineorderテーブルの列数）と一致する場合
            # 3. その他の妥当な範囲内（0 < 列数 < 100）
            if num_fields == num_cols:
                return pos
            elif num_fields == 17:  # lineorderテーブルの列数
                return pos
            elif num_fields > 0 and num_fields < 100:
                return pos
            
        # 次のバイトへ
        pos += 1
        
    return -1

@cuda.jit(device=True)
def atomic_add_global(array, idx, val):
    """
    アトミックな加算操作（グローバルメモリ用）
    """
    # 古い実装（compare_and_swapを使用）だとNumba 0.56以降でエラーが発生するため、
    # 単純なatomic.addを使用する実装に変更
    return cuda.atomic.add(array, idx, val)

@cuda.jit(device=True, inline=True)
def read_uint16_be(data, pos):
    """ Reads a 16-bit unsigned integer in big-endian format. """
    b0 = data[pos]
    b1 = data[pos + 1]
    return (b0 << 8) | b1

@cuda.jit
def count_rows_gpu(raw, header_size, row_cnt, debug_array=None, debug_idx_atomic=None): # Add debug args
    """
    各スレッドが (header_size + tid) から stride=gridsize で走査し、
    行ヘッダ (uint16 != 0xFFFF) を検出したらその行を最後までスキップして
    ローカルカウンタをインクリメント。
    """
    pos   = header_size + cuda.grid(1)      # 走査開始
    step  = cuda.gridsize(1)                # 全スレッド合計の stride
    end   = raw.size - 2                    # uint16 読み込み可否境界
    local = 0
    tid = cuda.grid(1) # Get thread ID for debug

    # Debug loop counter for this thread
    debug_loops_done = 0
    max_debug_loops_per_thread = 3 # Record first few attempts per thread

    while pos < end:
        # Conditional debug recording at the start of the loop
        if GPUPGPARSER_DEBUG_KERNELS and tid < 2 and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
            # Status code 10: Start of while loop iteration in count_rows_gpu
            _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, step, end, 10)

        num = read_uint16_be(raw, pos)      # Use helper

        # Conditional debug recording after reading num_fields
        if GPUPGPARSER_DEBUG_KERNELS and tid < 2 and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
            # Status code 11: Read num_fields in count_rows_gpu
             _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, num, 0, 11)


        if num == 0xFFFF:                   # COPY‐BINARY 終端
            if GPUPGPARSER_DEBUG_KERNELS and tid < 2 and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                # Status code 12: EOF marker found in count_rows_gpu
                _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, num, 0, 12)
            break

        cur = pos + 2                       # フィールド長へ
        # Check if reading num_fields itself is valid before loop
        if cur > raw.size: # Need at least 2 bytes for num_fields
            if GPUPGPARSER_DEBUG_KERNELS and tid < 2 and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                 # Status code 13: Boundary error before field loop in count_rows_gpu
                _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, num, cur, 13)
            break

        valid_row_parse = True
        for _ in range(num):                # num == ncols
            if cur + 4 > raw.size:          # 異常終了 guard (check against raw.size)
                valid_row_parse = False
                break
            flen = decode_int32_be(raw, cur) # Use existing helper
            cur += 4                        # 長さ部スキップ
            if flen > 0:
                if cur + flen > raw.size:   # Check bounds before skipping data
                    valid_row_parse = False
                    break
                cur += flen                # データ部スキップ
            elif flen < -1: # Invalid length
                valid_row_parse = False
                if GPUPGPARSER_DEBUG_KERNELS and tid < 2 and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                    _record_debug_info_device(debug_array, debug_idx_atomic, tid, cur - 4, num, flen, 15) # Invalid flen
                break
            # If flen is -1 (NULL), cur is already advanced by 4

        if valid_row_parse:
            local += 1
            if GPUPGPARSER_DEBUG_KERNELS and tid < 2 and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, num, local, 16) # Valid row parsed
            # Move pos to the end of the successfully parsed row *before* adding step
            pos = cur
            # Advance pos by step
            pos += step
        else:
            if GPUPGPARSER_DEBUG_KERNELS and tid < 2 and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, num, 0, 17) # Invalid row parse
            # If parsing failed mid-row, advance from the current `pos` by `step`.
            # `pos` is currently at the beginning of the row that failed to parse.
            # So, the next scan position for this thread should be `pos + step`.
            pos += step
        
        debug_loops_done +=1
            
    if local > 0:
        cuda.atomic.add(row_cnt, 0, local) # Add local count to global counter



@cuda.jit(device=True, inline=True)
def max(a, b):
    # Numba CUDA device functions don't support standard max directly sometimes
    return a if a > b else b

@cuda.jit
def parse_fields_from_offsets_gpu(raw, ncols, rows,
                                  row_offsets_in, f_off_out, f_len_out):
    """
    GPU kernel to parse fields based on pre-calculated row start offsets (Plan C approach).

    Args:
        raw: GPU device array containing the raw binary data.
        ncols: Number of columns per row.
        rows: Total number of rows.
        row_offsets_in: GPU device array containing the start offset of each row.
        f_off_out: GPU device array (rows, ncols) for field start offsets (absolute).
        f_len_out: GPU device array (rows, ncols) for field lengths (-1 for NULL).
    """
    gtid = cuda.grid(1) # Global thread ID = row index

    if gtid >= rows:
        return

    row_start = row_offsets_in[gtid]
    pos = row_start
    data_len = raw.size

    # --- Validate row_start and read num_fields ---
    if pos < 0 or pos + 2 > data_len: # Invalid start offset or not enough data for num_fields
        # Mark all fields as NULL for this row
        for c in range(ncols):
            f_len_out[gtid, c] = -1
            f_off_out[gtid, c] = 0
        return # Skip processing this row

    nf = read_uint16_be(raw, pos)
    pos += 2

    if nf == 0xFFFF: # End of data marker found unexpectedly at row start
         # Mark all fields as NULL
         for c in range(ncols):
            f_len_out[gtid, c] = -1
            f_off_out[gtid, c] = 0
         return # Skip processing this row

    # --- Parse Fields ---
    fields_to_process = min(nf, ncols) # Process up to the expected number of columns

    for c in range(fields_to_process):
        # Check bounds before reading field length
        if pos + 4 > data_len:
            # Mark remaining fields as NULL if data ends unexpectedly
            for rem_c in range(c, ncols):
                f_len_out[gtid, rem_c] = -1
                f_off_out[gtid, rem_c] = 0
            return # Stop processing this row

        fl = decode_int32_be(raw, pos)
        field_data_start = pos + 4

        if fl < 0: # NULL value
            f_len_out[gtid, c] = -1
            f_off_out[gtid, c] = 0 # Use 0 for offset of NULL
            pos += 4 # Advance past the length field
        else: # Non-NULL value
            # Check bounds before accessing/skipping data
            if field_data_start + fl > data_len:
                # Mark remaining fields as NULL if data ends unexpectedly
                for rem_c in range(c, ncols):
                    f_len_out[gtid, rem_c] = -1
                    f_off_out[gtid, rem_c] = 0
                return # Stop processing this row

            f_len_out[gtid, c] = fl
            f_off_out[gtid, c] = field_data_start # Store absolute offset
            pos += 4 + fl # Advance past length and data

    # --- Fill remaining expected columns with NULL ---
    # This handles cases where nf < ncols
    for c in range(fields_to_process, ncols):
         f_len_out[gtid, c] = -1
         f_off_out[gtid, c] = 0

@cuda.jit
def parse_rows_and_fields_gpu(raw_data, ncols, rows, header_size,
                              row_offsets_out, field_offsets_out, field_lengths_out,
                              global_prefix_sum_counter):
    """
    GPU kernel to parse PostgreSQL COPY BINARY data, calculating row offsets
    and field offsets/lengths in a single pass (Plan D).

    Args:
        raw_data: GPU device array containing the raw binary data.
        ncols: Number of columns per row.
        rows: Total number of rows (pre-calculated by count_rows_gpu).
        header_size: Size of the file header in bytes.
        row_offsets_out: GPU device array to store the start offset of each row.
        field_offsets_out: GPU device array (rows, ncols) for field start offsets (relative to row start).
        field_lengths_out: GPU device array (rows, ncols) for field lengths (-1 for NULL).
        global_prefix_sum_counter: GPU device array (single element) for global atomic counter,
                                   initialized to header_size before kernel launch.
    """
    # Thread identification
    tid = cuda.grid(1) # Global thread ID
    bid = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    block_dim = cuda.blockDim.x

    # Shared memory for block-local prefix sum of row lengths
    # Size needs to be block_dim
    sh_row_lengths = cuda.shared.array(shape=MAX_THREADS, dtype=int64) # Use constant or dynamically allocated size

    # --- Calculate row length for the current thread's assigned row ---
    my_row_length = int64(0)
    current_pos = int64(0) # Placeholder, actual start pos determined later

    if tid < rows:
        row_start = row_offsets_out[tid]
        pos = row_start
        data_len = raw_data.shape[0]

        if pos < 0 or pos + 2 > data_len: # Invalid start offset
            for c in range(ncols):
                field_lengths_out[tid, c] = -1
                field_offsets_out[tid, c] = 0
            return

        num_fields_read = read_uint16_be(raw_data, pos)
        pos += 2

        if num_fields_read == 0xFFFF: # Should not happen if row_offsets is correct
                for c in range(ncols):
                    field_lengths_out[tid, c] = -1
                    field_offsets_out[tid, c] = 0
                return

        fields_to_process = min(num_fields_read, ncols)

        for c in range(fields_to_process):
            if pos + 4 > data_len:
                # Mark remaining as NULL
                for rem_c in range(c, ncols):
                    field_lengths_out[tid, rem_c] = -1
                    field_offsets_out[tid, rem_c] = 0
                return # Exit row processing

            field_len = decode_int32_be(raw_data, pos)
            field_data_start = pos + 4

            if field_len < 0: # NULL
                field_offsets_out[tid, c] = 0 # Or relative offset 0 within row? Let's use absolute.
                field_lengths_out[tid, c] = -1
                pos += 4
            else: # Non-NULL
                if field_data_start + field_len > data_len:
                    # Mark remaining as NULL
                    for rem_c in range(c, ncols):
                        field_lengths_out[tid, rem_c] = -1
                        field_offsets_out[tid, rem_c] = 0
                    return # Exit row processing

                field_offsets_out[tid, c] = field_data_start
                field_lengths_out[tid, c] = field_len
                pos += 4 + field_len

        # Fill remaining expected columns with NULL if num_fields_read < ncols
        for c in range(fields_to_process, ncols):
                field_lengths_out[tid, c] = -1
                field_offsets_out[tid, c] = 0


@cuda.jit
def parse_binary_format_kernel_one_row(chunk_array, field_offsets, field_lengths, 
                                        num_cols, header_size, row_start_positions=None):
    """
    PostgreSQLバイナリデータを直接GPU上で解析するカーネル（1スレッド1行）
    [Simplified and Corrected Version]

    Args:
        chunk_array: バイナリデータ配列
        field_offsets: フィールド位置の出力配列
        field_lengths: フィールド長の出力配列
        num_cols: 1行あたりのカラム数
        header_size: ヘッダーサイズ（バイト）
        row_start_positions: 各行の開始位置配列（オプション、提供されない場合は計算）
    """
    # スレッド情報 - 各スレッドが1行だけを担当
    thread_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    # 有効な行数範囲チェック（field_offsetsの行数に基づく）
    max_rows = field_offsets.shape[0]
    if thread_id >= max_rows:
        return  # 範囲外のスレッドは早期リターン
    
    # 行の開始位置を決定
    row_start = 0
    if row_start_positions is not None:
        if thread_id < row_start_positions.shape[0]:
            row_start = row_start_positions[thread_id]
            # 開始位置が無効な場合（例: CPU計算でエラー時に-1など）はスキップ
            if row_start < 0:
                # Ensure remaining fields for this row are marked NULL if skipped early
                for field_idx in range(num_cols):
                    field_lengths[thread_id, field_idx] = -1
                    field_offsets[thread_id, field_idx] = 0
                return
        else:
            # row_start_positions 配列の範囲外
            return
    else:
        # row_start_positions が提供されない場合はエラーとして扱うか、
        # ここで非効率な計算を行う。今回は提供される前提でリターン。
        # 安全のため、この行のフィールドをNULLとしてマーク
        for field_idx in range(num_cols):
            field_lengths[thread_id, field_idx] = -1
            field_offsets[thread_id, field_idx] = 0
        return

    # この時点で row_start は現在のスレッドが処理すべき行の開始位置
    pos = row_start
    array_size = chunk_array.shape[0]

    # --- フィールド数読み込みと検証 ---
    if pos + 2 > array_size:
        # フィールド数を読み込む前にデータ終端
        for field_idx in range(num_cols):
            field_lengths[thread_id, field_idx] = -1
            field_offsets[thread_id, field_idx] = 0
        return

    num_fields_read = ((chunk_array[pos] & 0xFF) << 8) | (chunk_array[pos + 1] & 0xFF)

    # 終端マーカー (0xFFFF) チェック
    if num_fields_read == 0xFFFF:
        # この行以降はデータがないとみなし、この行をNULLで埋める
        for field_idx in range(num_cols):
            field_lengths[thread_id, field_idx] = -1
            field_offsets[thread_id, field_idx] = 0
        return

    pos += 2  # フィールド数フィールドをスキップ

    # 実際に処理するフィールド数を決定 (読み取った数と期待値の小さい方)
    fields_to_process = min(num_fields_read, num_cols)

    # --- 各フィールドの処理ループ ---
    for field_idx in range(fields_to_process):
        # フィールド長読み込み前の境界チェック
        if pos + 4 > array_size:
            # 行の途中でデータ終端。残りのフィールドをNULLにする
            for remaining_idx in range(field_idx, num_cols):
                field_lengths[thread_id, remaining_idx] = -1
                field_offsets[thread_id, remaining_idx] = 0
            return

        # フィールド長を取得
        field_len = decode_int32_be(chunk_array, pos)
        pos += 4 # フィールド長フィールドをスキップ

        if field_len < 0:  # NULL値 (-1)
            field_offsets[thread_id, field_idx] = 0
            field_lengths[thread_id, field_idx] = -1
        else: # Non-NULL value
            data_start_pos = pos
            # データ読み込み前の境界チェック
            if pos + field_len > array_size:
                # データが途中で切れている。このフィールドと残りをNULLにする
                for remaining_idx in range(field_idx, num_cols):
                    field_lengths[thread_id, remaining_idx] = -1
                    field_offsets[thread_id, remaining_idx] = 0
                return

            # 有効なデータ。オフセットと長さを記録
            field_offsets[thread_id, field_idx] = data_start_pos
            field_lengths[thread_id, field_idx] = field_len
            pos += field_len  # データフィールドをスキップ

    # --- ループ後の処理 ---
    # 読み取ったフィールド数が期待値より少ない場合、残りの列をNULLで埋める
    if fields_to_process < num_cols:
        for field_idx in range(fields_to_process, num_cols):
            field_lengths[thread_id, field_idx] = -1
            field_offsets[thread_id, field_idx] = 0

    # next_row_start の計算と最後のpos調整は削除 (ループが正しくposを進めるため)

@cuda.jit
def parse_binary_format_kernel(chunk_array, field_offsets, field_lengths, num_cols, header_shared):
    """
    PostgreSQLバイナリデータを直接GPU上で解析するカーネル（高性能最適化版）
    
    Args:
        chunk_array: バイナリデータ配列
        field_offsets: フィールド位置の出力配列
        field_lengths: フィールド長の出力配列
        num_cols: 1行あたりのカラム数（推奨値、実際のデータと異なる場合は自動検出）
        header_shared: 共有メモリ（ヘッダー情報と行カウンタ用）
            header_shared[0]: ヘッダーサイズ
            header_shared[1]: 検出された有効行数
            header_shared[2]: 検出された有効列数（自動検出用）
    """
    # スレッド情報 - スレッド識別をより効率的に
    thread_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    block_id = cuda.blockIdx.x
    tx = cuda.threadIdx.x  # ブロック内スレッドID
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    total_threads = block_size * grid_size
    
    # 共有メモリを宣言（各ブロックごとにデータの一部をキャッシュ）
    # スレッドブロック内での協調動作用
    cache_size = 1024  # キャッシュするデータサイズを2倍に増加
    shared_cache = cuda.shared.array(shape=(1024,), dtype=np.uint8)
    shared_row_count = cuda.shared.array(shape=(1,), dtype=np.int32)
    
    # スレッドをワープ単位で扱うための変数
    warp_size = 32
    warp_id = tx // warp_size
    lane_id = tx % warp_size
    
    # ブロック内の最初のスレッドが初期化
    if tx == 0:
        shared_row_count[0] = 0
    
    # データサイズ
    array_size = chunk_array.shape[0]
    
    # ヘッダー処理（スレッド0のみ）- 最適化バージョン
    if thread_id == 0:
        # 初期化
        header_shared[0] = 0  # ヘッダーサイズ
        header_shared[1] = 0  # 有効行数カウンタ
        header_shared[2] = 0  # 有効列数（未検出=0）
        
        # PostgreSQLバイナリフォーマットのヘッダーチェック
        if array_size >= 11:
            # 'P', 'G'を一度に検出（条件分岐削減）
            pg_header_match = (chunk_array[0] == 80 and chunk_array[1] == 71)
            
            if pg_header_match:
                header_size = 11
                
                # フラグとヘッダー拡張の効率的な処理
                has_extension = (array_size >= header_size + 8)
                if has_extension:
                    header_size += 8
                    
                    # 拡張データ長
                    ext_len = decode_int32_be(chunk_array, header_size - 4)
                    has_valid_extension = (ext_len > 0 and array_size >= header_size + ext_len)
                    if has_valid_extension:
                        header_size += ext_len
                
                header_shared[0] = header_size
                
                # 先頭から有効そうな列数を検出
                if array_size >= header_size + 2:
                    scan_pos = header_size
                    while scan_pos < min(array_size - 2, header_size + 1000):
                        potential_field_count = ((chunk_array[scan_pos] & 0xFF) << 8) | (chunk_array[scan_pos + 1] & 0xFF)
                        # 妥当な範囲の列数なら記録（0 < 列数 < 100）
                        if potential_field_count > 0 and potential_field_count < 100:
                            header_shared[2] = potential_field_count  # 有効列数を保存
                            break
                        scan_pos += 1
    
    # ヘッダー情報共有
    cuda.syncthreads()
    header_size = header_shared[0]
    
    # 処理可能な最大行数
    max_rows = field_offsets.shape[0] // num_cols
    
    # データ分割（各スレッドの担当範囲計算）
    # データサイズを元に適切なチャンク配分
    data_size = array_size - header_size
    min_chunk_size = 512  # 最小探索サイズ
    chunk_size = max(min_chunk_size, data_size // total_threads)
    
    # スレッドごとの処理範囲
    start_offset = header_size + thread_id * chunk_size
    end_offset = min(start_offset + chunk_size * 2, array_size)  # オーバーラップ戦略
    
    if start_offset >= array_size:
        return
    
    # 共有メモリから有効列数を取得
    valid_num_cols = header_shared[2]
    
    # 最初のスレッド以外は有効な行の先頭を探す
    if thread_id > 0:
        valid_start = find_next_row_start(chunk_array, start_offset, end_offset, num_cols, valid_num_cols)
        if valid_start < 0:
            return
        start_offset = valid_start
    
    # 各スレッドは担当範囲の行を処理
    pos = start_offset
    local_row_count = 0
    
    # 共有メモリを活用したキャッシング処理
    # 現在のブロック内のスレッド数だけループを回す
    for cache_round in range(0, 16):  # 最大16回までキャッシュを更新
        # 各スレッドはブロックのデータをキャッシュに読み込む
        cache_pos = start_offset + tx * 2  # 読み込み位置
        cache_idx = tx * 2  # キャッシュインデックス
        
        # 読み込み位置が有効範囲内ならキャッシュに読み込む
        if cache_pos < min(end_offset, start_offset + cache_size) and cache_idx < cache_size - 1:
            shared_cache[cache_idx] = chunk_array[cache_pos] if cache_pos < array_size else 0
            shared_cache[cache_idx + 1] = chunk_array[cache_pos + 1] if cache_pos + 1 < array_size else 0
        
        # ブロック内の全スレッドがキャッシュに書き込むまで待機
        cuda.syncthreads()
        
        # キャッシュからデータを読み取る処理（最大キャッシュサイズまで）
        cache_end = min(cache_size, end_offset - start_offset)
        
        # メインループ：キャッシュされたデータを処理
        while pos < start_offset + cache_end and local_row_count < 1000 and pos < end_offset:
            # キャッシュ内のオフセット計算
            cache_offset = pos - start_offset
            
            # キャッシュ範囲外ならブレーク
            if cache_offset >= cache_size - 1:
                break
                
            # フィールド数取得（キャッシュから）
            num_fields = ((shared_cache[cache_offset] & 0xFF) << 8) | (shared_cache[cache_offset + 1] & 0xFF)
            
            # 終端マーカーのチェック
            if num_fields == 0xFFFF:
                pos = end_offset  # 終了させる
                break
                
            # フィールド数検証（動的列数検出対応）
            valid_num_cols = header_shared[2]  # 共有メモリから有効列数を取得
            
            if valid_num_cols > 0:
                # 既に有効列数が検出されている場合、それと比較
                if num_fields != valid_num_cols:
                    pos += 1
                    continue
            else:
                # まだ有効列数が未検出の場合
                if num_fields > 0 and num_fields < 100:
                    # 有効そうな値なら記録し、現在の行を処理
                    # アトミック更新（compare_and_swapでなく単純な条件チェックに変更）
                    if header_shared[2] == 0:  # まだ設定されていなければ
                        cuda.atomic.add(header_shared, 2, num_fields)  # 値を加算（0から始まるため加算で設定と同じ）
                else:
                    # 無効そうな値はスキップ
                    pos += 1
                    continue
                
            # 行の先頭位置保存
            row_start = pos
            pos += 2  # フィールド数フィールドをスキップ
            
            # ここからはグローバルメモリから読み取る必要あり
            
            # まずブロック内でのローカル行カウントを更新
            local_idx = cuda.atomic.add(shared_row_count, 0, 1)
            
            # ブロック処理後にグローバル行インデックスを取得（パフォーマンス向上）
            # 各ブロックはバッチでグローバルカウンタに加算する
            if tx == 0 and local_idx == 0:
                # ブロックの最初のスレッドのみグローバル加算
                atomic_add_global(header_shared, 1, block_size)  # 最大処理可能行数を予約
            
            # 全スレッドが行カウント更新を待つ
            cuda.syncthreads()
            
            # グローバル行インデックスを計算
            row_idx = header_shared[1] - block_size + local_idx
            
            # 最大行数チェック
            if row_idx >= max_rows:
                break
                
            # 各フィールドを処理
            valid_row = True
            for field_idx in range(num_cols):
                # バイト数チェック
                if pos + 4 > array_size:
                    valid_row = False
                    break
                    
                # フィールド長を取得（グローバルメモリから直接）
                field_len = decode_int32_be(chunk_array, pos)
                
                # 出力インデックス計算
                # out_idx = row_idx * num_cols + field_idx # Use 2D index below
                
                # 結果を保存 (Use 2D indexing)
                if row_idx < max_rows: # Check row index bound first
                    if field_len < 0: # NULL
                         field_offsets[row_idx, field_idx] = 0
                         field_lengths[row_idx, field_idx] = -1
                    else: # Non-NULL
                         field_offsets[row_idx, field_idx] = pos + 4 # Offset is after length field
                         field_lengths[row_idx, field_idx] = field_len

                # ポインタを進める (length field)
                pos += 4
                
                # フィールドデータをスキップ (if not NULL)
                if field_len >= 0: 
                    if pos + field_len > array_size: # Bounds check
                        valid_row = False
                        break
                    pos += field_len
                    
            if not valid_row:
                # 行が不完全な場合、先頭から1バイト進めてやり直し
                pos = row_start + 1
                # 行カウンタを戻す
                # ローカルとグローバルの両方を調整
                cuda.atomic.add(shared_row_count, 0, -1)
                if tx == 0:  # ブロックごとに一度だけグローバルカウンタを調整
                    cuda.atomic.add(header_shared, 1, -1)
                continue
                
            # 有効な行を処理できた
            local_row_count += 1
            
        # 次のキャッシュ領域に移動
        start_offset += cache_size
        
        # 全スレッドの同期
        cuda.syncthreads()
        
        # すでに担当範囲の終端に達していたら終了
        if pos >= end_offset:
            break

@cuda.jit
def find_row_start_offsets_gpu(raw, header_size, row_starts_out, row_count_out, debug_array=None, debug_idx_atomic=None): # Add debug args
    """
    PostgreSQL COPY BINARYデータから各行の開始オフセットを検出し、配列に記録するGPUカーネル。
    
    複数スレッドが (header_size + thread_id) からグリッド全体のストライドでデータを走査し、 
    行の先頭（2バイトのフィールド数 != 0xFFFF）を見つけたらそのバイトオフセットを記録する。
    各行のフィールド長も確認し、データ範囲を超える不正な行があればその開始位置は記録しない。
    終端マーカー (0xFFFF) に達するかデータ末尾に到達した時点で処理を終了する。
    検出された行数は row_count_out[0] に格納される。
    
    Args:
        raw: バイナリデータ配列（uint8 型のGPUデバイス配列）
        header_size: ヘッダー部分のバイトサイズ（行データの開始オフセット）
        row_starts_out: 各行の開始オフセット（int32）を書き込む出力配列
        row_count_out: 検出した行数を格納するための配列（int32型、長さ1を想定）
    """
    # Simplified sequential scan by thread 0 for correctness check
    tid = cuda.grid(1)
    
    if tid == 0: # Only thread 0 performs the scan
        pos = header_size
        data_len = raw.size
        current_row_count = 0
        max_rows_to_store = row_starts_out.shape[0]

        debug_loops_done = 0
        max_debug_loops_per_thread = 20 # Allow more loops for sequential scan debug

        while pos < data_len - 1:
            if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, data_len, current_row_count, 30) # Start of loop iteration

            if pos + 1 >= data_len:
                if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                    _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, 0, 0, 34) # Boundary error before num_fields
                break
            
            num_fields = read_uint16_be(raw, pos)

            if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, num_fields, 0, 31) # Read num_fields

            if num_fields == 0xFFFF:
                if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                    _record_debug_info_device(debug_array, debug_idx_atomic, tid, pos, num_fields, 0, 32) # EOF
                break
            
            row_start_candidate = pos
            cur = pos + 2
            valid_row = True
            for j in range(num_fields):
                if cur + 4 > data_len:
                    valid_row = False
                    if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                        _record_debug_info_device(debug_array, debug_idx_atomic, tid, cur, num_fields, j, 35) # Boundary error field_len
                    break
                field_len = decode_int32_be(raw, cur)
                cur += 4
                if field_len >= 0:
                    if cur + field_len > data_len:
                        valid_row = False
                        if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                            _record_debug_info_device(debug_array, debug_idx_atomic, tid, cur - 4, num_fields, field_len, 36) # Boundary error field_data
                        break
                    cur += field_len
                elif field_len < -1:
                    valid_row = False
                    if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                        _record_debug_info_device(debug_array, debug_idx_atomic, tid, cur - 4, num_fields, field_len, 37) # Invalid field_len
                    break
            
            if valid_row:
                if current_row_count < max_rows_to_store:
                    row_starts_out[current_row_count] = row_start_candidate
                current_row_count += 1
                if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                     _record_debug_info_device(debug_array, debug_idx_atomic, tid, row_start_candidate, num_fields, current_row_count, 38) # Valid row
                pos = cur # Move to the start of the next potential row
            else:
                # If row is invalid, try to advance by 1 byte to find next potential row start.
                # This is a simple recovery, might not be robust for all malformed data.
                if GPUPGPARSER_DEBUG_KERNELS and debug_loops_done < max_debug_loops_per_thread and debug_array is not None and debug_idx_atomic is not None:
                    _record_debug_info_device(debug_array, debug_idx_atomic, tid, row_start_candidate, num_fields, 0, 39) # Invalid row
                pos = row_start_candidate + 1 
            
            debug_loops_done += 1
            if debug_loops_done >= max_debug_loops_per_thread and GPUPGPARSER_DEBUG_KERNELS: # Ensure we break if only debugging limited loops
                break


        # After loop, thread 0 updates the global count
        cuda.atomic.add(row_count_out, 0, current_row_count)
