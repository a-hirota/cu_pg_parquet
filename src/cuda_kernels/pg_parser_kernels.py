"""
PostgreSQLバイナリデータ用のCUDAパーサーカーネル
"""

import os
import numpy as np
from numba import cuda
import math
import time

# 環境変数から最適化パラメータを取得
MAX_THREADS = int(os.environ.get('GPUPASER_MAX_THREADS', '1024'))
MAX_BLOCKS = int(os.environ.get('GPUPASER_MAX_BLOCKS', '2048'))

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
