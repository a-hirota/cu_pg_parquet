"""
PostgreSQLバイナリデータ用のCUDAパーサーカーネル
"""

import numpy as np
from numba import cuda
import math

@cuda.jit(device=True)
def decode_int32_be(data, pos):
    """
    4バイト整数のビッグエンディアンからのデコード
    
    Args:
        data: バイナリデータ配列
        pos: 読み取り位置
        
    Returns:
        デコードされた32ビット整数値
    """
    # バイトを取得
    b0 = np.int32(data[pos]) & 0xFF
    b1 = np.int32(data[pos + 1]) & 0xFF
    b2 = np.int32(data[pos + 2]) & 0xFF
    b3 = np.int32(data[pos + 3]) & 0xFF
    
    # ビッグエンディアンからリトルエンディアンに変換
    val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    
    return val

@cuda.jit(device=True)
def find_next_row_start(data, start_pos, end_pos, num_cols):
    """
    次の有効な行の先頭位置を見つける
    
    Args:
        data: バイナリデータ配列
        start_pos: 探索開始位置
        end_pos: 探索終了位置
        num_cols: 期待されるカラム数
        
    Returns:
        有効な行の先頭位置、見つからない場合は-1
    """
    pos = start_pos
    while pos < end_pos - 1:
        # 残りのバイト数チェック
        if pos + 2 > len(data):
            return -1
            
        # タプルのフィールド数を確認
        num_fields = (np.int32(data[pos]) << 8) | np.int32(data[pos + 1])
        
        # 終端マーカーをチェック
        if num_fields == 0xFFFF:
            return -1
            
        # 有効なフィールド数をチェック
        if num_fields == num_cols:
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
def parse_binary_format_kernel(chunk_array, field_offsets, field_lengths, num_cols, header_shared):
    """
    PostgreSQLバイナリデータを直接GPU上で解析するカーネル（最適化版）
    
    Args:
        chunk_array: バイナリデータ配列
        field_offsets: フィールド位置の出力配列
        field_lengths: フィールド長の出力配列
        num_cols: 1行あたりのカラム数
        header_shared: 共有メモリ（ヘッダー情報と行カウンタ用）
            header_shared[0]: ヘッダーサイズ
            header_shared[1]: 検出された有効行数
    """
    # スレッド情報
    thread_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    total_threads = block_size * grid_size
    
    # データサイズ
    array_size = len(chunk_array)
    
    # ヘッダー処理（スレッド0のみ）
    if thread_id == 0:
        # 初期化
        header_shared[0] = 0  # ヘッダーサイズ
        header_shared[1] = 0  # 有効行数カウンタ
        
        # PostgreSQLバイナリフォーマットのヘッダーチェック
        if array_size >= 11:
            if chunk_array[0] == 80 and chunk_array[1] == 71:  # 'P', 'G'
                header_size = 11
                
                # フラグとヘッダー拡張
                if array_size >= header_size + 8:
                    header_size += 8
                    
                    # 拡張データ長
                    ext_len = decode_int32_be(chunk_array, header_size - 4)
                    if ext_len > 0 and array_size >= header_size + ext_len:
                        header_size += ext_len
                
                header_shared[0] = header_size
    
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
    
    # 最初のスレッド以外は有効な行の先頭を探す
    if thread_id > 0:
        valid_start = find_next_row_start(chunk_array, start_offset, end_offset, num_cols)
        if valid_start < 0:
            return
        start_offset = valid_start
    
    # 各スレッドは担当範囲の行を処理
    pos = start_offset
    local_row_count = 0
    
    while pos < end_offset and local_row_count < 1000:  # スレッド当たり最大1000行
        # 残りバイト数チェック
        if pos + 2 > array_size:
            break
            
        # フィールド数取得
        num_fields = (np.int32(chunk_array[pos]) << 8) | np.int32(chunk_array[pos + 1])
        
        # 終端マーカーのチェック
        if num_fields == 0xFFFF:
            break
            
        # フィールド数検証
        if num_fields != num_cols:
            pos += 1
            continue
            
        # 行の先頭位置保存
        row_start = pos
        pos += 2  # フィールド数フィールドをスキップ
        
        # グローバル行インデックスを取得（アトミック加算）
        row_idx = atomic_add_global(header_shared, 1, 1)
        
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
                
            # フィールド長を取得
            field_len = decode_int32_be(chunk_array, pos)
            
            # 出力インデックス計算
            out_idx = row_idx * num_cols + field_idx
            
            # 結果を保存
            if out_idx < field_offsets.shape[0]:
                field_offsets[out_idx] = pos + 4 if field_len != -1 else 0
                field_lengths[out_idx] = field_len
                
            # ポインタを進める
            pos += 4
            
            # フィールドデータをスキップ
            if field_len != -1 and field_len >= 0:
                if pos + field_len > array_size:
                    valid_row = False
                    break
                pos += field_len
                
        if not valid_row:
            # 行が不完全な場合、先頭から1バイト進めてやり直し
            pos = row_start + 1
            # 行カウンタを戻す
            if row_idx == header_shared[1] - 1:
                header_shared[1] -= 1
            continue
            
        # 有効な行を処理できた
        local_row_count += 1
