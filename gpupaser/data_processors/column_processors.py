"""
PostgreSQLカラムデータ処理用のCUDAカーネル
"""

import numpy as np
from numba import cuda

from gpupaser.cuda_kernels import decode_int32, decode_numeric_postgres, bulk_copy_64bytes

@cuda.jit(device=True)
def process_string_data(src, src_pos, src_len, dst, dst_pos, max_len):
    """
    PostgreSQLバイナリ形式の文字列データを処理するデバイス関数
    
    Args:
        src: ソースデータ配列
        src_pos: ソース内の開始位置
        src_len: ソースデータの長さ
        dst: 送信先バッファ
        dst_pos: 送信先の開始位置
        max_len: 送信先の最大長さ
        
    Returns:
        実際に処理したバイト数
    """
    # 有効な長さを計算（src_lenが負の場合はNULLデータ）
    if src_len < 0:
        return 0
    
    # 実際のコピー長を計算
    valid_len = np.int32(min(src_len, max_len))
    
    # 指定されたバイト数をすべてコピー（NULL終端を探さない）
    for i in range(valid_len):
        dst[dst_pos + i] = src[src_pos + i]
    
    # 残りのバッファをゼロクリア
    for i in range(valid_len, max_len):
        dst[dst_pos + i] = 0
    
    return valid_len

@cuda.jit
def decode_all_columns_kernel(raw_data, field_offsets, field_lengths,
                            int_outputs, num_hi_output, num_lo_output, num_scale_output,
                            str_outputs, str_null_pos, str_offsets,
                            col_types, col_lengths, chunk_size, num_cols):
    """
    全カラムを一度に処理する統合カーネル
    
    PostgreSQLバイナリデータから各カラムのデータを抽出し、型に応じて適切な形式に変換します。
    整数型、numeric型、文字列型などの複数のデータ型に対応し、並列に処理します。
    
    Args:
        raw_data: バイナリデータ配列
        field_offsets: フィールド位置の配列
        field_lengths: フィールド長の配列
        int_outputs: 整数型カラム出力用バッファ
        num_hi_output: numeric型上位64ビット出力用バッファ
        num_lo_output: numeric型下位64ビット出力用バッファ
        num_scale_output: numeric型スケール出力用バッファ
        str_outputs: 文字列型出力用バッファ
        str_null_pos: 文字列型NULL位置出力用バッファ
        str_offsets: 文字列型オフセット配列
        col_types: カラム型情報配列
        col_lengths: カラム長さ情報配列
        chunk_size: チャンクサイズ（行数）
        num_cols: カラム数
    """
    # スレッドインデックスの計算を改善
    thread_id = np.int32(cuda.threadIdx.x)
    block_id = np.int32(cuda.blockIdx.x)
    block_size = np.int32(cuda.blockDim.x)
    grid_size = np.int32(cuda.gridDim.x)
    
    # グリッド内の絶対位置を計算
    row = np.int32(block_id * block_size + thread_id)
    
    # ストライド処理を追加 - 効率的に全行を処理
    stride = np.int32(block_size * grid_size)
    while row < chunk_size:
        # 各カラムを処理
        for col in range(num_cols):
            # フィールドオフセットとデータ長を取得
            field_idx = np.int32(row * num_cols + col)
            if field_idx >= len(field_offsets):
                return
            
            pos = np.int32(field_offsets[field_idx])
            length = np.int32(field_lengths[field_idx])
            
            # データ型に応じた処理
            if col_types[col] <= 1:  # 数値型
                if length == -1:  # NULL値
                    # NULL値はゼロとして扱う
                    if col_types[col] == 0:  # integer
                        int_outputs[np.int32(col * chunk_size + row)] = 0
                    else:  # numeric
                        # numeric型のNULL値は0として記録
                        num_hi_output[np.int32(col * chunk_size + row)] = 0
                        num_lo_output[np.int32(col * chunk_size + row)] = 0
                        if row == 0:  # 最初の行でのみスケールを設定
                            num_scale_output[col] = 0
                else:
                    if col_types[col] == 0:  # integer
                        # 正確にint32として解釈
                        val = decode_int32(raw_data, pos)
                        int_outputs[np.int32(col * chunk_size + row)] = val
                    else:  # numeric/decimal
                        # PostgreSQLのnumeric型を128ビット固定小数点数として処理
                        decode_numeric_postgres(raw_data, pos,
                                              num_hi_output[np.int32(col * chunk_size):],
                                              num_lo_output[np.int32(col * chunk_size):],
                                              num_scale_output[np.int32(col):np.int32(col + 1)],
                                              row)
            else:  # 文字列型
                max_length = np.int32(col_lengths[col])
                
                # 文字列カラムのインデックスを計算 (文字列型のカラム数をカウント)
                str_col_idx = np.int32(0)
                for i in range(col):
                    if col_types[i] == 2:  # 文字列型
                        str_col_idx += 1
                
                # 文字列バッファの位置を計算
                buffer_offset = np.int32(str_offsets[str_col_idx])
                dst_pos = np.int32(buffer_offset + row * max_length)
                
                if length == -1:  # NULL値
                    # NULL値の場合は長さを0とする
                    str_null_pos[np.int32(str_col_idx * chunk_size + row)] = 0
                    continue
                
                # 文字列データの処理 - 専用関数を使用
                actual_length = process_string_data(
                    raw_data, pos, length,
                    str_outputs, dst_pos, max_length
                )
                
                # 実際の文字列長を記録 - 可変長文字列のための情報
                str_null_pos[np.int32(str_col_idx * chunk_size + row)] = actual_length
        
        # 次の行へ (ストライド処理)
        row += stride
