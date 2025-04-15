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
    全カラムを一度に処理する統合カーネル - 最適化バージョン
    
    PostgreSQLバイナリデータから各カラムのデータを抽出し、型に応じて適切な形式に変換します。
    整数型、numeric型、文字列型などの複数のデータ型に対応し、並列に処理します。
    ワープ効率とメモリアクセスパターンを最適化しています。
    
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
    # ワープ関連の定数（ハードコーディングして最適化）
    WARP_SIZE = 32
    
    # スレッドとブロック情報
    thread_id = np.int32(cuda.threadIdx.x)
    block_id = np.int32(cuda.blockIdx.x)
    block_size = np.int32(cuda.blockDim.x)
    grid_size = np.int32(cuda.gridDim.x)
    
    # ワープIDとレーンIDの計算（ワープ内での協調処理に使用）
    warp_id = thread_id // WARP_SIZE
    lane_id = thread_id % WARP_SIZE
    
    # グリッド内の絶対位置を計算
    row = np.int32(block_id * block_size + thread_id)
    
    # 事前計算：文字列カラムのインデックスマップを準備（ワープ内で共有）
    # これにより各カラムごとの繰り返し計算を回避
    str_col_indices = cuda.shared.array(shape=(64,), dtype=np.int32)  # 最大64カラムまで対応
    
    # 各ワープの先頭スレッドが初期化
    if lane_id == 0:
        # 文字列カラムのインデックスマップを作成
        str_idx = 0
        for i in range(min(64, num_cols)):
            if i < num_cols and col_types[i] == 2:  # 文字列型
                str_col_indices[i] = str_idx
                str_idx += 1
            else:
                str_col_indices[i] = -1  # 文字列型以外は-1
    
    # ワープ内同期
    cuda.syncwarp()
    
    # ストライド処理を追加 - 効率的に全行を処理
    stride = np.int32(block_size * grid_size)
    while row < chunk_size:
        # 各カラムを処理（条件分岐を減らして最適化）
        col = 0
        while col < num_cols:
            # フィールドオフセットとデータ長をアトミックに取得
            field_idx = np.int32(row * num_cols + col)
            if field_idx >= len(field_offsets):
                return
            
            pos = np.int32(field_offsets[field_idx])
            length = np.int32(field_lengths[field_idx])
            
            # データ型に基づいて処理を分岐（ワープダイバージェンスを最小化）
            col_type = col_types[col]
            
            # 数値型処理（整数、numeric）
            if col_type <= 1:
                # NULL値と有効値の両方を効率的に処理
                is_null = (length == -1)
                
                if col_type == 0:  # integer
                    # NULL値の場合は0、それ以外は実際の値を設定
                    output_idx = np.int32(col * chunk_size + row)
                    val = 0 if is_null else decode_int32(raw_data, pos)
                    int_outputs[output_idx] = val
                else:  # numeric
                    output_idx = np.int32(col * chunk_size + row)
                    if is_null:
                        # NULL値の処理
                        num_hi_output[output_idx] = 0
                        num_lo_output[output_idx] = 0
                        if row == 0:  # 最初の行でのみスケールを設定
                            num_scale_output[col] = 0
                    else:
                        # 有効値の処理
                        decode_numeric_postgres(
                            raw_data, pos,
                            num_hi_output[np.int32(col * chunk_size):],
                            num_lo_output[np.int32(col * chunk_size):],
                            num_scale_output[np.int32(col):np.int32(col + 1)],
                            row
                        )
            else:  # 文字列型
                # 事前計算したインデックスを使用（条件分岐削減）
                str_col_idx = str_col_indices[col]
                if str_col_idx >= 0:  # 有効な文字列カラム
                    max_length = np.int32(col_lengths[col])
                    
                    # 文字列バッファの位置を計算
                    buffer_offset = np.int32(str_offsets[str_col_idx])
                    dst_pos = np.int32(buffer_offset + row * max_length)
                    str_idx = np.int32(str_col_idx * chunk_size + row)
                    
                    if length == -1:  # NULL値
                        # NULL値の場合は長さを0とする
                        str_null_pos[str_idx] = 0
                    else:
                        # 文字列データの処理 - 専用関数を使用
                        actual_length = process_string_data(
                            raw_data, pos, length,
                            str_outputs, dst_pos, max_length
                        )
                        
                        # 実際の文字列長を記録
                        str_null_pos[str_idx] = actual_length
            
            # 次のカラムへ
            col += 1
        
        # 次の行へ (ストライド処理)
        row += stride
