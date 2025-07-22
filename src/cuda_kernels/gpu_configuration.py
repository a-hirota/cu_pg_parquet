"""
GPU設定ユーティリティ
====================

GPU実行パラメータの計算とGrid/Blockサイズの最適化
"""

import math

import numpy as np
from numba import cuda, types
from numba.cuda import atomic


@cuda.jit
def find_row_start_offsets_parallel(
    raw_data, row_offsets, total_rows_found, data_size, stride_size=1024
):
    """
    並列化された行開始オフセット検出カーネル

    各スレッドが一定間隔で0xFFFFマーカーを並列検索し、
    行開始位置をatomic操作で収集します。
    """
    thread_id = cuda.grid(1)

    # 各スレッドの検索開始位置
    start_pos = thread_id * stride_size

    if start_pos >= data_size - 1:
        return

    # 終了位置の計算（次のスレッドと重複しないように）
    end_pos = min(start_pos + stride_size, data_size - 1)

    # 共有メモリでローカル結果をキャッシュ
    local_offsets = cuda.shared.array(256, dtype=types.int32)
    local_count = cuda.shared.array(1, dtype=types.int32)

    # 共有メモリ初期化
    if cuda.threadIdx.x == 0:
        local_count[0] = 0

    cuda.syncthreads()

    # ストライド内で行終端マーカー（0xFFFF）を検索
    for pos in range(start_pos, end_pos - 1):
        if raw_data[pos] == 0xFF and raw_data[pos + 1] == 0xFF:
            # 次の行の開始位置は pos + 2
            row_start = pos + 2

            # 共有メモリに一時保存
            local_idx = atomic.add(local_count, 0, 1)
            if local_idx < 256:  # 共有メモリサイズ制限
                local_offsets[local_idx] = row_start

    cuda.syncthreads()

    # 共有メモリからグローバルメモリに結果をコピー
    if cuda.threadIdx.x == 0 and local_count[0] > 0:
        # グローバルインデックスを原子的に取得
        global_start_idx = atomic.add(total_rows_found, 0, local_count[0])

        # ローカル結果をグローバル配列にコピー
        for i in range(min(local_count[0], 256)):
            if global_start_idx + i < row_offsets.size:
                row_offsets[global_start_idx + i] = local_offsets[i]


@cuda.jit
def count_rows_parallel(raw_data, row_counter, data_size, stride_size=1024):
    """
    並列化された行数カウントカーネル

    メモリコアレッシングを考慮した行数カウント
    """
    thread_id = cuda.grid(1)

    # 各スレッドの検索範囲
    start_pos = thread_id * stride_size
    end_pos = min(start_pos + stride_size, data_size - 1)

    if start_pos >= data_size - 1:
        return

    local_count = 0

    # ベクトル化されたメモリアクセスを利用
    # 4バイト単位で読み取り、並列パターンマッチング
    for pos in range(start_pos, end_pos - 3, 4):
        # 4バイトを一度に読み取り
        if pos + 3 < data_size:
            chunk = (
                raw_data[pos] << 24
                | raw_data[pos + 1] << 16
                | raw_data[pos + 2] << 8
                | raw_data[pos + 3]
            )

            # 0xFFFFパターンの検出
            for offset in range(3):
                pattern = (chunk >> (8 * (2 - offset))) & 0xFFFF
                if pattern == 0xFFFF:
                    local_count += 1

    # 残りのバイトを個別に処理
    remainder_start = ((end_pos - start_pos) // 4) * 4 + start_pos
    for pos in range(remainder_start, end_pos - 1):
        if raw_data[pos] == 0xFF and raw_data[pos + 1] == 0xFF:
            local_count += 1

    # 原子的に総数に加算
    if local_count > 0:
        atomic.add(row_counter, 0, local_count)


@cuda.jit
def extract_fields_coalesced(
    raw_data, row_offsets, field_offsets, field_lengths, num_rows, num_cols
):
    """
    メモリコアレッシング対応フィールド抽出カーネル

    ワープ内のスレッドが連続するメモリアクセスを行うよう最適化
    """
    # 2Dグリッド: x=行インデックス, y=列インデックス
    row_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    col_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if row_idx >= num_rows or col_idx >= num_cols:
        return

    # 共有メモリでローカルデータをキャッシュ
    shared_raw_data = cuda.shared.array(1024, dtype=types.uint8)

    # 行の開始位置
    row_start = row_offsets[row_idx]

    # 列ヘッダー読み込み（先頭4バイトは列数）
    header_pos = row_start + 4  # 列数をスキップ

    # 各列のフィールド情報を取得
    field_start_pos = header_pos + col_idx * 4

    # フィールド長を読み取り（ビッグエンディアン）
    if field_start_pos + 3 < raw_data.size:
        field_len = (
            (raw_data[field_start_pos] << 24)
            | (raw_data[field_start_pos + 1] << 16)
            | (raw_data[field_start_pos + 2] << 8)
            | raw_data[field_start_pos + 3]
        )

        # 負の値はNULL
        if field_len < 0:
            field_lengths[row_idx, col_idx] = 0
            field_offsets[row_idx, col_idx] = 0
        else:
            field_lengths[row_idx, col_idx] = field_len

            # フィールドデータ開始位置の計算
            data_start = header_pos + num_cols * 4

            # 前の列のデータサイズを累積
            field_offset = data_start
            for prev_col in range(col_idx):
                prev_pos = header_pos + prev_col * 4
                if prev_pos + 3 < raw_data.size:
                    prev_len = (
                        (raw_data[prev_pos] << 24)
                        | (raw_data[prev_pos + 1] << 16)
                        | (raw_data[prev_pos + 2] << 8)
                        | raw_data[prev_pos + 3]
                    )
                    if prev_len > 0:
                        field_offset += prev_len

            field_offsets[row_idx, col_idx] = field_offset


@cuda.jit
def unified_decode_with_coalescing(
    raw_data,
    field_offsets,
    field_lengths,
    # 列メタデータ
    column_types,
    column_is_variable,
    # 固定長出力バッファ（列指向）
    fixed_outputs,  # 各列用のバッファ配列
    fixed_col_sizes,  # 各列のサイズ
    # 可変長出力バッファ
    var_data_buffer,
    var_offset_arrays,
    var_column_mapping,
    # NULL配列
    null_flags,
    num_rows,
    num_cols,
):
    """
    メモリコアレッシング対応統合デコードカーネル

    列指向のメモリレイアウトでワープ内の連続アクセスを実現
    """
    row_idx = cuda.grid(1)

    if row_idx >= num_rows:
        return

    # 共有メモリでデータをキャッシュ
    shared_buffer = cuda.shared.array(512, dtype=types.uint8)

    # 各列を順次処理
    for col_idx in range(num_cols):
        field_offset = field_offsets[row_idx, col_idx]
        field_length = field_lengths[row_idx, col_idx]

        # NULL チェック
        if field_length == 0:
            null_flags[row_idx, col_idx] = 1
            continue
        else:
            null_flags[row_idx, col_idx] = 0

        col_type = column_types[col_idx]
        is_variable = column_is_variable[col_idx] != 0

        if not is_variable:
            # 固定長列: 列指向バッファに直接書き込み
            col_size = fixed_col_sizes[col_idx]
            output_offset = row_idx * col_size

            # データをコピー（メモリコアレッシング対応）
            copy_size = min(field_length, col_size)
            for i in range(copy_size):
                if (
                    field_offset + i < raw_data.size
                    and output_offset + i < fixed_outputs[col_idx].size
                ):
                    fixed_outputs[col_idx][output_offset + i] = raw_data[field_offset + i]

        else:
            # 可変長列: 専用バッファに追加
            var_idx = var_column_mapping[col_idx]
            if var_idx >= 0:
                # 原子的にオフセットを取得
                current_offset = atomic.add(var_offset_arrays, (var_idx, row_idx + 1), field_length)

                # データをコピー
                for i in range(field_length):
                    if (
                        field_offset + i < raw_data.size
                        and current_offset + i < var_data_buffer.size
                    ):
                        var_data_buffer[current_offset + i] = raw_data[field_offset + i]


def calculate_gpu_grid_dimensions(data_size: int, num_rows: int, device_props: dict) -> tuple:
    """
    GPU特性に基づいたGrid/Blockサイズを計算

    Args:
        data_size: 処理データサイズ
        num_rows: 行数
        device_props: GPU デバイス特性

    Returns:
        (blocks, threads) のタプル
    """

    # デバイス特性の取得
    max_threads_per_block = device_props.get("MAX_THREADS_PER_BLOCK", 1024)
    multiprocessor_count = device_props.get("MULTIPROCESSOR_COUNT", 16)
    max_blocks_per_grid = device_props.get("MAX_GRID_DIM_X", 65535)

    # スレッド数の計算
    # - 1ワープ = 32スレッド を基本単位とする
    # - SMあたり複数ブロックを配置してオキュパンシーを最大化
    threads = 256  # 256 = 8ワープ, 良好なオキュパンシー

    # 行数ベースのブロック数計算
    if num_rows > 0:
        blocks_for_rows = (num_rows + threads - 1) // threads
    else:
        blocks_for_rows = 1

    # データサイズベースのブロック数計算（行検出用）
    stride_size = 1024
    blocks_for_data = (data_size + stride_size - 1) // stride_size

    # 最小でもSM数の倍数のブロックを確保
    min_blocks = multiprocessor_count * 2  # SMあたり2ブロック

    # 最終的なブロック数の決定
    final_blocks = max(min_blocks, blocks_for_rows, blocks_for_data // 4)
    final_blocks = min(final_blocks, max_blocks_per_grid)

    return final_blocks, threads


def parse_binary_chunk_gpu_enhanced(
    raw_dev, ncols: int, threads_per_block: int = 256, header_size: int = None
) -> tuple:
    """
    GPU拡張バイナリチャンクパース

    並列化とメモリコアレッシングを活用した高性能版
    既存の parse_binary_chunk_gpu と同じシグネチャで互換性を保持
    """

    # header_size の処理（既存版と同じ）
    if header_size is None:
        # PostgreSQL COPY BINARY の標準ヘッダーサイズを使用
        # 詳細検出は呼び出し元で実行済みと想定
        header_size = 19

    # デバイス特性をデフォルト値で設定
    device_props = {
        "MAX_THREADS_PER_BLOCK": 1024,
        "MULTIPROCESSOR_COUNT": 16,
        "MAX_GRID_DIM_X": 65535,
    }

    data_size = raw_dev.shape[0] - header_size
    if data_size <= 0:
        return cuda.device_array((0, ncols), dtype=np.int32), cuda.device_array(
            (0, ncols), dtype=np.int32
        )

    # データ部分の抽出
    data_section = raw_dev[header_size:]

    # === 1. 並列行数カウント ===
    row_counter = cuda.device_array(1, dtype=np.int32)
    row_counter[0] = 0

    # Grid/Blockサイズの計算
    count_blocks, count_threads = calculate_gpu_grid_dimensions(data_size, 0, device_props)

    count_rows_parallel_optimized[count_blocks, count_threads](data_section, row_counter, data_size)
    cuda.synchronize()

    total_rows = int(row_counter.copy_to_host()[0])
    if total_rows == 0:
        return cuda.device_array((0, ncols), dtype=np.int32), cuda.device_array(
            (0, ncols), dtype=np.int32
        )

    # === 2. 並列行オフセット収集 ===
    row_offsets = cuda.device_array(total_rows, dtype=np.int32)
    rows_found_counter = cuda.device_array(1, dtype=np.int32)
    rows_found_counter[0] = 0

    # Grid/Blockサイズの計算
    offset_blocks, offset_threads = calculate_gpu_grid_dimensions(
        data_size, total_rows, device_props
    )

    find_row_start_offsets_parallel_optimized[offset_blocks, offset_threads](
        data_section, row_offsets, rows_found_counter, data_size
    )
    cuda.synchronize()

    actual_rows_found = int(rows_found_counter.copy_to_host()[0])
    if actual_rows_found == 0:
        return cuda.device_array((0, ncols), dtype=np.int32), cuda.device_array(
            (0, ncols), dtype=np.int32
        )

    # === 3. 並列フィールド抽出 ===
    field_offsets_dev = cuda.device_array((actual_rows_found, ncols), dtype=np.int32)
    field_lengths_dev = cuda.device_array((actual_rows_found, ncols), dtype=np.int32)

    # 2Dブロック構成でメモリコアレッシング対応
    threads_x = 16  # 行方向
    threads_y = 16  # 列方向
    blocks_x = (actual_rows_found + threads_x - 1) // threads_x
    blocks_y = (ncols + threads_y - 1) // threads_y

    extract_fields_coalesced_optimized[(blocks_x, blocks_y), (threads_x, threads_y)](
        data_section, row_offsets, field_offsets_dev, field_lengths_dev, actual_rows_found, ncols
    )
    cuda.synchronize()

    return field_offsets_dev, field_lengths_dev


__all__ = [
    "find_row_start_offsets_parallel",
    "count_rows_parallel",
    "extract_fields_coalesced",
    "unified_decode_with_coalescing",
    "calculate_gpu_grid_dimensions",
    "parse_binary_chunk_gpu_enhanced",
]
