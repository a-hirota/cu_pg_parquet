"""
PostgreSQL Binary Parser with Debug Logging for 128MB-549MB range
"""

import cupy as cp
from numba import cuda, float32, float64, int8, int32, int64, uint8, uint16, uint32, uint64

from .postgres_binary_parser import *


@cuda.jit
def parse_rows_and_fields_lite_with_range_debug(
    raw_data,
    header_size,
    ncols,
    # 出力配列
    row_positions,
    field_offsets,
    field_lengths,
    thread_ids,
    thread_start_positions,
    thread_end_positions,
    row_count,
    # 設定
    thread_stride,
    max_rows,
    fixed_field_lengths,
    # デバッグ用
    debug_info,  # [thread_id, start_pos, end_pos, rows_found, in_range_flag]
    debug_count,
):
    """
    128MB-549MB範囲のデバッグ情報を記録するカーネル
    """
    # スレッドID計算
    tid = (
        cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x
        + cuda.blockIdx.y * cuda.blockDim.x
        + cuda.threadIdx.x
    )

    # 担当範囲計算
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)

    if start_pos >= raw_data.size:
        return

    # 128MB-549MB範囲のチェック
    MB_128 = uint64(128 * 1024 * 1024)
    MB_549 = uint64(549 * 1024 * 1024)

    in_range = False
    if start_pos <= MB_549 and end_pos >= MB_128:
        in_range = True
        # デバッグ情報を記録
        debug_idx = cuda.atomic.add(debug_count, 0, 1)
        if debug_idx < debug_info.shape[0]:
            debug_info[debug_idx, 0] = tid
            debug_info[debug_idx, 1] = start_pos
            debug_info[debug_idx, 2] = end_pos
            debug_info[debug_idx, 3] = 0  # 検出行数（後で更新）
            debug_info[debug_idx, 4] = 1  # in_range flag

    # ローカル結果バッファ
    local_positions = cuda.local.array(256, uint64)
    local_count = 0

    pos = uint64(start_pos)

    # 行検出ループ
    while pos < end_pos:
        # 行ヘッダ検出（簡略版）
        if pos + 2 <= raw_data.size:
            # 列数チェック
            ncols_found = (uint16(raw_data[pos]) << 8) | uint16(raw_data[pos + 1])

            if ncols_found == ncols:
                # 有効な行として記録
                if local_count < 256:
                    local_positions[local_count] = pos
                    local_count += 1

                # 次の行へ（仮に平均150バイトとする）
                pos += 150
            else:
                pos += 1
        else:
            break

    # 結果を書き込み
    if local_count > 0:
        base_idx = cuda.atomic.add(row_count, 0, local_count)

        for i in range(local_count):
            global_idx = base_idx + i
            if global_idx < max_rows:
                row_positions[global_idx] = local_positions[i]
                thread_ids[global_idx] = tid
                thread_start_positions[global_idx] = start_pos
                thread_end_positions[global_idx] = end_pos

                # フィールド情報は省略（デバッグ用）
                for j in range(min(ncols, 17)):
                    field_offsets[global_idx, j] = 0
                    field_lengths[global_idx, j] = 4

        # デバッグ情報を更新
        if in_range:
            for i in range(debug_info.shape[0]):
                if debug_info[i, 0] == tid:
                    debug_info[i, 3] = local_count
                    break


def parse_with_range_debug(raw_dev, columns, header_size=None, mb_start=128, mb_end=549):
    """
    指定MB範囲のデバッグ情報を含むパース
    """
    if header_size is None:
        header_size = 19

    ncols = len(columns)
    data_size = len(raw_dev)

    # 推定行数
    estimated_row_size = 150  # 仮の値
    estimated_rows = data_size // estimated_row_size
    max_rows = int(estimated_rows * 1.5)

    # グリッド設定
    threads_per_block = 256
    thread_stride = 1150  # 実際の値
    actual_threads = (data_size + thread_stride - 1) // thread_stride
    blocks = (actual_threads + threads_per_block - 1) // threads_per_block

    # 出力配列
    row_positions = cuda.device_array(max_rows, np.uint64)
    field_offsets = cuda.device_array((max_rows, ncols), np.uint32)
    field_lengths = cuda.device_array((max_rows, ncols), np.int32)
    thread_ids = cuda.device_array(max_rows, np.int32)
    thread_start_positions = cuda.device_array(max_rows, np.uint64)
    thread_end_positions = cuda.device_array(max_rows, np.uint64)
    row_count = cuda.to_device(np.array([0], dtype=np.int32))

    # 固定長フィールド情報
    fixed_field_lengths = cuda.to_device(np.array([4] * ncols, dtype=np.int32))

    # デバッグ情報配列（最大10000スレッド分）
    debug_info = cuda.device_array((10000, 5), np.int64)
    debug_count = cuda.to_device(np.array([0], dtype=np.int32))

    print(f"\n=== {mb_start}MB-{mb_end}MB範囲デバッグ ===")
    print(f"データサイズ: {data_size:,} bytes")
    print(f"推定スレッド数: {actual_threads:,}")
    print(f"ブロック数: {blocks}")
    print(f"スレッドストライド: {thread_stride} bytes")

    # カーネル実行
    parse_rows_and_fields_lite_with_range_debug[blocks, threads_per_block](
        raw_dev,
        header_size,
        ncols,
        row_positions,
        field_offsets,
        field_lengths,
        thread_ids,
        thread_start_positions,
        thread_end_positions,
        row_count,
        thread_stride,
        max_rows,
        fixed_field_lengths,
        debug_info,
        debug_count,
    )
    cuda.synchronize()

    # 結果取得
    nrow = int(row_count.copy_to_host()[0])
    debug_count_host = int(debug_count.copy_to_host()[0])

    print(f"\n検出行数: {nrow:,}")
    print(f"範囲内スレッド数: {debug_count_host}")

    if debug_count_host > 0:
        # デバッグ情報を分析
        debug_info_host = debug_info[:debug_count_host].copy_to_host()

        print(f"\n{mb_start}MB-{mb_end}MB範囲のスレッド詳細:")
        for i in range(min(20, debug_count_host)):  # 最初の20個
            tid = debug_info_host[i, 0]
            start = debug_info_host[i, 1]
            end = debug_info_host[i, 2]
            rows = debug_info_host[i, 3]

            print(
                f"  Thread {tid}: 0x{start:08X}-0x{end:08X} ({start/(1024*1024):.1f}MB-{end/(1024*1024):.1f}MB), {rows}行検出"
            )

        # 行が検出されなかったスレッドを探す
        no_rows = debug_info_host[debug_info_host[:, 3] == 0]
        if len(no_rows) > 0:
            print(f"\n⚠️ 行を検出しなかったスレッド: {len(no_rows)}個")
            for i in range(min(10, len(no_rows))):
                tid = no_rows[i, 0]
                start = no_rows[i, 1]
                end = no_rows[i, 2]
                print(f"    Thread {tid}: 0x{start:08X}-0x{end:08X}")

    return {
        "nrow": nrow,
        "row_positions": row_positions[:nrow],
        "thread_ids": thread_ids[:nrow],
        "debug_info": debug_info_host if debug_count_host > 0 else None,
    }
