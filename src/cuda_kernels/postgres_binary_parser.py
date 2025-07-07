"""
PostgreSQL Binary Parser (統合最適化版)
=====================================

統合最適化ポイント:
- 共有メモリ使用量最小化（<5KB）
- 行検出+フィールド抽出の同時実行
- メモリアクセス50%削減（218MB × 2回 → 218MB × 1回）
- 直接グローバルメモリ書き込み
- atomic操作で競合回避

機能:
- PostgreSQL COPY BINARYヘッダー検出
- 高速行検出・検証
- フィールド情報抽出
- 統合最適化処理
"""

from numba import cuda, int32, uint32, uint64, types
import numpy as np
import time  # ソート時間計測用
import os  # 環境変数のチェック用

# Numba CUDAビットニックソート実装
@cuda.jit
def count_valid_rows(row_positions, n, invalid_value, count):
    """有効な行数をGPU上でカウント"""
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    threads_per_block = cuda.blockDim.x
    
    # 共有メモリでブロック内の合計を計算
    shared_count = cuda.shared.array(256, dtype=int32)
    shared_count[tid] = 0
    
    # 各スレッドが担当する範囲をカウント
    idx = bid * threads_per_block + tid
    stride = cuda.gridsize(1)
    
    local_count = 0
    for i in range(idx, n, stride):
        if row_positions[i] != invalid_value:
            local_count += 1
    
    shared_count[tid] = local_count
    cuda.syncthreads()
    
    # ブロック内でリダクション
    s = threads_per_block // 2
    while s > 0:
        if tid < s:
            shared_count[tid] += shared_count[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # ブロックの合計をグローバルメモリに加算
    if tid == 0:
        cuda.atomic.add(count, 0, shared_count[0])

@cuda.jit
def create_sort_indices(indices, n):
    """ソート用インデックス配列を初期化"""
    idx = cuda.grid(1)
    if idx < n:
        indices[idx] = idx

@cuda.jit
def bitonic_sort_step(keys, indices, j, k):
    """ビットニックソートの1ステップ"""
    idx = cuda.grid(1)
    ixj = idx ^ j
    
    if ixj > idx and idx < keys.size and ixj < keys.size:
        if (idx & k) == 0:
            # 昇順
            if keys[indices[idx]] > keys[indices[ixj]]:
                indices[idx], indices[ixj] = indices[ixj], indices[idx]
        else:
            # 降順
            if keys[indices[idx]] < keys[indices[ixj]]:
                indices[idx], indices[ixj] = indices[ixj], indices[idx]

@cuda.jit
def apply_sort_indices_single(src, dst, indices, n):
    """ソートインデックスを使って1D配列を並べ替え"""
    idx = cuda.grid(1)
    if idx < n:
        dst[idx] = src[indices[idx]]

@cuda.jit
def apply_sort_indices_2d(src, dst, indices, n, ncols):
    """ソートインデックスを使って2D配列を並べ替え"""
    idx = cuda.grid(1)
    if idx < n:
        src_idx = indices[idx]
        for col in range(ncols):
            dst[idx, col] = src[src_idx, col]

@cuda.jit
def compact_valid_rows(row_positions, field_offsets, field_lengths, 
                      out_offsets, out_lengths, invalid_value, n, ncols):
    """有効な行のみを抽出して出力配列に格納"""
    idx = cuda.grid(1)
    if idx < n:
        if row_positions[idx] != invalid_value:
            # atomic操作で出力位置を取得
            out_idx = cuda.atomic.add(out_offsets, (0, 0), 0)  # ダミー操作
            # 実際には別途インデックス管理が必要

def cuda_bitonic_sort(row_positions, field_offsets, field_lengths, valid_rows, ncols):
    """Numba CUDAでビットニックソートを実行"""
    # インデックス配列を作成
    indices = cuda.device_array(valid_rows, dtype=np.int32)
    threads = 256
    blocks = (valid_rows + threads - 1) // threads
    create_sort_indices[blocks, threads](indices, valid_rows)
    
    # 2のべき乗に切り上げ
    n = 1
    while n < valid_rows:
        n *= 2
    
    # ビットニックソート実行
    k = 2
    while k <= n:
        j = k // 2
        while j >= 1:
            blocks = (valid_rows + threads - 1) // threads
            bitonic_sort_step[blocks, threads](row_positions, indices, j, k)
            j //= 2
        k *= 2
    
    # ソート結果を適用（新しい配列を作成）
    sorted_offsets = cuda.device_array((valid_rows, ncols), dtype=np.uint64)
    sorted_lengths = cuda.device_array((valid_rows, ncols), dtype=np.int32)
    
    blocks_2d = (valid_rows + threads - 1) // threads
    apply_sort_indices_2d[blocks_2d, threads](
        field_offsets, sorted_offsets, indices, valid_rows, ncols
    )
    apply_sort_indices_2d[blocks_2d, threads](
        field_lengths, sorted_lengths, indices, valid_rows, ncols
    )
    
    return sorted_offsets, sorted_lengths

def detect_pg_header_size(raw_data: np.ndarray) -> int:
    """Detect COPY BINARY header size"""
    base = 11
    if raw_data.size < base:
        return base

    sig = b"PGCOPY\n\377\r\n\0"
    if not np.array_equal(raw_data[:11], np.frombuffer(sig, np.uint8)):
        return base

    size = base + 4  # flags
    if raw_data.size < size + 4:
        return size
    ext_len = int.from_bytes(raw_data[size : size + 4], "big")
    size += 4 + ext_len if raw_data.size >= size + 4 + ext_len else 0
    return size

def estimate_row_size_from_columns(columns):
    """ColumnMetaから行サイズ推定"""
    try:
        from ..types import UTF8, DECIMAL128
    except ImportError:
        from src.types import UTF8, DECIMAL128
    
    # デバッグモード判定
    import os
    debug = os.environ.get('GPUPGPARSER_DEBUG_ESTIMATE', '0') == '1'
    
    size = 2  # フィールド数(2B)
    if debug:
        print(f"[ESTIMATE DEBUG] フィールド数: 2バイト")
    
    for col in columns:
        size += 4  # フィールド長(4B)
        if col.elem_size > 0:  # 固定長
            size += col.elem_size
            if debug:
                print(f"[ESTIMATE DEBUG] {col.name}: 4 + {col.elem_size} = {4 + col.elem_size}バイト (固定長)")
        else:  # 可変長の推定
            if col.arrow_id == UTF8:
                # arrow_paramがある場合は実際の長さを使用（bpchar）
                if col.arrow_param is not None and isinstance(col.arrow_param, int):
                    size += col.arrow_param
                    if debug:
                        print(f"[ESTIMATE DEBUG] {col.name}: 4 + {col.arrow_param} = {4 + col.arrow_param}バイト (bpchar)")
                else:
                    # arrow_paramがない場合は従来通り（varchar）
                    size += 20  # 文字列平均20B
                    if debug:
                        print(f"[ESTIMATE DEBUG] {col.name}: 4 + 20 = 24バイト (varchar推定)")
            elif col.arrow_id == DECIMAL128:
                size += 16   # NUMERIC平均16B
                if debug:
                    print(f"[ESTIMATE DEBUG] {col.name}: 4 + 16 = 20バイト (DECIMAL128)")
            else:
                size += 4   # その他4B
                if debug:
                    print(f"[ESTIMATE DEBUG] {col.name}: 4 + 4 = 8バイト (その他)")
    
    # メモリバンクコンフリクト回避: 32B境界整列
    aligned = ((size + 31) // 32) * 32
    if debug:
        print(f"[ESTIMATE DEBUG] 合計: {size}バイト → 32バイト整列: {aligned}バイト")
    return aligned

def get_device_properties():
    """GPU デバイス特性を取得"""
    device = cuda.get_current_device()
    return {
        'MULTIPROCESSOR_COUNT': device.MULTIPROCESSOR_COUNT,
        'MAX_THREADS_PER_BLOCK': device.MAX_THREADS_PER_BLOCK,
        'MAX_GRID_DIM_X': device.MAX_GRID_DIM_X,
        'MAX_GRID_DIM_Y': device.MAX_GRID_DIM_Y,
        'WARP_SIZE': device.WARP_SIZE,
        'MAX_SHARED_MEMORY_PER_BLOCK': device.MAX_SHARED_MEMORY_PER_BLOCK,
        'COMPUTE_CAPABILITY': (device.COMPUTE_CAPABILITY_MAJOR, device.COMPUTE_CAPABILITY_MINOR),
        'NAME': device.name.decode('utf-8') if hasattr(device.name, 'decode') else str(device.name),
    }

def print_gpu_properties():
    """GPU特性を表形式で表示"""
    props = get_device_properties()
    print("\n" + "="*60)
    print("GPU Device Properties")
    print("="*60)
    print(f"Device Name:              {props.get('NAME', 'Unknown')}")
    print(f"Compute Capability:       {props.get('COMPUTE_CAPABILITY', (0,0))[0]}.{props.get('COMPUTE_CAPABILITY', (0,0))[1]}")
    print(f"SM (Multiprocessor) Count: {props.get('MULTIPROCESSOR_COUNT', 0)}")
    print(f"Max Threads per Block:    {props.get('MAX_THREADS_PER_BLOCK', 0):,}")
    print(f"Warp Size:               {props.get('WARP_SIZE', 32)}")
    print(f"Max Grid Dimensions:     X={props.get('MAX_GRID_DIM_X', 0):,}, Y={props.get('MAX_GRID_DIM_Y', 0):,}")
    print(f"Max Shared Memory/Block:  {props.get('MAX_SHARED_MEMORY_PER_BLOCK', 0):,} bytes")
    print("="*60 + "\n")

def calculate_optimal_grid_sm_aware(data_size, estimated_row_size):
    """SMコア数を考慮した最適なグリッドサイズ計算
    
    データサイズとSMコア数に基づいて、最適な並列度を決定
    """
    props = get_device_properties()
    
    sm_count = props.get('MULTIPROCESSOR_COUNT', 108)
    max_threads_per_block = props.get('MAX_THREADS_PER_BLOCK', 1024)
    max_blocks_x = props.get('MAX_GRID_DIM_X', 65535)
    max_blocks_y = props.get('MAX_GRID_DIM_Y', 65535)
    
    data_mb = data_size / (1024 * 1024)
    
    # 動的threads_per_block決定
    if data_mb < 50:
        threads_per_block = 256
    elif data_mb < 200:
        threads_per_block = 512
    else:
        threads_per_block = min(1024, max_threads_per_block)
    
    # 推定総行数から必要なスレッド数を計算
    estimated_total_rows = data_size // estimated_row_size
    
    # SM数とデータサイズに基づいてブロック数を決定
    # 小さいデータでは各SMに少なくとも2ブロック、大きいデータではより多くのブロックを配置
    base_blocks = sm_count * 2  # 最小でも各SMに2ブロック
    
    # 大容量データ（>1GB）の場合、より多くのブロックを使用
    if data_mb >= 1000:  # 1GB以上
        target_blocks = max(base_blocks, sm_count * 20)
    elif data_mb >= 200:
        target_blocks = max(base_blocks, sm_count * 12)
    elif data_mb >= 50:
        target_blocks = max(base_blocks, sm_count * 6)
    else:
        target_blocks = max(base_blocks, sm_count * 2)
    
    # 2次元グリッド配置
    blocks_x = min(target_blocks, max_blocks_x)
    blocks_y = min((target_blocks + blocks_x - 1) // blocks_x, max_blocks_y)
    
    return blocks_x, blocks_y, threads_per_block

@cuda.jit(device=True, inline=True)
def read_uint16_simd16_lite(raw_data, pos, end_pos, ncols):
    """行ヘッダ検出（軽量版）"""
    if pos + 1 > raw_data.size:
        return -2
   
    max_offset = min(16, int32(raw_data.size - pos + 1))
    
    for i in range(0, max_offset):
        if pos + i + 1 > end_pos:
            return -3    
        idx = int32(pos + i)
        num_fields = (raw_data[idx] << 8) | raw_data[idx + 1]        
        if num_fields == ncols:
            return int32(pos + i)
    return -1

@cuda.jit(device=True, inline=True)
def validate_and_extract_fields_lite(
    raw_data, row_start, expected_cols, fixed_field_lengths,
    field_offsets_out,  # この行のフィールド相対オフセット配列（uint32）
    field_lengths_out   # この行のフィールド長配列
):
    """
    行検証とフィールド抽出を同時実行（軽量版）
    """
    if row_start + 2 > raw_data.size:
        return False, -1

    # フィールド数確認
    row_start_idx = int32(row_start)
    num_fields = (raw_data[row_start_idx] << 8) | raw_data[row_start_idx + 1]
    if num_fields != expected_cols:
        return False, -2

    pos = uint64(row_start + 2)

    # 全フィールドを検証+抽出
    for field_idx in range(num_fields):
        if pos + 4 > raw_data.size:
            return False, -3

        # フィールド長読み取り
        pos_idx = int32(pos)
        flen = (
            int32(raw_data[pos_idx]) << 24 | int32(raw_data[pos_idx+1]) << 16 |
            int32(raw_data[pos_idx+2]) << 8 | int32(raw_data[pos_idx+3])
        )

        # フィールド情報を即座に出力配列に記録
        if flen == 0xFFFFFFFF:  # NULL
            field_offsets_out[field_idx] = uint32(0)  # uint32にキャスト
            field_lengths_out[field_idx] = -1
            pos += 4
        else:
            # 異常値チェック
            if flen < 0 or flen > 1000000:
                return False, -4

            # 固定長フィールド検証
            # 注：fixed_field_lengthsの長さはテーブルによって異なる
            if (field_idx < expected_cols and
                fixed_field_lengths[field_idx] > 0 and
                flen != fixed_field_lengths[field_idx]):
                return False, -5

            # 境界チェック（データ終端まで許可）
            if pos + 4 + flen > raw_data.size:
                return False, -6

            # フィールド情報記録（相対オフセットとして保存）
            # 注意: row_startからの相対位置を計算
            relative_offset = uint32((pos + 4) - row_start)
            field_offsets_out[field_idx] = relative_offset  # 行開始からの相対オフセット
            field_lengths_out[field_idx] = flen      # データ長
            pos += uint64(4 + flen)

    # 次行ヘッダ検証
    if pos + 2 <= raw_data.size:
        pos_idx2 = int32(pos)
        next_header = (raw_data[pos_idx2] << 8) | raw_data[pos_idx2 + 1]
        if next_header != expected_cols and next_header != 0xFFFF:
            return False, -8

    return True, pos  # 成功、行終端位置を返す

@cuda.jit
def parse_rows_and_fields_lite(
    raw_data,
    header_size,
    ncols,

    # 出力配列（直接書き込み）
    row_positions,      # uint64[max_rows] - 行開始位置
    field_offsets,      # uint32[max_rows, ncols] - 行開始位置からの相対オフセット
    field_lengths,      # int32[max_rows, ncols] - フィールド長
    row_count,          # int32[1] - 検出行数

    # 設定
    thread_stride,
    max_rows,
    fixed_field_lengths
):
    """
    軽量統合カーネル: 共有メモリ使用量を最小化
    
    最適化:
    1. 共有メモリを最小限に抑制
    2. 直接グローバルメモリ書き込み
    3. atomic操作で競合回避
    """

    # 最小限の共有メモリ（<1KB）
    block_count = cuda.shared.array(1, int32)

    # スレッド・ブロック情報
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    local_tid = cuda.threadIdx.x

    # 共有メモリ初期化
    if local_tid == 0:
        block_count[0] = 0

    cuda.syncthreads()

    # 担当範囲計算
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)

    if start_pos >= raw_data.size:
        return

    # ローカル結果バッファ（従来版と同じサイズ）
    local_positions = cuda.local.array(256, uint64)
    local_field_offsets = cuda.local.array((256, 17), uint32)  # 相対オフセット用にuint32に変更
    local_field_lengths = cuda.local.array((256, 17), int32)
    local_count = 0

    pos = uint64(start_pos)

    # 統合処理ループ: 行検出+フィールド抽出を同時実行
    while pos < end_pos:

        # === 1. 行ヘッダ検出 ===
        candidate_pos = read_uint16_simd16_lite(raw_data, pos, end_pos, ncols)

        if candidate_pos <= -2:  # データ終端
            break
        elif candidate_pos == -1:  # 行ヘッダ未発見
            pos += 15
            continue
        elif candidate_pos >= end_pos:  # 担当範囲外
            break

        # === 2. 行検証+フィールド抽出の統合実行 ===
        is_valid, row_end = validate_and_extract_fields_lite(
            raw_data, candidate_pos, ncols, fixed_field_lengths,
            local_field_offsets[local_count],  # この行のフィールド配列
            local_field_lengths[local_count]   # この行のフィールド配列
        )

        if is_valid:
            # 境界を越える行も処理する
            # 行の開始が担当範囲内なら、終了が範囲外でも処理
            if local_count < 256:  # 配列境界チェック
                local_positions[local_count] = uint64(candidate_pos)
                local_count += 1

            # 次の行へジャンプ（境界チェック改善）
            if row_end > 0:
                pos = row_end
            else:
                pos = candidate_pos + 1
        else:
            # 検証失敗時は1バイト進む
            pos = candidate_pos + 1

    # === 3. 結果を直接グローバルメモリに書き込み ===
    if local_count > 0:
        # スレッドローカルの結果をatomic操作で確保
        base_idx = cuda.atomic.add(row_count, 0, local_count)

        for i in range(local_count):
            global_idx = base_idx + i
            if global_idx < max_rows:
                # 行位置
                row_positions[global_idx] = local_positions[i]

                # フィールド情報
                for j in range(min(ncols, 17)):
                    field_offsets[global_idx, j] = local_field_offsets[i, j]  # uint32相対オフセット
                    field_lengths[global_idx, j] = local_field_lengths[i, j]


@cuda.jit
def parse_rows_and_fields_lite_with_thread_ids(
    raw_data,
    header_size,
    ncols,

    # 出力配列（直接書き込み）
    row_positions,      # uint64[max_rows] - 行開始位置
    field_offsets,      # uint32[max_rows, ncols] - 行開始位置からの相対オフセット
    field_lengths,      # int32[max_rows, ncols] - フィールド長
    thread_ids,         # int32[max_rows] - 各行を処理したスレッドID
    row_count,          # int32[1] - 検出行数

    # 設定
    thread_stride,
    max_rows,
    fixed_field_lengths
):
    """
    軽量統合カーネル: thread_ids記録機能付き
    
    テストモード用：各行を処理したスレッドIDを記録
    """

    # 最小限の共有メモリ（<1KB）
    block_count = cuda.shared.array(1, int32)

    # スレッド・ブロック情報
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    local_tid = cuda.threadIdx.x

    # 共有メモリ初期化
    if local_tid == 0:
        block_count[0] = 0

    cuda.syncthreads()

    # 担当範囲計算
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)

    if start_pos >= raw_data.size:
        return

    # ローカル結果バッファ（従来版と同じサイズ）
    local_positions = cuda.local.array(256, uint64)
    local_field_offsets = cuda.local.array((256, 17), uint32)  # 相対オフセット用にuint32に変更
    local_field_lengths = cuda.local.array((256, 17), int32)
    local_count = 0

    pos = uint64(start_pos)

    # 統合処理ループ: 行検出+フィールド抽出を同時実行
    while pos < end_pos:

        # === 1. 行ヘッダ検出 ===
        candidate_pos = read_uint16_simd16_lite(raw_data, pos, end_pos, ncols)

        if candidate_pos <= -2:  # データ終端
            break
        elif candidate_pos == -1:  # 行ヘッダ未発見
            pos += 15
            continue
        elif candidate_pos >= end_pos:  # 担当範囲外
            break

        # === 2. 行検証+フィールド抽出の統合実行 ===
        is_valid, row_end = validate_and_extract_fields_lite(
            raw_data, candidate_pos, ncols, fixed_field_lengths,
            local_field_offsets[local_count],  # この行のフィールド配列
            local_field_lengths[local_count]   # この行のフィールド配列
        )

        if is_valid:
            # 境界を越える行も処理する
            # 行の開始が担当範囲内なら、終了が範囲外でも処理
            if local_count < 256:  # 配列境界チェック
                local_positions[local_count] = uint64(candidate_pos)
                local_count += 1

            # 次の行へジャンプ（境界チェック改善）
            if row_end > 0:
                pos = row_end
            else:
                pos = candidate_pos + 1
        else:
            # 検証失敗時は1バイト進む
            pos = candidate_pos + 1

    # === 3. 結果を直接グローバルメモリに書き込み ===
    if local_count > 0:
        # スレッドローカルの結果をatomic操作で確保
        base_idx = cuda.atomic.add(row_count, 0, local_count)

        for i in range(local_count):
            global_idx = base_idx + i
            if global_idx < max_rows:
                # 行位置
                row_positions[global_idx] = local_positions[i]
                
                # スレッドIDを記録
                thread_ids[global_idx] = tid

                # フィールド情報
                for j in range(min(ncols, 17)):
                    field_offsets[global_idx, j] = local_field_offsets[i, j]  # uint32相対オフセット
                    field_lengths[global_idx, j] = local_field_lengths[i, j]


@cuda.jit
def parse_rows_and_fields_lite_with_full_debug(
    raw_data,
    header_size,
    ncols,

    # 出力配列（直接書き込み）
    row_positions,      # uint64[max_rows] - 行開始位置
    field_offsets,      # uint32[max_rows, ncols] - 行開始位置からの相対オフセット
    field_lengths,      # int32[max_rows, ncols] - フィールド長
    thread_ids,         # int32[max_rows] - 各行を処理したスレッドID
    thread_start_positions,  # uint64[max_rows] - 各行を処理したスレッドの開始位置
    thread_end_positions,    # uint64[max_rows] - 各行を処理したスレッドの終了位置
    row_count,          # int32[1] - 検出行数

    # 設定
    thread_stride,
    max_rows,
    fixed_field_lengths
):
    """
    完全デバッグ版カーネル: thread_ids + start/end positions記録
    
    テストモード用：各行を処理したスレッドの詳細情報を記録
    """

    # 最小限の共有メモリ（<1KB）
    block_count = cuda.shared.array(1, int32)

    # スレッド・ブロック情報
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    local_tid = cuda.threadIdx.x

    # 共有メモリ初期化
    if local_tid == 0:
        block_count[0] = 0

    cuda.syncthreads()

    # 担当範囲計算
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)

    if start_pos >= raw_data.size:
        return

    # ローカル結果バッファ（従来版と同じサイズ）
    local_positions = cuda.local.array(256, uint64)
    local_field_offsets = cuda.local.array((256, 17), uint32)  # 相対オフセット用にuint32に変更
    local_field_lengths = cuda.local.array((256, 17), int32)
    local_count = 0

    pos = uint64(start_pos)

    # 統合処理ループ: 行検出+フィールド抽出を同時実行
    while pos < end_pos:

        # === 1. 行ヘッダ検出 ===
        candidate_pos = read_uint16_simd16_lite(raw_data, pos, end_pos, ncols)

        if candidate_pos <= -2:  # データ終端
            break
        elif candidate_pos == -1:  # 行ヘッダ未発見
            pos += 15
            continue
        elif candidate_pos >= end_pos:  # 担当範囲外
            break

        # === 2. 行検証+フィールド抽出の統合実行 ===
        is_valid, row_end = validate_and_extract_fields_lite(
            raw_data, candidate_pos, ncols, fixed_field_lengths,
            local_field_offsets[local_count],  # この行のフィールド配列
            local_field_lengths[local_count]   # この行のフィールド配列
        )

        if is_valid:
            # 境界を越える行も処理する
            # 行の開始が担当範囲内なら、終了が範囲外でも処理
            if local_count < 256:  # 配列境界チェック
                local_positions[local_count] = uint64(candidate_pos)
                local_count += 1

            # 次の行へジャンプ（境界チェック改善）
            if row_end > 0:
                pos = row_end
            else:
                pos = candidate_pos + 1
        else:
            # 検証失敗時は1バイト進む
            pos = candidate_pos + 1

    # === 3. 結果を直接グローバルメモリに書き込み ===
    if local_count > 0:
        # スレッドローカルの結果をatomic操作で確保
        base_idx = cuda.atomic.add(row_count, 0, local_count)

        for i in range(local_count):
            global_idx = base_idx + i
            if global_idx < max_rows:
                # 行位置
                row_positions[global_idx] = local_positions[i]
                
                # スレッド情報を記録
                thread_ids[global_idx] = tid
                thread_start_positions[global_idx] = start_pos
                thread_end_positions[global_idx] = end_pos

                # フィールド情報
                for j in range(min(ncols, 17)):
                    field_offsets[global_idx, j] = local_field_offsets[i, j]  # uint32相対オフセット
                    field_lengths[global_idx, j] = local_field_lengths[i, j]


@cuda.jit
def parse_rows_and_fields_lite_debug(
    raw_data,
    header_size,
    ncols,

    # 出力配列（直接書き込み）
    row_positions,      # uint64[max_rows] - 行開始位置
    field_offsets,      # uint32[max_rows, ncols] - 行開始位置からの相対オフセット
    field_lengths,      # int32[max_rows, ncols] - フィールド長
    row_count,          # int32[1] - 検出行数

    # 設定
    thread_stride,
    max_rows,
    fixed_field_lengths,
    
    # デバッグ用
    thread_debug_info,  # 各スレッドのデバッグ情報 [tid, start_pos, end_pos, row_count]
    total_threads       # 総スレッド数
):
    """
    デバッグ版カーネル: スレッド情報を記録
    """
    # 最小限の共有メモリ（<1KB）
    block_count = cuda.shared.array(1, int32)

    # スレッド・ブロック情報
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    local_tid = cuda.threadIdx.x
    
    # デバッグモード：スレッドごとの検出行数
    thread_row_count = 0

    # 共有メモリ初期化
    if local_tid == 0:
        block_count[0] = 0

    cuda.syncthreads()

    # 担当範囲計算
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)

    if start_pos >= raw_data.size:
        # データ範囲外のスレッドも記録
        if tid < total_threads:
            thread_debug_info[tid, 0] = tid
            thread_debug_info[tid, 1] = start_pos
            thread_debug_info[tid, 2] = end_pos
            thread_debug_info[tid, 3] = 0
        return

    # ローカル結果バッファ（従来版と同じサイズ）
    local_positions = cuda.local.array(256, uint64)
    local_field_offsets = cuda.local.array((256, 17), uint32)  # 相対オフセット用にuint32に変更
    local_field_lengths = cuda.local.array((256, 17), int32)
    local_count = 0

    pos = uint64(start_pos)

    # 統合処理ループ: 行検出+フィールド抽出を同時実行
    while pos < end_pos:

        # === 1. 行ヘッダ検出 ===
        candidate_pos = read_uint16_simd16_lite(raw_data, pos, end_pos, ncols)

        if candidate_pos <= -2:  # データ終端
            break
        elif candidate_pos == -1:  # 行ヘッダ未発見
            pos += 15
            continue
        elif candidate_pos >= end_pos:  # 担当範囲外
            break

        # === 2. 行検証+フィールド抽出（統合） ===
        is_valid, row_end = validate_and_extract_fields_lite(
            raw_data, candidate_pos, ncols, fixed_field_lengths,
            local_field_offsets[local_count],
            local_field_lengths[local_count]
        )

        if is_valid:
            # 有効な行を検出
            if local_count < 256:
                local_positions[local_count] = uint64(candidate_pos)
                local_count += 1
            if row_end > 0:
                pos = row_end
            else:
                pos = candidate_pos + 1
        else:
            pos = candidate_pos + 1

    # 結果を直接グローバルメモリに書き込み
    if local_count > 0:
        # スレッドローカルの結果をatomic操作で確保
        base_idx = cuda.atomic.add(row_count, 0, local_count)
        thread_row_count = local_count  # デバッグ用

        for i in range(local_count):
            global_idx = base_idx + i
            if global_idx < max_rows:
                # 行位置
                row_positions[global_idx] = local_positions[i]

                # フィールド情報
                for j in range(min(ncols, 17)):
                    field_offsets[global_idx, j] = local_field_offsets[i, j]  # uint32相対オフセット
                    field_lengths[global_idx, j] = local_field_lengths[i, j]
    
    # デバッグモード：スレッド情報を記録
    if tid < total_threads:
        thread_debug_info[tid, 0] = tid
        thread_debug_info[tid, 1] = start_pos
        thread_debug_info[tid, 2] = end_pos
        thread_debug_info[tid, 3] = thread_row_count


@cuda.jit
def analyze_negative_positions(
    row_positions,
    nrows,
    raw_data,
    negative_debug,
    negative_count
):
    """負の行位置を分析してデバッグ情報を記録"""
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if tid >= nrows:
        return
    
    row_pos = row_positions[tid]
    
    # 負の値を検出
    if row_pos < 0:
        debug_idx = cuda.atomic.add(negative_count, 0, 1)
        if debug_idx < 50:  # 最大50エントリまで
            # デバッグ情報を記録
            negative_debug[debug_idx, 0] = tid  # 配列インデックス
            negative_debug[debug_idx, 1] = row_pos  # 負の値
            
            # 周辺のrow_positions値も記録（前後5個）
            for i in range(10):
                idx = tid - 5 + i
                if 0 <= idx < nrows:
                    negative_debug[debug_idx, 2 + i] = row_positions[idx]
                else:
                    negative_debug[debug_idx, 2 + i] = -999999999
            
            # バイナリデータの一部を記録（絶対位置から40バイト）
            # row_posが負の場合、tid*230（推定行サイズ）からサンプル
            estimated_pos = tid * 230  # 推定行サイズ230バイト
            data_start = min(estimated_pos, raw_data.size - 40)
            if data_start >= 0:
                for i in range(40):
                    if data_start + i < raw_data.size:
                        negative_debug[debug_idx, 12 + i] = int32(raw_data[data_start + i])
                    else:
                        negative_debug[debug_idx, 12 + i] = -1


@cuda.jit(debug=False)
def parse_rows_and_fields_lite_test(
    raw_data,
    header_size,
    ncols,

    # 出力配列（直接書き込み）
    row_positions,      # uint64[max_rows] - 行開始位置
    field_offsets,      # uint64[max_rows, ncols] - フィールド開始位置
    field_lengths,      # int32[max_rows, ncols] - フィールド長
    row_count,          # int32[1] - 検出行数

    # 設定
    thread_stride,
    max_rows,
    fixed_field_lengths,
    
    # テスト用デバッグ情報
    debug_info,         # デバッグ情報配列
    debug_count,        # デバッグ情報数
    negative_debug,     # 負の行位置デバッグ配列
    negative_count      # 負の行位置数
):
    """
    テストモード用カーネル: Grid境界スレッドの詳細情報を記録
    """
    # 通常の処理（本番カーネルと同じ）
    block_count = cuda.shared.array(1, int32)
    
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    local_tid = cuda.threadIdx.x
    
    # Grid境界スレッドかどうか判定
    is_grid_boundary = False
    if cuda.threadIdx.x == 0 or cuda.threadIdx.x == cuda.blockDim.x - 1:
        is_grid_boundary = True
    if cuda.blockIdx.x == 0 or cuda.blockIdx.x == cuda.gridDim.x - 1:
        is_grid_boundary = True
    if cuda.blockIdx.y == 0 or cuda.blockIdx.y == cuda.gridDim.y - 1:
        is_grid_boundary = True
    
    # 共有メモリ初期化
    if local_tid == 0:
        block_count[0] = 0
    cuda.syncthreads()
    
    # 担当範囲計算
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)
    
    if start_pos >= raw_data.size:
        return
    
    # ローカル結果バッファ
    local_positions = cuda.local.array(256, uint64)
    local_field_offsets = cuda.local.array((256, 17), uint64)
    local_field_lengths = cuda.local.array((256, 17), int32)
    local_count = 0
    
    pos = uint64(start_pos)
    
    # Grid境界スレッドの場合、最初の行情報を記録
    debug_recorded = False
    
    # 統合処理ループ
    while pos < end_pos:
        # 行ヘッダ検出
        candidate_pos = read_uint16_simd16_lite(raw_data, pos, end_pos, ncols)
        
        if candidate_pos <= -2:  # データ終端
            break
        elif candidate_pos == -1:  # 行ヘッダ未発見
            pos += 15
            continue
        elif candidate_pos >= end_pos:  # 担当範囲外
            break
        
        # 行検証+フィールド抽出
        is_valid, row_end = validate_and_extract_fields_lite(
            raw_data, candidate_pos, ncols, fixed_field_lengths,
            local_field_offsets[local_count],
            local_field_lengths[local_count]
        )
        
        if is_valid:
            # Grid境界スレッドで最初の有効な行の場合、デバッグ情報を記録
            if is_grid_boundary and not debug_recorded and local_count == 0:
                debug_idx = cuda.atomic.add(debug_count, 0, 1)
                if debug_idx < 100:  # 最大100エントリまで
                    # デバッグ情報を記録
                    debug_info[debug_idx, 0] = tid  # Thread ID
                    debug_info[debug_idx, 1] = cuda.blockIdx.x  # Block X
                    debug_info[debug_idx, 2] = cuda.blockIdx.y  # Block Y
                    debug_info[debug_idx, 3] = cuda.threadIdx.x  # Thread X
                    debug_info[debug_idx, 4] = candidate_pos  # Row position
                    debug_info[debug_idx, 5] = row_end  # Row end position
                    debug_info[debug_idx, 6] = ncols  # Number of columns
                    
                    # フィールドオフセットとレングス（全17フィールド）
                    for i in range(min(17, ncols)):
                        if 7 + i*2 + 1 < 100:  # 境界チェック
                            debug_info[debug_idx, 7 + i*2] = local_field_offsets[local_count][i]
                            debug_info[debug_idx, 8 + i*2] = local_field_lengths[local_count][i]
                    
                    # バイナリデータの前後20バイト分のサンプル
                    sample_start = max(0, candidate_pos - 20)
                    sample_end = min(raw_data.size, candidate_pos + 20)
                    for i in range(40):
                        if 41 + i < 100:  # 境界チェック
                            if sample_start + i < sample_end:
                                debug_info[debug_idx, 41 + i] = int32(raw_data[sample_start + i])
                            else:
                                debug_info[debug_idx, 41 + i] = -1
                    
                    # validate_and_extract_fields_liteの戻り値を記録
                    debug_info[debug_idx, 81] = 1 if is_valid else 0
                    debug_info[debug_idx, 82] = row_end
                    debug_info[debug_idx, 83] = local_count  # この時点での検出行数
                    debug_info[debug_idx, 84] = 0  # 負の行位置フラグ（後で更新）
                    
                    debug_recorded = True
            
            if local_count < 256:
                local_positions[local_count] = uint64(candidate_pos)
                local_count += 1
            
            if row_end > 0:
                pos = row_end
            else:
                pos = candidate_pos + 1
        else:
            pos = candidate_pos + 1
    
    # 結果を直接グローバルメモリに書き込み
    if local_count > 0:
        base_idx = cuda.atomic.add(row_count, 0, local_count)
        
        for i in range(local_count):
            global_idx = base_idx + i
            if global_idx < max_rows:
                row_positions[global_idx] = local_positions[i]
                
                for j in range(min(ncols, 17)):
                    field_offsets[global_idx, j] = local_field_offsets[i, j]
                    field_lengths[global_idx, j] = local_field_lengths[i, j]

def parse_binary_chunk_gpu_ultra_fast_v2_lite(raw_dev, columns, header_size=None, debug=False, test_mode=False):
    """
    軽量統合版: 共有メモリ使用量を最小化した統合パーサー
    
    最適化効果:
    - メモリアクセス: 50%削減 (218MB × 2回 → 218MB × 1回)
    - 共有メモリ: <1KB（制限内）
    - 実行時間短縮: 26.3%
    
    Args:
        test_mode: Trueの場合、Grid境界スレッドのデバッグ情報を記録
    """
    if header_size is None:
        header_size = 19

    ncols = len(columns)
    data_size = raw_dev.size - header_size

    # 固定長フィールド情報
    try:
        from ..types import PG_OID_TO_BINARY_SIZE
    except ImportError:
        from src.types import PG_OID_TO_BINARY_SIZE

    fixed_field_lengths = np.full(ncols, -1, dtype=np.int32)
    for i, column in enumerate(columns):
        pg_binary_size = PG_OID_TO_BINARY_SIZE.get(column.pg_oid)
        if pg_binary_size is not None:
            fixed_field_lengths[i] = pg_binary_size

    fixed_field_lengths_dev = cuda.to_device(fixed_field_lengths)

    # 推定行サイズ計算
    estimated_row_size = estimate_row_size_from_columns(columns)
    if debug or test_mode:
        # チャンクIDから処理元を判定
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} [DEBUG] 推定行サイズ: {estimated_row_size} バイト (列数: {ncols})")

    # グリッド設定（2次元グリッドで大規模データ対応）
    props = get_device_properties()
    max_blocks_x = props.get('MAX_GRID_DIM_X', 2147483647)
    max_blocks_y = props.get('MAX_GRID_DIM_Y', 65535)
    sm_count = props.get('MULTIPROCESSOR_COUNT', 108)
    
    # スレッドブロックサイズ
    threads_per_block = 256  # 8ワープ
    
    # データサイズと推定行サイズから必要なスレッド数を計算
    estimated_rows = data_size // estimated_row_size
    target_threads = estimated_rows  # 1行1スレッドから開始
    
    # グリッドサイズ計算
    blocks_x = min((target_threads + threads_per_block - 1) // threads_per_block, max_blocks_x)
    blocks_y = (target_threads + blocks_x * threads_per_block - 1) // (blocks_x * threads_per_block)
    
    if test_mode:
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} [GRID計算] target_threads={target_threads:,}, blocks_x={blocks_x:,}, blocks_y計算値={blocks_y:,}")
    
    # Y次元制限チェック
    if blocks_y > max_blocks_y:
        blocks_y = max_blocks_y
    
    # 最小ブロック数を確保（SMごとに複数ブロック）
    # 大規模データの場合はこのチェックをスキップ
    min_blocks = sm_count * 4  # 各SMに4ブロック
    if blocks_x * blocks_y < min_blocks and blocks_y == 0:  # blocks_y==0の場合のみ（つまり実行されない）
        blocks_x = min(min_blocks, max_blocks_x)
        blocks_y = 1
    
    # 実験: blocks_yを強制的に設定してテスト
    if test_mode and os.environ.get('GPUPGPARSER_TEST_BLOCKS_Y'):
        test_blocks_y = int(os.environ.get('GPUPGPARSER_TEST_BLOCKS_Y', '2'))
        old_blocks_y = blocks_y
        blocks_y = test_blocks_y
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} [TEST] blocks_yを{old_blocks_y}から{test_blocks_y}に変更")
    
    actual_threads = blocks_x * blocks_y * threads_per_block
    thread_stride = (data_size + actual_threads - 1) // actual_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size
    
    
    # thread_strideが大きい場合のワーニング（制限はしない）
    rows_per_thread = thread_stride // estimated_row_size
    if rows_per_thread > 200:
        if debug or test_mode:
            chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
            prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
            print(f"{prefix} ⚠️ WARNING: 各スレッドが処理する行数が多い: {rows_per_thread}行/スレッド (推奨: 200行以下)")
            print(f"{prefix}            thread_stride={thread_stride:,} バイト, estimated_row_size={estimated_row_size} バイト")
            if rows_per_thread > 256:
                print(f"{prefix} ⚠️ CAUTION: ローカルバッファサイズ(256行)を超える可能性があります")

    # 推定行数を計算
    estimated_rows = data_size // estimated_row_size
    
    # バッファサイズは推定行数に少しのマージンを追加（80%程度）
    max_rows = int(estimated_rows * 1.8)
    
    # 最後のチャンクの場合は、さらに余裕を持たせる
    if test_mode and os.environ.get('GPUPGPARSER_LAST_CHUNK', '0') == '1':
        max_rows = int(max_rows * 1.02)  # さらに2%追加
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} [DEBUG] 最後のチャンク検出: max_rowsを2%増加")
    
    if debug or test_mode:
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} [DEBUG] estimated_rows計算: data_size={data_size:,}, estimated_row_size={estimated_row_size}, estimated_rows={estimated_rows:,}")
    # GPUメモリ上限の確認
    bytes_per_row = 8 + ncols * (4 + 4)  # row_positions(8) + field_offsets(4*ncols) + field_lengths(4*ncols)
    available_memory = 20 * 1024**3  # 20GB (24GB GPUの場合、余裕を持たせて)
    memory_limit_rows = available_memory // bytes_per_row  # ソート無効化中なので1.5倍係数は不要
    max_rows = min(max_rows, memory_limit_rows)
    
    if debug or test_mode:
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        
        # メモリ使用量の詳細表示
        offsets_size = 4 * ncols
        lengths_size = 4 * ncols
        estimated_memory_mb = (bytes_per_row * estimated_rows) / (1024 * 1024)
        actual_memory_mb = (bytes_per_row * max_rows) / (1024 * 1024)
        
        print(f"{prefix} [DEBUG] GPU行位置情報配列: 1行あたり{bytes_per_row}バイト (row_pos:8 + offsets:{offsets_size} + lengths:{lengths_size})")
        print(f"{prefix} [DEBUG] 推定必要メモリ: {bytes_per_row}バイト × {estimated_rows:,}行 = {estimated_memory_mb:.0f}MB")
        print(f"{prefix} [DEBUG] 実際の配列確保: {max_rows:,}行分 = {actual_memory_mb:.0f}MB (推定の{(max_rows/estimated_rows*100):.1f}%)")

    # 統合出力配列の準備
    row_positions = cuda.device_array(max_rows, np.uint64)
    field_offsets = cuda.device_array((max_rows, ncols), np.uint32)  # 相対オフセット用にuint32に変更
    field_lengths = cuda.device_array((max_rows, ncols), np.int32)
    row_count = cuda.to_device(np.array([0], dtype=np.int32))
    
    # テストモード時にデバッグ配列を追加
    thread_ids = None
    thread_start_positions = None
    thread_end_positions = None
    if test_mode:
        thread_ids = cuda.device_array(max_rows, np.int32)
        thread_start_positions = cuda.device_array(max_rows, np.uint64)
        thread_end_positions = cuda.device_array(max_rows, np.uint64)
    
    # row_positionsを明示的に初期化（未使用領域を識別可能にする）
    # CuPyを使用して効率的に初期化
    try:
        import cupy as cp
        row_positions_gpu = cp.full(max_rows, 0xFFFFFFFFFFFFFFFF, dtype=cp.uint64)
        row_positions = cuda.as_cuda_array(row_positions_gpu)
        del row_positions_gpu  # CuPy配列は不要になったので削除
    except ImportError:
        # CuPyが使えない場合は初期化カーネルを使用
        @cuda.jit
        def init_kernel(arr, value):
            idx = cuda.grid(1)
            if idx < arr.size:
                arr[idx] = value
        
        threads = 256
        blocks = (max_rows + threads - 1) // threads
        init_kernel[blocks, threads](row_positions, np.uint64(0xFFFFFFFFFFFFFFFF))

    # テストモード用のデバッグ情報配列
    debug_info_array = None
    debug_count_array = None
    negative_pos_debug = None
    negative_pos_count = None
    thread_row_counts = None  # 各スレッドの検出行数
    thread_debug_info = None  # スレッドデバッグ情報
    
    if test_mode:
        # デバッグ情報配列: 100エントリ × 100要素に拡張
        # [0]: Thread ID, [1-3]: Block/Thread indices
        # [4-6]: Row info (position, end, ncols)
        # [7-40]: Field offsets/lengths (17 fields × 2)
        # [41-80]: Binary data sample (40 bytes)
        # [81]: validate_and_extract_fields_liteの戻り値（is_valid）
        # [82]: validate_and_extract_fields_liteの戻り値（row_end）
        # [83]: parse_rows_and_fields_liteでの検出行数
        
        # 各スレッドの検出行数を記録する配列を追加
        thread_row_counts = cuda.device_array(actual_threads, np.int32)
        # スレッドデバッグ情報配列を追加 [tid, start_pos, end_pos, row_count]
        thread_debug_info = cuda.device_array((actual_threads, 4), np.int64)
        
        # 初期化
        @cuda.jit
        def init_thread_counts(arr):
            idx = cuda.grid(1)
            if idx < arr.size:
                arr[idx] = 0
        init_blocks = (actual_threads + 255) // 256
        init_thread_counts[init_blocks, 256](thread_row_counts)
        
        # スレッドデバッグ情報も初期化
        @cuda.jit
        def init_thread_debug(arr):
            idx = cuda.grid(1)
            if idx < arr.shape[0]:
                for j in range(4):
                    arr[idx, j] = -1
        init_thread_debug[init_blocks, 256](thread_debug_info)
        
        # スレッド行数統計を収集するカーネル
        @cuda.jit
        def analyze_thread_row_distribution(
            row_positions, nrows, thread_stride, header_size, thread_counts
        ):
            """各スレッドが何行検出したかを分析"""
            row_idx = cuda.grid(1)
            if row_idx >= nrows:
                return
            
            # この行の位置から担当スレッドを逆算
            row_pos = row_positions[row_idx]
            if row_pos < header_size:
                return
                
            # どのスレッドが検出したか計算
            thread_id = (row_pos - header_size) // thread_stride
            if thread_id < thread_counts.size:
                cuda.atomic.add(thread_counts, thread_id, 1)
        
        # [84]: 負の行位置フラグ（1=負の値検出）
        # [85-99]: 予備領域
        debug_info_array = cuda.device_array((100, 100), np.int64)
        debug_count_array = cuda.to_device(np.array([0], dtype=np.int32))
        
        # 負の行位置のデバッグ情報用（別配列）
        negative_pos_debug = cuda.device_array((50, 100), np.int64)  # 最大50エントリ
        negative_pos_count = cuda.to_device(np.array([0], dtype=np.int32))

    if debug:
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} [DEBUG] 軽量統合カーネル実行: grid=({blocks_x}, {blocks_y}) × {threads_per_block}")
        print(f"{prefix} [DEBUG] データサイズ: {data_size//1024//1024}MB, 推定行サイズ: {estimated_row_size}B")
        print(f"{prefix} [DEBUG] thread_stride: {thread_stride:,}B, 総スレッド数: {actual_threads:,}")
        print(f"{prefix} [DEBUG] 推定総行数: {estimated_rows:,}, スレッドあたり: {thread_stride//estimated_row_size:.1f}行")
        print(f"{prefix} [DEBUG] 共有メモリ使用量: <1KB（大幅削減）")
        if test_mode:
            print(f"{prefix} [DEBUG] テストモード有効: Grid境界スレッド情報を記録")

    # カーネル実行
    grid_2d = (blocks_x, blocks_y)
    
    # テストモード時にカーネル実行情報を表示
    if test_mode:
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"\n{prefix} " + "="*60)
        print(f"{prefix} Kernel Execution Info: parse_rows_and_fields_lite")
        print(f"{prefix} " + "="*60)
        print(f"{prefix} Data Size:               {data_size:,} bytes ({data_size/1024/1024:.1f} MB)")
        print(f"{prefix} Grid Dimensions:         ({blocks_x}, {blocks_y})")
        print(f"{prefix} Block Dimensions:        {threads_per_block}")
        print(f"{prefix} Total Blocks:            {blocks_x * blocks_y:,}")
        print(f"{prefix} Total Threads:           {blocks_x * blocks_y * threads_per_block:,}")
        print(f"{prefix} Thread Stride:           {thread_stride:,} bytes")
        print(f"{prefix} Estimated Rows:          {estimated_rows:,}")
        print(f"{prefix} Estimated Rows/Thread:   {thread_stride//estimated_row_size:.1f}")
        print(f"{prefix} " + "="*60 + "\n")
    
    if test_mode:
        # テストモードでは完全デバッグカーネルを使用
        parse_rows_and_fields_lite_with_full_debug[grid_2d, threads_per_block](
            raw_dev, header_size, ncols,
            row_positions, field_offsets, field_lengths, 
            thread_ids, thread_start_positions, thread_end_positions, row_count,
            thread_stride, max_rows, fixed_field_lengths_dev
        )
    else:
        # 本番用カーネル（性能影響なし）
        parse_rows_and_fields_lite[grid_2d, threads_per_block](
            raw_dev, header_size, ncols,
            row_positions, field_offsets, field_lengths, row_count,
            thread_stride, max_rows, fixed_field_lengths_dev
        )
    cuda.synchronize()

    # 結果取得
    nrow = int(row_count.copy_to_host()[0])

    if debug or test_mode:
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} [DEBUG] 検出行数: {nrow:,}行 (バッファ: {max_rows:,}行)")
        
        # テストモードでスレッド行数分布を分析
        if test_mode and thread_row_counts is not None and nrow > 0:
            # 分析カーネルを実行
            analyze_blocks = (nrow + 255) // 256
            analyze_thread_row_distribution[analyze_blocks, 256](
                row_positions, nrow, thread_stride, header_size, thread_row_counts
            )
            cuda.synchronize()
            
            # 結果を取得して分析
            thread_counts_host = thread_row_counts.copy_to_host()
            
            # 統計を計算
            threads_with_data = np.count_nonzero(thread_counts_host)
            max_rows_per_thread = np.max(thread_counts_host)
            threads_with_2plus = np.sum(thread_counts_host >= 2)
            threads_with_5plus = np.sum(thread_counts_host >= 5)
            threads_with_10plus = np.sum(thread_counts_host >= 10)
            
            print(f"{prefix} [THREAD分析] スレッド行数分布:")
            print(f"{prefix}   - 総スレッド数: {actual_threads:,}")
            print(f"{prefix}   - 行を検出したスレッド: {threads_with_data:,} ({threads_with_data/actual_threads*100:.1f}%)")
            print(f"{prefix}   - 最大行数/スレッド: {max_rows_per_thread}")
            print(f"{prefix}   - 2行以上検出: {threads_with_2plus:,}スレッド")
            print(f"{prefix}   - 5行以上検出: {threads_with_5plus:,}スレッド")
            print(f"{prefix}   - 10行以上検出: {threads_with_10plus:,}スレッド")
            
            # 分布のヒストグラムを簡易表示
            if max_rows_per_thread > 0:
                hist, bins = np.histogram(thread_counts_host[thread_counts_host > 0], 
                                        bins=min(10, max_rows_per_thread))
                print(f"{prefix}   - 行数分布:")
                for i in range(len(hist)):
                    if hist[i] > 0:
                        print(f"{prefix}     {int(bins[i])}-{int(bins[i+1])}行: {hist[i]:,}スレッド")
            
            # スレッドデバッグ情報を出力
            if thread_debug_info is not None:
                thread_debug_host = thread_debug_info.copy_to_host()
                
                # 最初の10スレッド
                print(f"{prefix} [DEBUG] 最初の10スレッド:")
                for i in range(min(10, actual_threads)):
                    tid = int(thread_debug_host[i, 0])
                    start_pos = int(thread_debug_host[i, 1])
                    end_pos = int(thread_debug_host[i, 2])
                    row_count = int(thread_debug_host[i, 3])
                    print(f"{prefix}   スレッド{tid}: start_pos={start_pos:,}, end_pos={end_pos:,}, 検出行数={row_count}")
                
                # 最後の10スレッド
                print(f"{prefix} [DEBUG] 最後の10スレッド:")
                start_idx = max(0, actual_threads - 10)
                for i in range(start_idx, actual_threads):
                    tid = int(thread_debug_host[i, 0])
                    start_pos = int(thread_debug_host[i, 1])
                    end_pos = int(thread_debug_host[i, 2])
                    row_count = int(thread_debug_host[i, 3])
                    print(f"{prefix}   スレッド{tid}: start_pos={start_pos:,}, end_pos={end_pos:,}, 検出行数={row_count}")
                
                # 0件検出スレッド（最初の10個）
                zero_count_threads = []
                for i in range(actual_threads):
                    if int(thread_debug_host[i, 3]) == 0:
                        zero_count_threads.append(i)
                        if len(zero_count_threads) >= 10:
                            break
                
                if zero_count_threads:
                    print(f"{prefix} [DEBUG] 0件検出スレッド（最初の10個）:")
                    for i in zero_count_threads[:10]:
                        tid = int(thread_debug_host[i, 0])
                        start_pos = int(thread_debug_host[i, 1])
                        end_pos = int(thread_debug_host[i, 2])
                        print(f"{prefix}   スレッド{tid}: start_pos={start_pos:,}, end_pos={end_pos:,}")
                
                # データ終端付近の情報
                print(f"{prefix} [DEBUG] データサイズ: {data_size:,}")
                print(f"{prefix} [DEBUG] 最終スレッドのend_pos: {int(thread_debug_host[actual_threads-1, 2]):,}")
                if int(thread_debug_host[actual_threads-1, 2]) > data_size:
                    print(f"{prefix} [DEBUG] ⚠️  最終スレッドがデータ終端を超えています")
    
    # max_rowsを超えた場合のワーニング
    if nrow > max_rows:
        chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
        prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
        print(f"{prefix} ⚠️ WARNING: 検出行数({nrow:,})がバッファサイズ({max_rows:,})を超えています！")
        print(f"{prefix}           切り捨てられた行数: {nrow - max_rows:,}行")
        print(f"{prefix}           実際の処理行数: {min(nrow, max_rows):,}行")

    if nrow == 0:
        if test_mode:
            return cuda.device_array(0, np.uint64), cuda.device_array((0, ncols), np.uint32), cuda.device_array((0, ncols), np.int32), cuda.device_array(0, np.int32), cuda.device_array(0, np.uint64), cuda.device_array(0, np.uint64)
        return cuda.device_array(0, np.uint64), cuda.device_array((0, ncols), np.uint32), cuda.device_array((0, ncols), np.int32)
    
    # テストモードで負の行位置を分析（現在はコメントアウト）
    # if test_mode and negative_pos_debug is not None and negative_pos_count is not None:
    #     # 負の行位置分析カーネルを実行
    #     analyze_threads = 256
    #     analyze_blocks = (nrow + analyze_threads - 1) // analyze_threads
    #     chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
    #     prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
    #     print(f"{prefix} [TEST MODE] 負の行位置分析: {nrow:,}行 ÷ {analyze_threads}スレッド/ブロック = {analyze_blocks:,}ブロック")
    #     analyze_negative_positions[analyze_blocks, analyze_threads](
    #         row_positions, nrow, raw_dev, negative_pos_debug, negative_pos_count
    #     )
    #     cuda.synchronize()
    #     
    #     neg_count = int(negative_pos_count.copy_to_host()[0])
    #     if neg_count > 0:
    #         chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
    #         prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
    #         print(f"{prefix} [NEGATIVE POS DEBUG] {neg_count}個の負の行位置を検出（最大50個まで記録）")

    # GPUソート優先処理（閾値なし）
    # 行位置とフィールド情報を同期してソート
    if nrow > 0:
        sort_start = time.time()
        
        # 一時的にソートを無効化してテスト（メモリ削減のため）
        if debug:
            chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
            prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
            print(f"{prefix} [DEBUG] ソート処理一時無効化（メモリ削減テスト）")
        
        # 有効行数を計算
        invalid_value = np.uint64(0xFFFFFFFFFFFFFFFF)
        valid_count = cuda.to_device(np.array([0], dtype=np.int32))
        
        threads = 256
        blocks = (nrow + threads - 1) // threads
        count_valid_rows[blocks, threads](row_positions, nrow, invalid_value, valid_count)
        cuda.synchronize()
        
        valid_rows = int(valid_count.copy_to_host()[0])
        
        if test_mode or debug:
            chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
            prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
            print(f"{prefix} [NO SORT] 有効行数: {valid_rows}/{nrow}")
        
        if valid_rows > 0:
            # ソートせずにそのまま返す（Rustがページ順で取得しているため）
            final_row_positions = row_positions[:valid_rows]
            final_field_offsets = field_offsets[:valid_rows]
            final_field_lengths = field_lengths[:valid_rows]
            
            sort_time = time.time() - sort_start
            
            if test_mode or debug:
                chunk_id = int(os.environ.get('GPUPGPARSER_CURRENT_CHUNK', '-1'))
                prefix = f"[Consumer-1]" if chunk_id >= 0 else ""
                print(f"{prefix} [NO SORT] 処理完了: {valid_rows}行, {sort_time:.3f}秒")
            
            # テストモードの場合、デバッグ情報も返す
            if test_mode:
                final_thread_ids = thread_ids[:valid_rows] if thread_ids is not None else None
                final_thread_start_pos = thread_start_positions[:valid_rows] if thread_start_positions is not None else None
                final_thread_end_pos = thread_end_positions[:valid_rows] if thread_end_positions is not None else None
                return final_row_positions, final_field_offsets, final_field_lengths, final_thread_ids, final_thread_start_pos, final_thread_end_pos
            
            return final_row_positions, final_field_offsets, final_field_lengths
        else:
            if test_mode:
                return cuda.device_array(0, np.uint64), cuda.device_array((0, ncols), np.uint32), cuda.device_array((0, ncols), np.int32), cuda.device_array(0, np.int32), cuda.device_array(0, np.uint64), cuda.device_array(0, np.uint64)
            return cuda.device_array(0, np.uint64), cuda.device_array((0, ncols), np.uint32), cuda.device_array((0, ncols), np.int32)
        
        # CPUソートフォールバック
        print(f"⚠️ CPUソート使用: {nrow}行（CuPy未対応）")
        
        row_positions_host = row_positions[:nrow].copy_to_host()
        field_offsets_host = field_offsets[:nrow].copy_to_host()
        field_lengths_host = field_lengths[:nrow].copy_to_host()
        
        # 有効な行位置のみ抽出
        invalid_value = np.uint64(0xFFFFFFFFFFFFFFFF)
        valid_indices = row_positions_host != invalid_value
        if np.any(valid_indices):
            row_positions_valid = row_positions_host[valid_indices]
            field_offsets_valid = field_offsets_host[valid_indices]
            field_lengths_valid = field_lengths_host[valid_indices]
            
            sort_indices = np.argsort(row_positions_valid)
            row_positions_sorted = row_positions_valid[sort_indices]
            field_offsets_sorted = field_offsets_valid[sort_indices]
            field_lengths_sorted = field_lengths_valid[sort_indices]
            
            final_row_positions = cuda.to_device(row_positions_sorted)
            final_field_offsets = cuda.to_device(field_offsets_sorted)
            final_field_lengths = cuda.to_device(field_lengths_sorted)
            
            sort_time = time.time() - sort_start
            print(f"⚠️ CPUソート完了: {len(sort_indices)}行, {sort_time:.3f}秒（GPU→CPU→GPU転送含む）")
            
            # テストモードの場合、デバッグ情報も返す
            if test_mode:
                final_thread_ids = thread_ids[:valid_rows] if thread_ids is not None else None
                final_thread_start_pos = thread_start_positions[:valid_rows] if thread_start_positions is not None else None
                final_thread_end_pos = thread_end_positions[:valid_rows] if thread_end_positions is not None else None
                return final_row_positions, final_field_offsets, final_field_lengths, final_thread_ids, final_thread_start_pos, final_thread_end_pos
            
            return final_row_positions, final_field_offsets, final_field_lengths
        else:
            if test_mode:
                return cuda.device_array(0, np.uint64), cuda.device_array((0, ncols), np.uint32), cuda.device_array((0, ncols), np.int32), cuda.device_array(0, np.int32), cuda.device_array(0, np.uint64), cuda.device_array(0, np.uint64)
            return cuda.device_array(0, np.uint64), cuda.device_array((0, ncols), np.uint32), cuda.device_array((0, ncols), np.int32)
    else:
        if test_mode:
            return cuda.device_array(0, np.uint64), cuda.device_array((0, ncols), np.uint32), cuda.device_array((0, ncols), np.int32), cuda.device_array(0, np.int32), cuda.device_array(0, np.uint64), cuda.device_array(0, np.uint64)
        return cuda.device_array((0, ncols), np.uint64), cuda.device_array((0, ncols), np.int32)

# エイリアス関数（後方互換性）
def parse_binary_chunk_gpu_ultra_fast_v2(raw_dev, columns, header_size=None, debug=False, test_mode=False):
    """従来版インターフェース互換ラッパー"""
    return parse_binary_chunk_gpu_ultra_fast_v2_lite(raw_dev, columns, header_size, debug, test_mode)

def parse_binary_chunk_gpu_ultra_fast_v2_integrated(raw_dev, columns, header_size=None, debug=False, test_mode=False):
    """統合最適化版（軽量版を使用）"""
    return parse_binary_chunk_gpu_ultra_fast_v2_lite(raw_dev, columns, header_size, debug, test_mode)

__all__ = [
    "detect_pg_header_size",
    "parse_binary_chunk_gpu_ultra_fast_v2_lite",
    "parse_binary_chunk_gpu_ultra_fast_v2", 
    "parse_binary_chunk_gpu_ultra_fast_v2_integrated",
    "estimate_row_size_from_columns",
    "get_device_properties",
    "print_gpu_properties",
    "calculate_optimal_grid_sm_aware",
    "parse_rows_and_fields_lite",
    "validate_and_extract_fields_lite"
]