"""
軽量統合パーサー: 共有メモリ使用量を最小化
===========================================

最適化ポイント:
- 共有メモリを最小限に抑制（<5KB）
- 直接グローバルメモリ書き込み
- atomic操作で競合回避
- フィールド情報の即座保存

メモリ効率:
- 重複読み込み排除: 218MB × 2回 → 218MB × 1回
- 共有メモリ使用量: <5KB（制限内）
"""

from numba import cuda, int32, types
import numpy as np
import time  # ソート時間計測用

@cuda.jit(device=True, inline=True)
def read_uint16_simd16_lite(raw_data, pos, end_pos, ncols):
    """行ヘッダ検出（軽量版）"""
    if pos + 1 > raw_data.size:
        return -2
   
    max_offset = min(16, raw_data.size - pos + 1)
    
    for i in range(0, max_offset):
        if pos + i + 1 > end_pos:
            return -3    
        num_fields = (raw_data[pos + i] << 8) | raw_data[pos + i + 1]        
        if num_fields == ncols:
            return pos + i
    return -1

@cuda.jit(device=True, inline=True)
def validate_and_extract_fields_lite(
    raw_data, row_start, expected_cols, fixed_field_lengths,
    field_offsets_out,  # この行のフィールド開始位置配列
    field_lengths_out   # この行のフィールド長配列
):
    """
    行検証とフィールド抽出を同時実行（軽量版）
    """
    if row_start + 2 > raw_data.size:
        return False, -1

    # フィールド数確認
    num_fields = (raw_data[row_start] << 8) | raw_data[row_start + 1]
    if num_fields != expected_cols:
        return False, -2

    pos = row_start + 2

    # **全フィールドを検証+抽出**
    for field_idx in range(num_fields):
        if pos + 4 > raw_data.size:
            return False, -3

        # フィールド長読み取り
        flen = (
            int32(raw_data[pos]) << 24 | int32(raw_data[pos+1]) << 16 |
            int32(raw_data[pos+2]) << 8 | int32(raw_data[pos+3])
        )

        # **フィールド情報を即座に出力配列に記録**
        if flen == 0xFFFFFFFF:  # NULL
            field_offsets_out[field_idx] = 0
            field_lengths_out[field_idx] = -1
            pos += 4
        else:
            # 異常値チェック
            if flen < 0 or flen > 1000000:
                return False, -4

            # 固定長フィールド検証
            if (field_idx < len(fixed_field_lengths) and
                fixed_field_lengths[field_idx] > 0 and
                flen != fixed_field_lengths[field_idx]):
                return False, -5

            # 境界チェック（データ終端まで許可）
            if pos + 4 + flen > raw_data.size:
                return False, -6

            # **フィールド情報記録**
            field_offsets_out[field_idx] = pos + 4  # データ開始位置
            field_lengths_out[field_idx] = flen      # データ長
            pos += 4 + flen

    # 次行ヘッダ検証
    if pos + 2 <= raw_data.size:
        next_header = (raw_data[pos] << 8) | raw_data[pos + 1]
        if next_header != expected_cols and next_header != 0xFFFF:
            return False, -8

    return True, pos  # 成功、行終端位置を返す

@cuda.jit
def parse_rows_and_fields_lite(
    raw_data,
    header_size,
    ncols,

    # 出力配列（直接書き込み）
    row_positions,      # int32[max_rows] - 行開始位置
    field_offsets,      # int32[max_rows, ncols] - フィールド開始位置
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
    start_pos = header_size + tid * thread_stride
    end_pos = header_size + (tid + 1) * thread_stride

    if start_pos >= raw_data.size:
        return

    # ローカル結果バッファ（従来版と同じサイズ）
    local_positions = cuda.local.array(256, int32)  # 従来版と一致
    local_field_offsets = cuda.local.array((256, 17), int32)
    local_field_lengths = cuda.local.array((256, 17), int32)
    local_count = 0

    pos = start_pos

    # **統合処理ループ: 行検出+フィールド抽出を同時実行**
    while pos < end_pos:  # 従来版と同じ条件（制限なし）

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
            # **重要修正: 境界を越える行も処理する**
            # 行の開始が担当範囲内なら、終了が範囲外でも処理
            if local_count < 256:  # 配列境界チェック
                local_positions[local_count] = candidate_pos
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
                    field_offsets[global_idx, j] = local_field_offsets[i, j]
                    field_lengths[global_idx, j] = local_field_lengths[i, j]

def parse_binary_chunk_gpu_ultra_fast_v2_lite(raw_dev, columns, header_size=None, debug=False):
    """
    軽量統合版: 共有メモリ使用量を最小化した統合パーサー
    
    最適化効果:
    - メモリアクセス: 50%削減 (218MB × 2回 → 218MB × 1回)
    - 共有メモリ: <1KB（制限内）
    - 実行時間短縮予想: 33%
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
    try:
        from .postgresql_binary_parser import estimate_row_size_from_columns, calculate_optimal_grid_sm_aware
    except ImportError:
        from postgresql_binary_parser import estimate_row_size_from_columns, calculate_optimal_grid_sm_aware

    estimated_row_size = estimate_row_size_from_columns(columns)

    # グリッド設定
    blocks_x, blocks_y, threads_per_block = calculate_optimal_grid_sm_aware(
        data_size, estimated_row_size
    )

    actual_threads = blocks_x * blocks_y * threads_per_block
    thread_stride = (data_size + actual_threads - 1) // actual_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size

    max_rows = min(2_000_000, (data_size // estimated_row_size) * 2)

    # **統合出力配列の準備**
    row_positions = cuda.device_array(max_rows, np.int32)
    field_offsets = cuda.device_array((max_rows, ncols), np.int32)
    field_lengths = cuda.device_array((max_rows, ncols), np.int32)
    row_count = cuda.device_array(1, np.int32)
    row_count[0] = 0

    if debug:
        print(f"[DEBUG] 軽量統合カーネル実行: grid=({blocks_x}, {blocks_y}) × {threads_per_block}")
        print(f"[DEBUG] データサイズ: {data_size//1024//1024}MB, 推定行サイズ: {estimated_row_size}B")
        print(f"[DEBUG] 共有メモリ使用量: <1KB（大幅削減）")

    # **軽量統合カーネル実行（1回のみ）**
    grid_2d = (blocks_x, blocks_y)
    parse_rows_and_fields_lite[grid_2d, threads_per_block](
        raw_dev, header_size, ncols,
        row_positions, field_offsets, field_lengths, row_count,
        thread_stride, max_rows, fixed_field_lengths_dev
    )
    cuda.synchronize()

    # 結果取得
    nrow = int(row_count.copy_to_host()[0])

    if debug:
        print(f"[DEBUG] 軽量統合処理完了: {nrow}行 (メモリ読み込み1回のみ)")
        print(f"[DEBUG] メモリ効率: 50%向上（重複アクセス排除）")

    if nrow == 0:
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)

    # GPUソート優先処理（闾値廃止）
    # 行位置とフィールド情報を同期してソート
    if nrow > 0:
        sort_start = time.time()
        
        # GPUソートを優先的に使用
        try:
            import cupy as cp
            
            if debug:
                print(f"[DEBUG] GPUソート開始: {nrow}行")
            
            # GPU上で直接ソート処理（転送不要）
            row_positions_gpu = cp.asarray(row_positions[:nrow])
            field_offsets_gpu = cp.asarray(field_offsets[:nrow])
            field_lengths_gpu = cp.asarray(field_lengths[:nrow])
            
            # 有効な行位置のみ抽出（GPU上）
            valid_mask = row_positions_gpu >= 0
            if cp.any(valid_mask):
                row_positions_valid = row_positions_gpu[valid_mask]
                field_offsets_valid = field_offsets_gpu[valid_mask]
                field_lengths_valid = field_lengths_gpu[valid_mask]
                
                # GPU上で高速ソート
                sort_indices = cp.argsort(row_positions_valid)
                field_offsets_sorted = field_offsets_valid[sort_indices]
                field_lengths_sorted = field_lengths_valid[sort_indices]
                
                # CuPy配列をNumba CUDA配列に変換
                final_field_offsets = cuda.as_cuda_array(field_offsets_sorted)
                final_field_lengths = cuda.as_cuda_array(field_lengths_sorted)
                
                sort_time = time.time() - sort_start
                print(f"✅ GPUソート完了: {len(sort_indices)}行, {sort_time:.3f}秒")
                
                return final_field_offsets, final_field_lengths
            else:
                return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
                
        except ImportError:
            # CuPy未対応時はCPUソートにフォールバック
            if debug:
                print("[DEBUG] CuPy未対応環境、CPUソートを使用")
        
        # CPUソートフォールバック
        print(f"⚠️ CPUソート使用: {nrow}行（CuPy未対応）")
        
        row_positions_host = row_positions[:nrow].copy_to_host()
        field_offsets_host = field_offsets[:nrow].copy_to_host()
        field_lengths_host = field_lengths[:nrow].copy_to_host()
        
        valid_indices = row_positions_host >= 0
        if np.any(valid_indices):
            row_positions_valid = row_positions_host[valid_indices]
            field_offsets_valid = field_offsets_host[valid_indices]
            field_lengths_valid = field_lengths_host[valid_indices]
            
            sort_indices = np.argsort(row_positions_valid)
            field_offsets_sorted = field_offsets_valid[sort_indices]
            field_lengths_sorted = field_lengths_valid[sort_indices]
            
            final_field_offsets = cuda.to_device(field_offsets_sorted)
            final_field_lengths = cuda.to_device(field_lengths_sorted)
            
            sort_time = time.time() - sort_start
            print(f"⚠️ CPUソート完了: {len(sort_indices)}行, {sort_time:.3f}秒（GPU→CPU→GPU転送含む）")
            
            return final_field_offsets, final_field_lengths
        else:
            return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
                
    else:
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)

__all__ = [
    "parse_binary_chunk_gpu_ultra_fast_v2_lite",
    "parse_rows_and_fields_lite",
    "validate_and_extract_fields_lite"
]