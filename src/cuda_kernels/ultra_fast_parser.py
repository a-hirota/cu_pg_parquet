"""
Ultra Fast PostgreSQL Binary Parser – coalescing‑fix v7
------------------------------------------------------
* warp 内 32 thread が 16 B ずつ連続ロード → 完全コアレッシング
* thread 0 が gap 埋め込み実施 → 未初期化バイト回避
* 各 thread は 512 B ストライド内で 16 B step の 32 回スライド判定 → 範囲澄底
* fallback: tile 終端は shared memory を線形スキャン
* 行ヘッダ検証 (2 B colCnt + 4 B len0 …) は raw_data を直接参照し誤検出ゼロ
* `debug=True` オプションで行長統計をダンプ
"""

from numba import cuda, int32, types
import numpy as np

# ───── パラメータ ─────
ROWS_PER_TILE       = 32
ALIGN_BYTES         = 16
WARP_STRIDE         = ROWS_PER_TILE * ALIGN_BYTES
TILE_BYTES          = WARP_STRIDE * 32
MAX_HITS_PER_THREAD = 512
MAX_FIELD_LEN       = 8_388_608

@cuda.jit(device=True, inline=True)
def read_uint16_simd8(buf, off, expected_cols):
    for i in range(8):
        p = off + i*2
        if p+1 < buf.size:
            v = (buf[p] << 8) | buf[p+1]
            if v == expected_cols:  # ピンポイント検出
                return v, i*2
    return 0xFFFF, -1

@cuda.jit(device=True, inline=True)
def load16(raw_data, gpos, shmem, sh_off):
    if sh_off + ALIGN_BYTES >= shmem.size or gpos + 15 >= raw_data.size:
        return False
    for i in range(ALIGN_BYTES):
        shmem[sh_off + i] = raw_data[gpos + i]
    return True

# row_cnt[0]: グローバルメモリ上の行カウント。atomic に加算し row_off[] の書き込み先を確保。
# row_off[]: 各行の開始オフセットを保持。次段階の列抽出カーネルで参照される。
@cuda.jit
def detect_rows(raw_data, header, row_off, row_cnt, ncols, first_col_size):
    sh = cuda.shared.array(TILE_BYTES, dtype=types.uint8)
    tid = int32(cuda.threadIdx.x)
    bid = int32(cuda.blockIdx.x)

    gbase = header + bid*TILE_BYTES
    gend  = gbase + TILE_BYTES

    # 安全な共有メモリロード（XORなし）
    for chunk in range(tid*ALIGN_BYTES, TILE_BYTES, WARP_STRIDE):
        load16(raw_data, gbase + chunk, sh, chunk)
    cuda.syncthreads()

    local_hits = cuda.local.array(MAX_HITS_PER_THREAD, types.int32)
    nhit = 0

    for stripe in range(0, WARP_STRIDE, ALIGN_BYTES):
        pos = gbase + tid*ALIGN_BYTES + stripe
        if pos + 6 >= gend:
            break
        sh_off = pos - gbase
        cnt, rel = read_uint16_simd8(sh, sh_off, ncols)
        if rel >= 0:
            cand = pos + rel
            if cand + 6 >= raw_data.size:
                continue
            if (raw_data[cand] << 8 | raw_data[cand+1]) == cnt:
                # 最初のフィールド長も検証
                if validate_first_field(raw_data, cand + 2, first_col_size):
                    good = True
                    tmp = cand + 2
                else:
                    continue  # 最初のフィールド長が不正なのでスキップ
                for _ in range(cnt):
                    if tmp + 3 >= raw_data.size:
                        good = False; break
                    flen = (
                        int32(raw_data[tmp  ]) << 24 | int32(raw_data[tmp+1]) << 16 |
                        int32(raw_data[tmp+2]) << 8  | int32(raw_data[tmp+3])
                    )
                    if flen == 0xFFFFFFFF:
                        tmp += 4
                    elif 0 <= flen <= MAX_FIELD_LEN:
                        tmp += 4 + flen
                    else:
                        good = False; break
                if good and nhit < MAX_HITS_PER_THREAD:
                    local_hits[nhit] = cand; nhit += 1

    if nhit:
        base = cuda.atomic.add(row_cnt, 0, nhit)
        for i in range(nhit):
            if base + i < row_off.size:
                row_off[base+i] = local_hits[i]

# ───── 列パース ─────
@cuda.jit
def extract_fields(raw, roff, foff, flen, nrow, ncol):
    rid = cuda.grid(1)
    if rid >= nrow:
        return
    pos = int32(roff[rid])
    nc = (raw[pos] << 8) | raw[pos+1]
    pos += 2
    for c in range(ncol):
        if c >= nc:
            flen[rid,c] = -1; foff[rid,c]=0; continue
        ln = (
            int32(raw[pos])<<24 | int32(raw[pos+1])<<16 |
            int32(raw[pos+2])<<8 | int32(raw[pos+3])
        )
        if ln == 0xFFFFFFFF:
            flen[rid,c] = -1; foff[rid,c]=0; pos += 4
        else:
            foff[rid,c] = pos+4; flen[rid,c] = ln; pos += 4+ln

# ───── 新★機能実装 ─────

def estimate_row_size_from_columns(columns):
    """要件0: ColumnMetaから行サイズ推定（★実装）"""
    try:
        from ..types import UTF8, DECIMAL128
    except ImportError:
        from src.types import UTF8, DECIMAL128
    
    size = 2  # フィールド数(2B)
    for col in columns:
        size += 4  # フィールド長(4B)
        if col.elem_size > 0:  # 固定長
            size += col.elem_size
        else:  # 可変長の推定
            if col.arrow_id == UTF8:
                size += 20  # 文字列平均20B
            elif col.arrow_id == DECIMAL128:
                size += 16   # NUMERIC平均16B
            else:
                size += 4   # その他4B
    
    # ★メモリバンクコンフリクト回避: 32B境界整列
    return ((size + 31) // 32) * 32

# @cuda.jit(device=True, inline=True)
# def read_uint16_simd16(raw_data, pos, end_pos,ncols):
#     """要件1: 16B読み込みで行ヘッダ探索（★高速化 + 0xFFFF終端検出）"""
#     if pos + 1 > raw_data.size:  # 最低2B必要
#         return -2
    
#     # 実際に読み込み可能な範囲を計算
#     max_offset = min(16, raw_data.size - pos + 1)
    
#     # 16Bを一度に読み込み、全ての2B位置をチェック（0-15B全範囲）
#     for i in range(0, max_offset):  # 安全な範囲内でスキャン
#         num_fields = (raw_data[pos + i] << 8) | raw_data[pos + i + 1]
#         if num_fields == ncols:
#             return pos + i
#     return -1

@cuda.jit(device=True, inline=True)
def read_uint16_simd16_in_thread(raw_data, pos, end_pos, ncols):
    """要件1: 16B読み込みで行ヘッダ探索（★高速化 + 0xFFFF終端検出）"""
    
    # 実際に読み込み可能な範囲を計算
    # max_offset = min(16, raw_data.size - pos + 1)

    if pos + 1 > raw_data.size:  # 最低2B必要(終端位置)
        return -2

    if pos + 1 > end_pos: # 最低2B必要(担当外)
        return -3
    
    min_end = min(end_pos, raw_data.size)    
    max_offset = min(16, min_end - pos + 1)
    
    # 16Bを一度に読み込み、全ての2B位置をチェック（0-15B全範囲）
    for i in range(0, max_offset):  # 安全な範囲内でスキャン
        num_fields = (raw_data[pos + i] << 8) | raw_data[pos + i + 1]        
        if num_fields == ncols:
            return pos + i
    return -1



@cuda.jit(device=True, inline=True)
def validate_complete_row_fast(raw_data, row_start, expected_cols, fixed_field_lengths):
    # raw_dataには、PostgreSQLのCOPY BINARYデータ全体が入ります。
    """要件2: 完全な行検証 + 越境処理対応 + ColumnMetaベース固定長フィールド検証"""
    if row_start + 2 > raw_data.size:
        return False, -1
    else:
        num_fields = (raw_data[row_start] << 8) | raw_data[row_start+1]

    if num_fields != expected_cols:
        return False, -2
    else:
        pos = row_start + 2
    
    
    for field_idx in range(num_fields):
        if pos + 3 > raw_data.size:
            return False, -3
        else:
            flen = (
                int32(raw_data[pos  ]) << 24 | int32(raw_data[pos+1]) << 16 |
                int32(raw_data[pos+2]) << 8  | int32(raw_data[pos+3])
            )

        if flen != 0xFFFFFFFF and  ( flen < 0 or flen > 1000000): # 異常な値を排除
            return False, -4
        
        # ★ColumnMetaベースの固定長フィールド検証（偽ヘッダ排除の強化）
        if field_idx < len(fixed_field_lengths) and fixed_field_lengths[field_idx] > 0: 
            expected_len = fixed_field_lengths[field_idx]
            if flen != expected_len:
                return False, -5 
        
        if pos + 4 + flen > raw_data.size:
            return False, -6        
        else:
            pos = pos + 4  + flen
    
    # 次行ヘッダー検証（偽ヘッダー排除）
    if pos + 1 > raw_data.size:
        return False, -7 
    else:
        next_header = (raw_data[pos] << 8) | raw_data[pos + 1]

    if next_header != expected_cols and next_header != 0xFFFF:
        return False, -8  # 次行ヘッダーが不正
    return True, pos  # (検証成功, 行終端位置)

@cuda.jit
def detect_rows_optimized(raw_data, header_size, thread_stride, estimated_row_size, ncols,
                           row_positions, row_count, max_rows, fixed_field_lengths, debug_array=None):
    """★初期化徹底版: 未初期化メモリ問題を完全解決 + whileループ終了原因トラッキング"""
    
    # ★共有メモリ拡張版（GPU制限: 48KB/ブロック）
    # lineorder worst-case: 512スレッド × 平均231B = 180KB範囲で最大1000行程度
    # 安全マージンを追加して2048行対応（8KB使用、まだ十分安全）
    MAX_SHARED_ROWS = 2048  # 8KB使用（worst-case対応 + 安全マージン）
    block_positions = cuda.shared.array(MAX_SHARED_ROWS, int32)
    block_count = cuda.shared.array(1, int32)
    
    # スレッドID計算
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    local_tid = cuda.threadIdx.x
    
    # ★安全な初期化：ブロック内共有メモリ
    if local_tid == 0:
        block_count[0] = 0
        # スレッド0が全て初期化（安全確実）
        for i in range(MAX_SHARED_ROWS):
            block_positions[i] = -1
    
    cuda.syncthreads()  # ★重要: 初期化完了を保証
    
    # 各スレッドの担当範囲計算（オーバーラップなし）
    start_pos = header_size + tid * thread_stride
    end_pos = header_size + (tid + 1) * thread_stride
    search_end = min(end_pos+thread_stride, raw_data.size - 1)
    
    # Debug: record thread boundaries
    if debug_array is not None:
        total_threads = cuda.gridDim.x * cuda.gridDim.y * cuda.blockDim.x
        if tid == 0:
            debug_array[0] = start_pos
            debug_array[1] = end_pos
        elif tid == total_threads - 1:
            debug_array[2] = start_pos
            debug_array[3] = end_pos
    
    if start_pos >= raw_data.size:
        return
    
    # ★完全初期化：ローカル結果保存用
    local_positions = cuda.local.array(256, int32)
    # ★Numba制約対応：ローカル配列は使用時に初期化
    local_count = 0
    pos = start_pos
    loop_iterations = 0
    exit_reason = 0  # 0:未設定, 1:正常終了(pos>=end_pos), 2:終端マーカー, 3:候補位置が担当外, 4:検証失敗で担当外, 5:row_end >= end_pos
    
    # ★高速行検出ループ（詳細動作トラッキング付き）
    # 見逃し担当スレッドの詳細ログ用配列（特定スレッドのみ）
    target_thread_82496 = (tid == 82496)
    target_thread_529584 = (tid == 529584)
    is_target_thread = target_thread_82496 or target_thread_529584
    
    while pos < end_pos:
        loop_iterations += 1
        old_pos = pos  # ★ループ開始時の位置を記録
        
        candidate_pos = read_uint16_simd16_in_thread(raw_data, pos, end_pos,ncols)
        
        if candidate_pos <= -2: # 終端マーカー検出
            exit_reason = 2
            break  # データ終端

        if candidate_pos == -1: # 行ヘッダ候補なし
            pos += 15  # 15Bステップ（元の設計に戻す）
            # ★特定スレッドの詳細ログ：15Bステップ
            if is_target_thread and debug_array is not None and len(debug_array) > 1000:
                step_log_base = 1000 + loop_iterations * 4
                if step_log_base + 3 < len(debug_array):
                    debug_array[step_log_base] = old_pos         # ステップ前位置
                    debug_array[step_log_base + 1] = pos         # ステップ後位置
                    debug_array[step_log_base + 2] = -1          # candidate_pos = -1
                    debug_array[step_log_base + 3] = 15          # 15Bステップ実行
            continue
        
        if candidate_pos >= 0: # 有効な行ヘッダ候補あり
            if candidate_pos >= end_pos: # 最近傍が担当外のためloop終了
                exit_reason = 3
                # ★特定スレッドの詳細ログ：候補位置が担当外
                if is_target_thread and debug_array is not None and len(debug_array) > 1000:
                    exit_log_base = 1000 + loop_iterations * 4
                    if exit_log_base + 3 < len(debug_array):
                        debug_array[exit_log_base] = old_pos         # 終了時位置
                        debug_array[exit_log_base + 1] = candidate_pos  # 担当外候補位置
                        debug_array[exit_log_base + 2] = end_pos        # end_pos
                        debug_array[exit_log_base + 3] = 3              # exit_reason = 3
                break
            
            # 完全行検証
            is_valid, row_end = validate_complete_row_fast(raw_data, candidate_pos, ncols, fixed_field_lengths)
            if not is_valid:
                if candidate_pos + 1 < end_pos:
                    pos = candidate_pos + 1
                    # ★特定スレッドの詳細ログ：検証失敗、+1で継続
                    if is_target_thread and debug_array is not None and len(debug_array) > 1000:
                        fail_log_base = 1000 + loop_iterations * 4
                        if fail_log_base + 3 < len(debug_array):
                            debug_array[fail_log_base] = candidate_pos      # 検証失敗位置
                            debug_array[fail_log_base + 1] = pos            # +1後位置
                            debug_array[fail_log_base + 2] = row_end        # row_end値
                            debug_array[fail_log_base + 3] = -99            # 検証失敗マーカー
                    continue
                else: # 最近傍+1も担当外のためloop終了
                    exit_reason = 4
                    # ★特定スレッドの詳細ログ：検証失敗で担当外
                    if is_target_thread and debug_array is not None and len(debug_array) > 1000:
                        exit_log_base = 1000 + loop_iterations * 4
                        if exit_log_base + 3 < len(debug_array):
                            debug_array[exit_log_base] = candidate_pos      # 検証失敗位置
                            debug_array[exit_log_base + 1] = candidate_pos + 1  # +1予定位置
                            debug_array[exit_log_base + 2] = end_pos            # end_pos
                            debug_array[exit_log_base + 3] = 4                  # exit_reason = 4
                    break
            
            # 検証成功 → ローカル保存（安全性チェック付き）
            local_positions[local_count] = candidate_pos
            local_count += 1
            
            # ★最後のrow_end値を記録
            last_row_end = row_end
            
            # 次の行開始位置へジャンプ（境界条件改善）
            if row_end > 0 and row_end < end_pos:
                pos = row_end
                # ★特定スレッドの詳細ログ：検証成功、row_endジャンプ
                if is_target_thread and debug_array is not None and len(debug_array) > 1000:
                    success_log_base = 1000 + loop_iterations * 4
                    if success_log_base + 3 < len(debug_array):
                        debug_array[success_log_base] = candidate_pos       # 検証成功位置
                        debug_array[success_log_base + 1] = row_end         # ジャンプ先
                        debug_array[success_log_base + 2] = local_count     # 検出カウント
                        debug_array[success_log_base + 3] = 99              # 検証成功マーカー
                continue
            else: # row_endより手前は行ヘッダなし。row_end以降は担当外のため
                exit_reason = 5
                # ★特定スレッドの詳細ログ：row_end >= end_pos
                if is_target_thread and debug_array is not None and len(debug_array) > 1000:
                    exit_log_base = 1000 + loop_iterations * 4
                    if exit_log_base + 3 < len(debug_array):
                        debug_array[exit_log_base] = candidate_pos          # 検証成功位置
                        debug_array[exit_log_base + 1] = row_end            # row_end値
                        debug_array[exit_log_base + 2] = end_pos            # end_pos値
                        debug_array[exit_log_base + 3] = 5                  # exit_reason = 5
                break
    
    # 正常終了の場合
    if pos >= end_pos and exit_reason == 0:
        exit_reason = 1
        # ★特定スレッドの詳細ログ：正常終了
        if is_target_thread and debug_array is not None and len(debug_array) > 1000:
            exit_log_base = 1000 + loop_iterations * 4
            if exit_log_base + 3 < len(debug_array):
                debug_array[exit_log_base] = pos                # 終了位置
                debug_array[exit_log_base + 1] = end_pos        # end_pos値
                debug_array[exit_log_base + 2] = loop_iterations # ループ回数
                debug_array[exit_log_base + 3] = 1              # exit_reason = 1
    
    # ★最後の行検証時のrow_end値を記録（見逃し分析用）
    last_row_end = -1
    
    # ★デバッグ記録: ループ終了原因とスレッド情報（グローバルメモリ直接書き込み方式）
    if debug_array is not None:
        # グローバルメモリに全スレッド情報を直接書き込み（確実な全スレッド対応）
        # debug_array構造: 各スレッドが20要素ずつ使用 [tid*20:tid*20+19]（row_end追加）
        thread_base = tid * 20
        
        if thread_base + 19 < len(debug_array):
            debug_array[thread_base] = tid
            debug_array[thread_base + 1] = exit_reason
            debug_array[thread_base + 2] = pos  # 終了時のpos
            debug_array[thread_base + 3] = local_count
            # ★拡張デバッグ情報
            debug_array[thread_base + 4] = start_pos
            debug_array[thread_base + 5] = end_pos
            debug_array[thread_base + 6] = search_end
            debug_array[thread_base + 7] = thread_stride
            # ★row_end情報
            debug_array[thread_base + 8] = last_row_end  # 最後の行検証時のrow_end
            debug_array[thread_base + 9] = loop_iterations  # ループ回数
            debug_array[thread_base + 10] = 0  # 予約
            debug_array[thread_base + 11] = 0  # 予約
            
            # ★バイトダンプ記録: 全スレッドのpos周辺32バイトを記録
            if pos >= 32 and pos + 32 < raw_data.size:
                # pos-16からpos+15までの32バイトを8つの4バイト整数として記録
                dump_start = pos - 16
                for i in range(8):
                    byte_group = 0
                    for j in range(4):
                        if dump_start + i * 4 + j < raw_data.size:
                            byte_value = raw_data[dump_start + i * 4 + j]
                            byte_group |= (byte_value << (j * 8))
                    debug_array[thread_base + 12 + i] = byte_group
            else:
                # バイトダンプ不可能な場合は0で初期化
                for i in range(8):
                    debug_array[thread_base + 12 + i] = 0
        

    
    # ★ブロック単位協調処理（オーバーフロー対応版）
    if local_count > 0:
        # 共有メモリ容量を確認
        local_base_idx = cuda.atomic.add(block_count, 0, local_count)
        
        # ★オーバーフロー分は直接グローバル配列へ書き込み
        for i in range(local_count):
            shared_idx = local_base_idx + i
            if shared_idx < MAX_SHARED_ROWS and local_positions[i] >= 0:
                # 共有メモリに収まる場合
                block_positions[shared_idx] = local_positions[i]
            elif local_positions[i] >= 0:
                # ★オーバーフロー分を直接グローバルへ（欠落防止）
                global_idx = cuda.atomic.add(row_count, 0, 1)
                if global_idx < max_rows:
                    row_positions[global_idx] = local_positions[i]
    
    cuda.syncthreads()  # ★必要最小限の同期
    
    # ★スレッド0による一括グローバル書き込み（初期化徹底版）
    if local_tid == 0 and block_count[0] > 0:
        # 共有メモリに保存された分のみ処理
        actual_rows = min(block_count[0], MAX_SHARED_ROWS)
        if actual_rows > 0:
            global_base_idx = cuda.atomic.add(row_count, 0, actual_rows)
            
            # ★安全な連続書き込み：有効値のみ書き込み
            for i in range(actual_rows):
                if (global_base_idx + i < max_rows and
                    block_positions[i] >= 0 and  # 有効性チェック
                    block_positions[i] < raw_data.size):  # 範囲チェック
                    row_positions[global_base_idx + i] = block_positions[i]

def get_device_properties():
    """GPU デバイス特性を取得"""
    device = cuda.get_current_device()
    return {
        'MULTIPROCESSOR_COUNT': device.MULTIPROCESSOR_COUNT,
        'MAX_THREADS_PER_BLOCK': device.MAX_THREADS_PER_BLOCK,
        'MAX_GRID_DIM_X': device.MAX_GRID_DIM_X,
        'MAX_GRID_DIM_Y': device.MAX_GRID_DIM_Y,
    }

def calculate_optimal_grid_sm_aware(data_size, estimated_row_size):
    """SMコア数を考慮した最適なグリッドサイズ＋threads_per_block計算 - 完全動的最適化"""
    props = get_device_properties()
    
    sm_count = props.get('MULTIPROCESSOR_COUNT', 108)  # デバイス固有のSM数
    max_threads_per_block = props.get('MAX_THREADS_PER_BLOCK', 1024)
    max_blocks_x = props.get('MAX_GRID_DIM_X', 65535)
    max_blocks_y = props.get('MAX_GRID_DIM_Y', 65535)
    
    data_mb = data_size / (1024 * 1024)
    
    # ★動的threads_per_block決定
    if data_mb < 50:
        # 小データ: 精度重視、中程度の並列度
        threads_per_block = 256
    elif data_mb < 200:
        # 中データ: バランス重視
        threads_per_block = 512
    else:
        # 大データ: スループット重視、最大並列度
        threads_per_block = min(1024, max_threads_per_block)
    
    # データサイズベースの基本必要ブロック数
    num_threads = (data_size + estimated_row_size - 1) // estimated_row_size
    base_blocks = (num_threads + threads_per_block - 1) // threads_per_block
    
    # ★データサイズ閾値: 50MB未満は従来方式、50MB以上でSM最適化適用
    if data_mb < 50:
        # 小〜中データ: 従来のデータベース計算（精度重視）
        target_blocks = min(base_blocks, sm_count * 2)  # 適度なSM活用
    else:
        # ★大データのみSM最適化適用（メモリバウンドタスク向け）
        # キャッシュ効率とメモリ帯域を重視した適応的ブロック配置
        if data_mb < 200:
            # 中〜大データ: メモリバウンド最適範囲（4-6ブロック/SM）
            target_blocks = max(sm_count * 4, min(base_blocks, sm_count * 6))
        else:
            # 超大データ: データ並列性とメモリ帯域のバランス（6-12ブロック/SM）
            target_blocks = max(sm_count * 6, min(base_blocks, sm_count * 12))
    
    # 最低限のSM活用保証（全SM使用）
    target_blocks = max(target_blocks, sm_count)  # 最低1ブロック/SM
    
    # 2次元グリッド配置
    blocks_x = min(target_blocks, max_blocks_x)
    blocks_y = min((target_blocks + blocks_x - 1) // blocks_x, max_blocks_y)
    
    return blocks_x, blocks_y, threads_per_block

def parse_binary_chunk_gpu_ultra_fast_v2(raw_dev, columns, header_size: int = None, traditional_positions=None, *, debug: bool=False):
    """★完全実装版: 初期化徹底 + 単一パス高精度検出 + 動的見逃し位置検出"""
    if header_size is None:
        header_size = 19
    
    ncols = len(columns)
    
    # ★ColumnMetaから固定長フィールド情報を動的抽出（PostgreSQLバイナリサイズ使用）
    try:
        from ..types import PG_OID_TO_BINARY_SIZE
    except ImportError:
        from src.types import PG_OID_TO_BINARY_SIZE
    
    fixed_field_lengths = np.full(ncols, -1, dtype=np.int32)  # -1は可変長
    for i, column in enumerate(columns):
        # PostgreSQLバイナリでの実際のサイズを取得
        pg_binary_size = PG_OID_TO_BINARY_SIZE.get(column.pg_oid)
        if pg_binary_size is not None:  # PostgreSQLバイナリで固定長
            fixed_field_lengths[i] = pg_binary_size
    
    # GPU用固定長配列に転送
    fixed_field_lengths_dev = cuda.to_device(fixed_field_lengths)
    
    # 要件0: 推定行サイズ計算（★実装）
    estimated_row_size = estimate_row_size_from_columns(columns)
    
    # データサイズ
    data_size = raw_dev.size - header_size
    
    # ★SM対応グリッド配置（動的最適化）
    blocks_x, blocks_y, optimal_threads_per_block = calculate_optimal_grid_sm_aware(
        data_size, estimated_row_size
    )
    threads_per_block = optimal_threads_per_block
    
    # ★実際に起動するスレッド数でthread_strideを計算（100%カバレッジ保証）
    actual_threads = blocks_x * blocks_y * threads_per_block
    thread_stride = (data_size + actual_threads - 1) // actual_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size  # 最小でも推定行サイズ
    
    # SM対応最適化後の実際の値
    num_threads = actual_threads
    total_blocks = blocks_x * blocks_y
    
    # GPU特性情報の取得（debugモードに関係なく必要）
    props = get_device_properties()
    sm_count = props.get('MULTIPROCESSOR_COUNT', 'Unknown')
    max_threads = props.get('MAX_THREADS_PER_BLOCK', 'Unknown')
    max_blocks_per_dim = props.get('MAX_GRID_DIM_X', 65535)
    
    # ★詳細デバッグ: SM対応グリッド + 完全カバレッジ分析 + 固定長フィールド情報
    if debug:
        print(f"[DEBUG] ★Ultra Fast v8 (初期化徹底版): ncols={ncols}, estimated_row_size={estimated_row_size}")
        print(f"[DEBUG] ★GPU特性: SM数={sm_count}, 最大スレッド/ブロック={max_threads}")
        print(f"[DEBUG] ★data_size={data_size//1024//1024}MB ({data_size}B)")
        fixed_count = sum(1 for x in fixed_field_lengths if x > 0)
        fixed_info = [f'{i}:{x}B' for i, x in enumerate(fixed_field_lengths) if x > 0]
        print(f"[DEBUG] ★固定長フィールド検証: {fixed_count}/{ncols}個 ({fixed_info})")
        print(f"[DEBUG] ★設計値: threads={num_threads}, total_blocks={total_blocks}")
        print(f"[DEBUG] ★SM対応グリッド: ({blocks_x}, {blocks_y}) × {threads_per_block}threads")
        print(f"[DEBUG] ★実際のブロック数: {blocks_x * blocks_y} (SM効率: {(blocks_x * blocks_y) / sm_count:.1f}ブロック/SM)")
        print(f"[DEBUG] ★thread_stride={thread_stride}B (密度: {data_size//num_threads}B/thread)")
        
        # カバレッジ計算（実際のカバー範囲を正確に表示）
        coverage_bytes = min(num_threads * thread_stride, data_size)
        coverage_gap = max(0, data_size - coverage_bytes)
        coverage_ratio = coverage_bytes / data_size
        print(f"[DEBUG] ★期待カバー範囲: {coverage_bytes//1024//1024}MB ({coverage_bytes}B)")
        print(f"[DEBUG] ★カバレッジ: {coverage_ratio*100:.3f}% (不足: {coverage_gap}B)")
        
        # GPU制限チェック（SM対応最適化後は基本的に制限なし）
        max_possible_threads = max_blocks_per_dim * max_blocks_per_dim * threads_per_block
        if num_threads >= max_possible_threads:
            print(f"[DEBUG] ★GPU制限に到達: {num_threads}スレッド")
        if total_blocks >= max_blocks_per_dim:
            print(f"[DEBUG] ★ブロック制限に到達: {total_blocks}ブロック")
    
    # ★積和方式リトライ：デバイス配列準備
    max_rows = min(2_000_000, (data_size // estimated_row_size) * 2)
    target_rows = 1_000_000  # 目標行数
    max_attempts = 2  # デバッグ用: 2回実行（1回目で見逃し特定、2回目で詳細取得）
    
    if debug:
        print(f"[DEBUG] ★2回ループデバッグ: 1回目で見逃し特定、2回目で詳細取得")
    
    # 累積結果保存用（CPU側）
    all_positions = set()  # 重複除去のためのセット
    first_attempt_missing_positions = []  # 1回目で検出した見逃し位置
    
    # デバッグ配列サイズを事前定義（row_end追加対応）
    # 全スレッド基本情報: 658432 × 20 = 13,168,640要素（row_end等4要素追加）
    debug_info_size = num_threads * 20  # 全スレッド分（row_end、ループ回数等込み）
    
    for attempt in range(max_attempts):
        if debug:
            if attempt == 0:
                print(f"[DEBUG] ★試行 {attempt + 1}/{max_attempts} 開始... (見逃し位置特定)")
            else:
                print(f"[DEBUG] ★試行 {attempt + 1}/{max_attempts} 開始... (特定スレッド詳細取得)")
        
        # ★試行毎に非決定論的要素を導入：未初期化GPUメモリを活用
        if attempt == 0:
            # 1回目：完全初期化（基準結果）
            row_positions_host = np.full(max_rows, -1, dtype=np.int32)
            row_positions = cuda.to_device(row_positions_host)
            row_count_host = np.zeros(1, dtype=np.int32)
            row_count = cuda.to_device(row_count_host)
            debug_array_host = np.full(debug_info_size, -1, dtype=np.int32)
            debug_array = cuda.to_device(debug_array_host)
        else:
            # 2回目以降：部分的に未初期化メモリを利用して非決定論的動作を誘発
            row_positions = cuda.device_array(max_rows, np.int32)  # 未初期化
            row_count = cuda.device_array(1, np.int32)  # 未初期化
            debug_array = cuda.device_array(debug_info_size, np.int32)  # 未初期化
            
            # 最低限のゼロクリア（動作保証のため）
            row_count[0] = 0
        
        # ★カーネル実行前の確実な同期
        cuda.synchronize()
        
        # ★実行（同じパラメータだが、未初期化メモリの影響で異なる結果期待）
        grid_2d = (blocks_x, blocks_y)
        detect_rows_optimized[grid_2d, threads_per_block](
            raw_dev, header_size, thread_stride, estimated_row_size, ncols,
            row_positions, row_count, max_rows, fixed_field_lengths_dev, debug_array
        )
        cuda.synchronize()
        
        # ★結果取得
        nrow = int(row_count.copy_to_host()[0])
        new_positions_count = 0
        if nrow > 0:
            positions = row_positions[:nrow].copy_to_host()
            # 有効な位置のみ追加（-1を除外）
            valid_positions = positions[positions >= 0]
            before_count = len(all_positions)
            all_positions.update(valid_positions)
            new_positions_count = len(all_positions) - before_count
        
        # 1回目終了時: 見逃し位置を動的検出
        if attempt == 0 and traditional_positions is not None and debug:
            traditional_positions_set = set(traditional_positions)
            ultra_fast_positions_set = set(valid_positions)
            missing_positions_set = traditional_positions_set - ultra_fast_positions_set
            first_attempt_missing_positions = sorted(list(missing_positions_set))[:10]
            
            init_method = "見逃し特定"
            print(f"[DEBUG] ★試行 {attempt + 1}({init_method}): {nrow}行検出, 見逃し: {len(missing_positions_set)}行")
            if first_attempt_missing_positions:
                print(f"[DEBUG] ★1回目で特定した見逃し位置: {first_attempt_missing_positions[:5]}")
        elif debug:
            init_method = "詳細取得" if attempt > 0 else "見逃し特定"
            print(f"[DEBUG] ★試行 {attempt + 1}({init_method}): {nrow}行検出, 新規追加: {new_positions_count}行, 累積ユニーク: {len(all_positions)}行")
        
        # 目標達成チェック
        if len(all_positions) >= target_rows:
            if debug:
                print(f"[DEBUG] ★目標達成！ {len(all_positions)}行 >= {target_rows}行")
            break
    
    # ★最終結果をソートしてGPUに転送
    if len(all_positions) > 0:
        row_offsets_host = np.sort(np.array(list(all_positions), dtype=np.int32))
        # 目標行数にトリミング
        if len(row_offsets_host) > target_rows:
            row_offsets_host = row_offsets_host[:target_rows]
        row_offsets = cuda.to_device(row_offsets_host)
        nrow = len(row_offsets_host)
    else:
        row_offsets = cuda.device_array(0, np.int32)
        row_offsets_host = np.array([], dtype=np.int32)
        nrow = 0
    
    if debug:
        print(f"[DEBUG] ★積和方式完了: {attempt + 1}試行で{nrow}行達成")
    
    # ★初期化徹底版のデバッグ結果解析
    if debug:
        debug_results = debug_array.copy_to_host()
        
        print(f"[DEBUG] ★初期化徹底版デバッグ結果:")
        print(f"[DEBUG] ★検出行数: {nrow}行")
        
        # 検証成功位置の次ヘッダ分析
        success_count = debug_results[14] if debug_results[14] >= 0 else 0
        if success_count > 0:
            print(f"[DEBUG_SUCCESS] 検証成功位置 {success_count}件の次ヘッダ分析（実際のバイト値）:")
            invalid_next_header_count = 0
            raw_host_data = raw_dev.copy_to_host()
            
            for i in range(min(success_count, 5)):  # 表示数を削減（5個まで）
                idx = i * 2
                position = debug_results[idx]
                recorded_value = debug_results[idx + 1]  # GPU記録値（位置情報の可能性）
                
                # ★実際のバイトダンプで次ヘッダの値を確認
                print(f"\n  位置 {position}: GPU記録値={recorded_value}")
                
                # 行開始位置から実際の行終了位置を計算してバイトダンプ
                if position >= 0 and position + 200 < len(raw_host_data):
                    # 現在行のヘッダ値を確認
                    current_header = (raw_host_data[position] << 8) | raw_host_data[position + 1]
                    print(f"    現在行ヘッダ値: {current_header} (17フィールド期待)")
                    
                    # 行開始位置周辺をダンプ
                    start_dump = max(0, position - 8)
                    end_dump = min(len(raw_host_data), position + 200)
                    chunk = raw_host_data[start_dump:end_dump]
                    
                    hex_dump = ' '.join(f'{b:02x}' for b in chunk[:32])  # 最初の32バイト
                    print(f"    行開始付近: {hex_dump}")
                    
                    # GPU記録値が実際の次ヘッダ値かチェック
                    if recorded_value == 17:
                        print(f"    ★GPU記録: 次ヘッダ値=17 (正常)")
                    elif recorded_value == -1:
                        print(f"    ★GPU記録: 境界外 (次ヘッダ値=-1)")
                    elif recorded_value == 99999:
                        print(f"    ★GPU記録: 範囲超過エラー (次ヘッダ値=99999)")
                    else:
                        print(f"    ★GPU記録: 異常値 次ヘッダ値={recorded_value}")
                        invalid_next_header_count += 1
                else:
                    print(f"    ★位置{position}が範囲外")
            
            print(f"\n  ★不正な次ヘッダを持つ成功位置: {invalid_next_header_count}/{success_count}")
        
        # ★複数検証失敗位置の詳細分析（6要素構造対応）
        fail_count = debug_results[39] if debug_results[39] >= 0 else 0
        if fail_count > 0:
            print(f"[DEBUG_FALSE] 検証失敗位置 {fail_count}件の詳細分析:")
            
            for i in range(min(fail_count, 10)):  # 最大10件表示
                base_idx = 40 + i * 6  # 6要素構造に修正
                if base_idx + 4 < len(debug_results):
                    position = debug_results[base_idx]
                    error_code = debug_results[base_idx + 1]  # row_endがエラーコード
                    thread_id = debug_results[base_idx + 2]
                    next_header = debug_results[base_idx + 3]
                    fail_marker = debug_results[base_idx + 4]
                    
                    if fail_marker == 999:  # 有効な失敗記録
                        print(f"\n  失敗#{i+1} - 位置 {position} (Thread {thread_id}):")
                        print(f"    validate_complete_row_fast戻り値: False")
                        
                        # エラーコード（row_end）による原因分析
                        print(f"    ★検証失敗原因: {error_code}")
                        
                        # 次ヘッダ値の詳細
                        if next_header >= 0:
                            print(f"    次ヘッダ値: {next_header} (期待値: 17 or 0xFFFF)")
                        else:
                            print(f"    次ヘッダ値: 読み取り不可 ({next_header})")
            
            print(f"\n  ★検証失敗総数: {fail_count}件")
        
        # ★従来の単一失敗記録も表示（互換性維持）
        elif debug_results[38] == 999:  # 検証失敗マーカー確認
            position = debug_results[35]
            row_end = debug_results[36]
            next_header = debug_results[37]
            
            print(f"[DEBUG_FALSE] 検証失敗位置 {position} 詳細分析 (従来形式):")
            print(f"  validate_complete_row_fast戻り値: False")
            print(f"  行終了位置: {row_end}")
            
            if next_header == 99999:
                print(f"  ★検証失敗原因: validate_complete_row_fast関数の行終了位置計算エラー")
            elif next_header == -1:
                print(f"  ★検証失敗原因: row_start + 2 > raw_data.size")
            elif next_header == -2:
                print(f"  ★検証失敗原因: num_fields != expected_cols")
            elif next_header == -3:
                print(f"  ★検証失敗原因: pos + 4 > raw_data.size")
            elif next_header == -4:
                print(f"  ★検証失敗原因: flen <= 0:  # 0xFFFFFFFF以外の負値は不正")
            elif next_header == -5:
                print(f"  ★検証失敗原因: flen > 1000000:  # 異常に大きな値")
            elif next_header == -6:
                print(f"  ★検証失敗原因: flen != expected_len")
            elif next_header == -7:
                print(f"  ★検証失敗原因: pos + flen > raw_data.size")
            elif next_header == -8:
                print(f"  ★検証失敗原因: データ境界での次ヘッダ読み取り不可")
            else:
                print(f"  GPU記録の次ヘッダ値: {next_header} (期待値: 17)")
                if next_header not in [17, 0xFFFF]:
                    print(f"  ★検証失敗原因: 不正な次ヘッダ値({next_header})")
                else:
                    print(f"  ★検証失敗原因: フィールド検証で失敗（次ヘッダ値は正常）")
        
        # ★新機能: whileループ終了原因の分析（全スレッド対応 - グローバルメモリ直接読み取り）
        print(f"[DEBUG] ★Whileループ終了原因分析（全スレッド対応 - グローバルメモリ直接方式）:")
        
        # グローバルメモリから全スレッド情報を収集
        exit_reason_stats = {}
        thread_info = {}  # 辞書形式（スレッドID -> 情報）
        missing_threads = []  # 見逃し担当スレッドの詳細情報
        
        # 全スレッドを解析（20要素構造対応）
        valid_records = 0
        for tid in range(num_threads):
            thread_base = tid * 20
            if thread_base + 19 < len(debug_results):
                thread_id = debug_results[thread_base]
                exit_reason = debug_results[thread_base + 1]
                final_pos = debug_results[thread_base + 2]
                detected_rows = debug_results[thread_base + 3]
                # ★拡張デバッグ情報
                start_pos_debug = debug_results[thread_base + 4]
                end_pos_debug = debug_results[thread_base + 5]
                search_end_debug = debug_results[thread_base + 6]
                thread_stride_debug = debug_results[thread_base + 7]
                # ★row_end情報
                last_row_end_debug = debug_results[thread_base + 8]
                loop_iterations_debug = debug_results[thread_base + 9]
                # ★バイトダンプ情報
                byte_dump = []
                for i in range(8):
                    byte_dump.append(debug_results[thread_base + 12 + i])
                
                # 有効なスレッド情報かチェック（thread_idが自分自身と一致）
                if thread_id == tid:
                    valid_records += 1
                    if exit_reason not in exit_reason_stats:
                        exit_reason_stats[exit_reason] = 0
                    exit_reason_stats[exit_reason] += 1
                    
                    thread_info[thread_id] = {
                        'tid': thread_id,
                        'reason': exit_reason,
                        'final_pos': final_pos,
                        'detected_rows': detected_rows,
                        # ★拡張デバッグ情報
                        'start_pos': start_pos_debug,
                        'end_pos': end_pos_debug,
                        'search_end': search_end_debug,
                        'thread_stride': thread_stride_debug,
                        # ★row_end情報
                        'last_row_end': last_row_end_debug,
                        'loop_iterations': loop_iterations_debug,
                        # ★バイトダンプ情報
                        'byte_dump': byte_dump
                    }
        
        print(f"  有効な記録数: {valid_records}/{num_threads}")
        
        # 終了原因の説明
        reason_names = {
            0: "未設定（異常）",
            1: "正常終了(pos>=end_pos)",
            2: "終端マーカー検出",
            3: "候補位置が担当外",
            4: "検証失敗で担当外",
            5: "row_end >= end_pos"
        }
        
        print(f"  終了原因統計:")
        for reason, count in sorted(exit_reason_stats.items()):
            reason_name = reason_names.get(reason, f"未知({reason})")
            print(f"    {reason_name}: {count}スレッド")
        
        # ★見逃し担当スレッドの特定と詳細分析
        # 1回目で特定した見逃し位置を活用
        if first_attempt_missing_positions:
            known_missing_positions = first_attempt_missing_positions
            if debug:
                print(f"[DEBUG] ★1回目で特定した見逃し位置を活用: {known_missing_positions}")
        elif traditional_positions is not None:
            # 最終結果での見逃し位置検出
            traditional_positions_set = set(traditional_positions)
            final_ultra_fast_positions = list(all_positions)
            ultra_fast_positions_set = set(final_ultra_fast_positions)
            missing_positions_set = traditional_positions_set - ultra_fast_positions_set
            known_missing_positions = sorted(list(missing_positions_set))[:10]
            if debug:
                print(f"[DEBUG] ★最終結果から見逃し位置を検出: {len(missing_positions_set)}行見逃し")
        else:
            # フォールバック: 従来版位置なしの場合
            known_missing_positions = []
            if debug:
                print(f"[DEBUG] ★従来版位置が提供されていません")
        
        print(f"\n  ★見逃し担当スレッドの詳細分析:")
        for missing_pos in known_missing_positions:
            # 担当スレッドID計算
            responsible_tid = (missing_pos - header_size) // thread_stride
            
            print(f"\n    見逃し位置 {missing_pos}:")
            print(f"      計算された担当Thread: {responsible_tid}")
            
            # グローバルメモリから該当スレッドの情報を取得
            if responsible_tid in thread_info:
                info = thread_info[responsible_tid]
                reason_name = reason_names.get(info['reason'], f"未知({info['reason']})")
                
                print(f"      ★Thread {responsible_tid}の終了原因: {reason_name}")
                print(f"      ★実際の値検証:")
                print(f"        start_pos (計算値): {header_size + responsible_tid * thread_stride}")
                print(f"        start_pos (実測値): {info['start_pos']}")
                print(f"        end_pos   (計算値): {header_size + (responsible_tid + 1) * thread_stride}")
                print(f"        end_pos   (実測値): {info['end_pos']}")
                print(f"        search_end        : {info['search_end']}")
                print(f"        thread_stride     : {info['thread_stride']}")
                print(f"        終了位置 (final_pos): {info['final_pos']}")
                print(f"        最後のrow_end     : {info['last_row_end']}")
                print(f"        ループ回数        : {info['loop_iterations']}")
                print(f"        検出行数          : {info['detected_rows']}")
                
                # 実測値での範囲判定
                actual_start = info['start_pos']
                actual_end = info['end_pos']
                actual_search_end = info['search_end']
                
                print(f"      ★実測範囲: {actual_start}-{actual_end} (search_end={actual_search_end})")
                
                # ★バイトダンプ解析（見逃し担当スレッドのみ）
                if responsible_tid in [82496, 529584]:
                    print(f"      ★バイトダンプ解析 (pos={info['final_pos']}周辺32バイト):")
                    byte_dump = info['byte_dump']
                    hex_bytes = []
                    ascii_chars = []
                    
                    for i, int_value in enumerate(byte_dump):
                        if int_value != 0:  # 有効なデータがある場合
                            for j in range(4):
                                byte_val = (int_value >> (j * 8)) & 0xFF
                                hex_bytes.append(f"{byte_val:02x}")
                                ascii_chars.append(chr(byte_val) if 32 <= byte_val <= 126 else '.')
                    
                    if hex_bytes:
                        # 16バイトずつ2行で表示
                        print(f"        HEX: {' '.join(hex_bytes[:16])}")
                        print(f"             {' '.join(hex_bytes[16:32])}")
                        print(f"        ASCII: {''.join(ascii_chars[:16])}")
                        print(f"               {''.join(ascii_chars[16:32])}")
                        
                        # 見逃し位置と終了位置の関係分析
                        dump_start = info['final_pos'] - 16
                        missing_offset = missing_pos - dump_start
                        print(f"        見逃し位置オフセット: {missing_offset} (dump_start={dump_start})")
                        if 0 <= missing_offset < 32:
                            print(f"        ★見逃し位置がダンプ範囲内！オフセット{missing_offset}番目")
                
                # 位置が担当範囲内かチェック（実測値使用）
                if actual_start <= missing_pos < actual_end:
                    print(f"      ★範囲判定: 担当範囲内（見逃し確定）")
                    
                    # 15Bステップでの捕捉可能性分析
                    step_number = (missing_pos - actual_start) // 15
                    step_pos = actual_start + step_number * 15
                    read_end = step_pos + 15
                    
                    print(f"      15Bステップ進行: step#{step_number}, 読み込み範囲 {step_pos}-{read_end}")
                    if step_pos <= missing_pos <= read_end:
                        print(f"      ★捕捉判定: 理論上検出可能")
                        print(f"      ★根本原因詳細分析:")
                        print(f"        終了条件: pos < end_pos ({info['final_pos']} < {actual_end})")
                        print(f"        終了条件: pos < search_end ({info['final_pos']} < {actual_search_end})")
                        print(f"        row_end >= end_pos 検証: final_pos >= end_pos = {info['final_pos']} >= {actual_end} = {info['final_pos'] >= actual_end}")
                        print(f"        last_row_end >= end_pos検証: {info['last_row_end']} >= {actual_end} = {info['last_row_end'] >= actual_end}")
                        print(f"        ★真の問題: {reason_name}により早期終了で位置{missing_pos}に未到達")
                    else:
                        print(f"      ★捕捉判定: 15Bステップの隙間に位置")
                elif actual_start <= missing_pos < actual_search_end:
                    print(f"      ★範囲判定: search_end範囲内（オーバーラップ領域）")
                else:
                    print(f"      ★範囲判定: 担当範囲外（境界問題）")
                
                missing_threads.append(info)
            else:
                print(f"      ❌ Thread {responsible_tid}の情報が記録されていません")
                print(f"         (有効記録: {valid_records}スレッド/{num_threads}スレッド)")
        
        # 詳細分析: 検出行数が0のスレッドの原因
        zero_detection_threads = [info for info in thread_info.values() if info['detected_rows'] == 0]
        if zero_detection_threads:
            print(f"\n  検出行数0のスレッド詳細（{len(zero_detection_threads)}件）:")
            reason_distribution = {}
            for t in zero_detection_threads[:10]:  # 最初の10件表示
                reason_name = reason_names.get(t['reason'], f"未知({t['reason']})")
                if reason_name not in reason_distribution:
                    reason_distribution[reason_name] = 0
                reason_distribution[reason_name] += 1
                
                # スレッドの担当範囲計算
                start_pos = header_size + t['tid'] * thread_stride
                end_pos = header_size + (t['tid'] + 1) * thread_stride
                coverage = end_pos - start_pos
                
                print(f"    Thread {t['tid']}: {reason_name}")
                print(f"      担当範囲: {start_pos}-{end_pos} ({coverage}B)")
                print(f"      終了位置: {t['final_pos']}")
            
            print(f"    検出失敗の主因:")
            for reason, count in sorted(reason_distribution.items(), key=lambda x: x[1], reverse=True):
                print(f"      {reason}: {count}スレッド")
        
        # ★グローバルメモリ記録の有効性確認
        total_collected = len(thread_info)
        expected_threads = num_threads
        coverage_ratio = total_collected / expected_threads
        print(f"\n  ★グローバルメモリ収集率: {total_collected}/{expected_threads} ({coverage_ratio*100:.1f}%)")
        if coverage_ratio < 0.9:
            print(f"  ⚠️ 収集率が低い: グローバルメモリ書き込みに問題がある可能性")
    
    print(f"[DEBUG] ★Ultra Fast v8 (初期化徹底版) detected {nrow} rows")
    
    if nrow == 0:
        print("[DEBUG] No rows detected")
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
    
    # 詳細デバッグ情報
    if debug and nrow > 1:
        gaps = np.diff(row_offsets_host)
        print(f"[DEBUG] ★初期化徹底版結果: rows {nrow} min {gaps.min()} max {gaps.max()} avg {float(gaps.mean()):.2f}")
        small = (gaps < 8).sum()
        print(f"[DEBUG] ★small_rows: {small} (目標: 0)")
        
        # 検出率分析
        expected_rows = 1_000_000
        detection_rate = nrow / expected_rows * 100
        missing_rows = expected_rows - nrow
        print(f"[DEBUG] ★期待: {expected_rows}, 検出: {nrow}, 検出率: {detection_rate:.1f}%")
        print(f"[DEBUG] ★不足: {missing_rows}行 ({missing_rows/expected_rows*100:.1f}%)")
        
        # データ分布分析
        data_start = row_offsets_host[0]
        data_end = row_offsets_host[-1]
        data_span = data_end - data_start
        print(f"[DEBUG] ★データ分布: {data_start}-{data_end} (範囲: {data_span//1024//1024}MB)")
        
        # ★境界分析：未初期化メモリ問題のチェック
        boundary_16 = row_offsets_host % 16
        boundary_32 = row_offsets_host % 32
        boundary_352 = row_offsets_host % 352  # 推定行サイズ
        
        print(f"[DEBUG] ★境界分析（初期化問題チェック）:")
        print(f"  16B境界: {np.bincount(boundary_16, minlength=16)}")
        print(f"  32B境界: {np.bincount(boundary_32, minlength=32)}")
        print(f"  推定行サイズ境界: 偏り = {np.std(boundary_352):.2f}")
    
    # ★安全性チェック：フィールド抽出前の検証
    if debug:
        print(f"[DEBUG] ★フィールド抽出準備: nrow={nrow}, ncols={ncols}")
        if nrow > 0:
            # 行位置の有効性チェック
            positions_check = row_offsets_host[:min(5, nrow)]
            print(f"[DEBUG] ★行位置サンプル: {positions_check}")
    
    # フィールド抽出（安全性強化版）
    if nrow > 0:
        # ★フィールド配列の確実な初期化
        field_offsets_host = np.zeros((nrow, ncols), dtype=np.int32)  # CPU側で初期化
        field_lengths_host = np.full((nrow, ncols), -1, dtype=np.int32)  # CPU側で初期化
        field_offsets = cuda.to_device(field_offsets_host)  # GPU転送
        field_lengths = cuda.to_device(field_lengths_host)  # GPU転送
        
        # 安全な抽出ブロック数計算
        extract_blocks = min((nrow + threads_per_block - 1) // threads_per_block, max_blocks_per_dim)
        if debug:
            print(f"[DEBUG] ★フィールド抽出: {extract_blocks}ブロック × {threads_per_block}スレッド")
            print(f"[DEBUG] ★フィールド配列初期化: numpy完了")
        
        try:
            extract_fields[extract_blocks, threads_per_block](
                raw_dev, row_offsets, field_offsets, field_lengths, nrow, ncols
            )
            cuda.synchronize()
        except Exception as e:
            print(f"[DEBUG] ★フィールド抽出エラー: {e}")
            # エラー時は空配列を返す
            return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
        
        return field_offsets, field_lengths
    else:
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)

# ───── 下位互換ホストドライバ ─────
def parse_binary_chunk_gpu_ultra_fast(raw_dev, ncols: int, header_size: int = None, first_col_size: int = None, *, debug: bool=False):
    """下位互換ラッパー（従来インターフェース対応）"""
    # 簡易的なcolumns作成
    try:
        from ..types import ColumnMeta, INT32
    except ImportError:
        from src.types import ColumnMeta, INT32
    
    columns = [ColumnMeta(name=f"col_{i}", pg_oid=23, pg_typmod=0,
                         arrow_id=INT32, elem_size=4) for i in range(ncols)]
    
    return parse_binary_chunk_gpu_ultra_fast_v2(raw_dev, columns, header_size, debug=debug)

__all__ = [
    "parse_binary_chunk_gpu_ultra_fast",
    "parse_binary_chunk_gpu_ultra_fast_v2",
    "estimate_row_size_from_columns",
    "get_device_properties",
    "calculate_optimal_grid_sm_aware"
]
