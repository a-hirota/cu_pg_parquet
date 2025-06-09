"""
Ultra Fast PostgreSQL Binary Parser – whileループ終了原因トラッキング版
----------------------------------------------------------------------
* detect_rows_optimizedにwhileループ終了原因の詳細トラッキング機能を追加
* 未検出部分の原因分析とスレッド別デバッグ情報出力
* 見逃し位置の担当スレッド特定と境界問題の診断
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
def read_uint16_simd16_in_thread(raw_data, pos, end_pos, ncols):
    """要件1: 16B読み込みで行ヘッダ探索（★高速化 + 0xFFFF終端検出）"""
    
    # 実際に読み込み可能な範囲を計算
    min_end = min(end_pos, raw_data.size)
    
    if pos + 1 > min_end: # 最低2B必要
        return -2
    
    max_offset = min(16, min_end - pos + 1)
    
    # 16Bを一度に読み込み、全ての2B位置をチェック（0-15B全範囲）
    for i in range(0, max_offset):  # 安全な範囲内でスキャン
        num_fields = (raw_data[pos + i] << 8) | raw_data[pos + i + 1]        
        # 0xFFFF終端マーカー検出
        if num_fields == 0xFFFF:
            return -3  # 終端マーカー検出
        
        if num_fields == ncols:
            return pos + i
    return -1

@cuda.jit(device=True, inline=True)
def validate_complete_row_fast(raw_data, row_start, expected_cols, fixed_field_lengths):
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
def detect_rows_optimized_debug(raw_data, header_size, thread_stride, estimated_row_size, ncols,
                               row_positions, row_count, max_rows, fixed_field_lengths, debug_array=None):
    """★whileループ終了原因トラッキング版: 未検出部分の詳細分析機能追加"""
    
    # ★共有メモリ拡張版（GPU制限: 48KB/ブロック）
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
        for i in range(MAX_SHARED_ROWS):
            block_positions[i] = -1
    
    cuda.syncthreads()  # ★重要: 初期化完了を保証
    
    # 各スレッドの担当範囲計算（境界オーバーラップ強化版）
    start_pos = header_size + tid * thread_stride
    end_pos = header_size + (tid + 1) * thread_stride + estimated_row_size  # ★修正1: オーバーラップ強化
    search_end = min(end_pos, raw_data.size - 1)
    
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
        # ★デバッグ記録: スレッド開始位置がデータ範囲外
        if debug_array is not None and tid < 100:  # 最初の100スレッドのみ記録
            loop_debug_base = 200 + tid * 4  # 各スレッド4要素使用
            if loop_debug_base + 3 < len(debug_array):
                debug_array[loop_debug_base] = tid
                debug_array[loop_debug_base + 1] = -1  # 終了原因: 開始位置が範囲外
                debug_array[loop_debug_base + 2] = start_pos
                debug_array[loop_debug_base + 3] = raw_data.size
        return
    
    # ★完全初期化：ローカル結果保存用
    local_positions = cuda.local.array(256, int32)
    local_count = 0
    pos = start_pos
    loop_iterations = 0
    exit_reason = 0  # 0:未設定, 1:正常終了(pos>=end_pos), 2:終端マーカー, 3:候補位置が担当外, 4:検証失敗で担当外
    
    # ★whileループ終了原因トラッキング + 境界オーバーラップ強化
    while pos < search_end:  # ★修正: end_pos → search_end（オーバーラップ範囲まで）
        loop_iterations += 1
        candidate_pos = read_uint16_simd16_in_thread(raw_data, pos, search_end, ncols)
        
        if candidate_pos == -2: # 終端マーカー検出
            exit_reason = 2
            break
        if candidate_pos == -3: # 担当外
            exit_reason = 3
            break

        if candidate_pos == -1: # 行ヘッダ候補なし
            pos += 15  # 15Bステップ
            continue
        
        if candidate_pos >= 0: # 有効な行ヘッダ候補あり
            if candidate_pos >= search_end: # 検索範囲外
                exit_reason = 3
                break
            
            # 完全行検証
            is_valid, row_end = validate_complete_row_fast(raw_data, candidate_pos, ncols, fixed_field_lengths)
            if not is_valid:
                if candidate_pos + 1 < search_end:
                    pos = candidate_pos + 1
                    continue
                else: # 検証失敗で検索範囲外
                    exit_reason = 4
                    break
            
            # 検証成功 → ローカル保存
            local_positions[local_count] = candidate_pos
            local_count += 1
            
            # 次の行開始位置へジャンプ
            if row_end > 0 and row_end < search_end:
                pos = row_end
                continue
            else: # row_end >= search_end
                exit_reason = 5
                break
    
    # 正常終了の場合
    if pos >= search_end and exit_reason == 0:
        exit_reason = 1
    
    # ★デバッグ記録: ループ終了原因とスレッド情報
    if debug_array is not None and tid < 100:
        loop_debug_base = 200 + tid * 4
        if loop_debug_base + 3 < len(debug_array):
            debug_array[loop_debug_base] = tid
            debug_array[loop_debug_base + 1] = exit_reason
            debug_array[loop_debug_base + 2] = pos  # 終了時の位置
            debug_array[loop_debug_base + 3] = local_count  # 検出した行数
    
    # ★ブロック単位協調処理（オーバーフロー対応版）
    if local_count > 0:
        local_base_idx = cuda.atomic.add(block_count, 0, local_count)
        
        for i in range(local_count):
            shared_idx = local_base_idx + i
            if shared_idx < MAX_SHARED_ROWS and local_positions[i] >= 0:
                block_positions[shared_idx] = local_positions[i]
            elif local_positions[i] >= 0:
                global_idx = cuda.atomic.add(row_count, 0, 1)
                if global_idx < max_rows:
                    row_positions[global_idx] = local_positions[i]
    
    cuda.syncthreads()
    
    # ★スレッド0による一括グローバル書き込み
    if local_tid == 0 and block_count[0] > 0:
        actual_rows = min(block_count[0], MAX_SHARED_ROWS)
        if actual_rows > 0:
            global_base_idx = cuda.atomic.add(row_count, 0, actual_rows)
            
            for i in range(actual_rows):
                if (global_base_idx + i < max_rows and
                    block_positions[i] >= 0 and
                    block_positions[i] < raw_data.size):
                    row_positions[global_base_idx + i] = block_positions[i]

def analyze_loop_exit_reasons(debug_results, num_threads, header_size, thread_stride, estimated_row_size):
    """whileループ終了原因の詳細分析"""
    
    print(f"[DEBUG] ★Whileループ終了原因分析（最初の100スレッド）:")
    
    # 終了原因の統計
    exit_reason_stats = {}
    thread_info = []
    
    for tid in range(min(100, num_threads)):
        loop_debug_base = 200 + tid * 4
        if loop_debug_base + 3 < len(debug_results):
            thread_id = debug_results[loop_debug_base]
            exit_reason = debug_results[loop_debug_base + 1]
            final_pos = debug_results[loop_debug_base + 2]
            detected_rows = debug_results[loop_debug_base + 3]
            
            if thread_id == tid:  # 有効なデータ
                if exit_reason not in exit_reason_stats:
                    exit_reason_stats[exit_reason] = 0
                exit_reason_stats[exit_reason] += 1
                
                thread_info.append({
                    'tid': thread_id,
                    'reason': exit_reason,
                    'final_pos': final_pos,
                    'detected_rows': detected_rows
                })
    
    # 終了原因の説明
    reason_names = {
        -1: "開始位置が範囲外",
        0: "未設定（異常）", 
        1: "正常終了(pos>=search_end)",
        2: "終端マーカー検出", 
        3: "候補位置が検索範囲外",
        4: "検証失敗で検索範囲外",
        5: "row_end >= search_end"
    }
    
    print(f"  終了原因統計:")
    for reason, count in sorted(exit_reason_stats.items()):
        reason_name = reason_names.get(reason, f"未知({reason})")
        print(f"    {reason_name}: {count}スレッド")
    
    # 詳細分析: 検出行数が0のスレッドの原因
    zero_detection_threads = [t for t in thread_info if t['detected_rows'] == 0]
    if zero_detection_threads:
        print(f"\n  検出行数0のスレッド詳細（{len(zero_detection_threads)}件）:")
        reason_distribution = {}
        for t in zero_detection_threads[:10]:  # 最初の10件表示
            reason_name = reason_names.get(t['reason'], f"未知({t['reason']})")
            if reason_name not in reason_distribution:
                reason_distribution[reason_name] = 0
            reason_distribution[reason_name] += 1
            
            # スレッドの担当範囲計算（オーバーラップ強化版）
            start_pos = header_size + t['tid'] * thread_stride
            end_pos = header_size + (t['tid'] + 1) * thread_stride + estimated_row_size
            coverage = end_pos - start_pos
            
            print(f"    Thread {t['tid']}: {reason_name}")
            print(f"      担当範囲: {start_pos}-{end_pos} ({coverage}B)")
            print(f"      終了位置: {t['final_pos']}")
        
        print(f"    検出失敗の主因:")
        for reason, count in sorted(reason_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"      {reason}: {count}スレッド")
    
    # 特定の見逃し位置に対応するスレッドの特定
    missing_positions = [39374583, 229138155]  # 既知の見逃し位置
    print(f"\n  見逃し位置 {missing_positions} の担当スレッド分析:")
    
    for missing_pos in missing_positions:
        # 位置を担当するスレッドを特定
        responsible_tid = (missing_pos - header_size) // thread_stride
        if responsible_tid < len(thread_info):
            thread_data = next((t for t in thread_info if t['tid'] == responsible_tid), None)
            if thread_data:
                reason_name = reason_names.get(thread_data['reason'], f"未知({thread_data['reason']})")
                start_pos = header_size + responsible_tid * thread_stride
                end_pos = header_size + (responsible_tid + 1) * thread_stride + estimated_row_size
                
                print(f"    位置 {missing_pos}:")
                print(f"      担当Thread: {responsible_tid}")
                print(f"      担当範囲: {start_pos}-{end_pos}")
                print(f"      終了原因: {reason_name}")
                print(f"      検出行数: {thread_data['detected_rows']}")
                print(f"      終了位置: {thread_data['final_pos']}")
                
                # 位置が担当範囲内かチェック
                if start_pos <= missing_pos < end_pos:
                    print(f"      ★範囲判定: 担当範囲内（見逃し確定）")
                else:
                    print(f"      ★範囲判定: 担当範囲外（境界問題の可能性）")
    
    return thread_info, exit_reason_stats

__all__ = [
    "detect_rows_optimized_debug",
    "analyze_loop_exit_reasons"
]