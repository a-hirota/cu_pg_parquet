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
    from ..types import UTF8, DECIMAL128
    
    size = 2  # フィールド数(2B)
    for col in columns:
        size += 4  # フィールド長(4B)
        if col.elem_size > 0:  # 固定長
            size += col.elem_size
        else:  # 可変長の推定
            if col.arrow_id == UTF8:
                size += 20  # 文字列平均20B
            elif col.arrow_id == DECIMAL128:
                size += 8   # NUMERIC平均8B
            else:
                size += 4   # その他4B
    
    # ★メモリバンクコンフリクト回避: 32B境界整列
    return ((size + 31) // 32) * 32

@cuda.jit(device=True, inline=True)
def read_uint16_simd16(raw_data, pos, ncols):
    """要件1: 16B読み込みで行ヘッダ探索（★高速化）"""
    if pos + 16  > raw_data.size: # 2B単位の探索のため、16+1Bまで検索が必要 
        return -1
    
    # 16Bを一度に読み込み、全ての2B位置をチェック（0-15B全範囲）
    for i in range(0, 15):  # (pos+i+1 ≤ pos+15)
        num_fields = (raw_data[pos + i] << 8) | raw_data[pos + i + 1]
        if num_fields == ncols:
            return pos + i
    return -1

@cuda.jit(device=True, inline=True)
def validate_complete_row_fast(raw_data, row_start, expected_cols):
    # raw_dataには、PostgreSQLのCOPY BINARYデータ全体が入ります。
    """要件2: 完全な行検証 + 越境処理対応"""
    if row_start + 2 > raw_data.size:
        return False, -1
    
    # フィールド数確認
    num_fields = (raw_data[row_start] << 8) | raw_data[row_start+1]
    if num_fields != expected_cols:
        return False, -1
    
    pos = row_start + 2
    # 全フィールドを順次検証
    for _ in range(num_fields):
        if pos + 4 >= raw_data.size:
            return False, -1
        
        flen = (
            int32(raw_data[pos  ]) << 24 | int32(raw_data[pos+1]) << 16 |
            int32(raw_data[pos+2]) << 8  | int32(raw_data[pos+3])
        )
        pos += 4
        
        if flen > 0:
            if pos + flen >= raw_data.size:
                return False, -1
            pos += flen
        elif flen == 0xFFFFFFFF:
            continue
        elif flen < 0 and flen != 0xFFFFFFFF:  # 0xFFFFFFFF以外の負値は不正
            return False, -1
    
    return True, pos  # (検証成功, 行終端位置)

@cuda.jit
def detect_rows_optimized(raw_data, header_size, thread_stride, estimated_row_size, ncols,
                          row_positions, row_count, max_rows, debug_array=None):
    """★境界条件修正版: 行開始位置で判定 + 正確なestimated_row_size + デバッグ機能"""
    tid = cuda.grid(1)
    
    # 各スレッドの担当範囲計算（1B被りオーバーラップ）
    overlap = 1 if tid > 0 else 0
    start_pos = header_size + tid * thread_stride - overlap
    end_pos = header_size + (tid + 1) * thread_stride
    
    # オーバーラップ領域（正確なestimated_row_sizeを使用）
    overlap_size = max(estimated_row_size * 2, 1024)  # 最低1KB
    search_end = min(end_pos + overlap_size, raw_data.size - 16)
    
    if start_pos >= raw_data.size:
        return
    
    pos = start_pos
    local_positions = cuda.local.array(256, int32)
    local_count = 0
    
    # 16Bずつ高速スキャン
    while pos < search_end and local_count < 256:
        # 16B読み込みで行ヘッダ"17"探索
        candidate_pos = read_uint16_simd16(raw_data, pos, ncols)
        
        # デバッグログ記録（特定の見逃し位置のみ）
        if debug_array is not None:
            # 位置 2948855 のチェック
            if pos <= 2948855 < pos + 16:
                if candidate_pos == 2948855:
                    debug_array[0] = 2948855  # 対象位置
                    debug_array[1] = tid      # スレッドID
                    debug_array[2] = 1        # ステップ: 候補発見
                elif candidate_pos >= 0:
                    debug_array[2] = 2        # ステップ: 別候補発見
                    debug_array[3] = candidate_pos
                else:
                    debug_array[2] = 0        # ステップ: 候補なし
        
        if candidate_pos >= 0:
            # 完全行検証
            is_valid, row_end = validate_complete_row_fast(raw_data, candidate_pos, ncols)
            
            # デバッグログ: 検証結果記録（位置 2948855 のみ）
            if debug_array is not None and candidate_pos == 2948855:
                debug_array[4] = 1 if is_valid else 0  # 検証結果
            
            if is_valid:
                # ★境界条件修正: 行開始位置で判定
                if candidate_pos < end_pos:
                    # 行開始が担当領域内 → カウント
                    local_positions[local_count] = candidate_pos
                    local_count += 1
                    
                    # デバッグログ: カウント成功（位置 2948855 のみ）
                    if debug_array is not None and candidate_pos == 2948855:
                        debug_array[2] = 3  # ステップ: カウント成功
                    
                    # 次の行開始位置へジャンプ（高速化）
                    pos = row_end
                    continue
                elif candidate_pos < search_end:
                    # オーバーラップ領域 → 確認のみ、カウントしない
                    
                    # デバッグログ: オーバーラップ領域（位置 2948855 のみ）
                    if debug_array is not None and candidate_pos == 2948855:
                        debug_array[2] = 4  # ステップ: オーバーラップ
                    
                    # 次の行開始位置へジャンプ（高速化）
                    pos = row_end
                    continue
                else:
                    # デバッグログ: 範囲外（位置 2948855 のみ）
                    if debug_array is not None and candidate_pos == 2948855:
                        debug_array[2] = 5  # ステップ: 範囲外
            else:
                # 検証失敗：候補位置+1から再開（16B内の他の候補を見逃さない）
                pos = candidate_pos + 1
                continue
        
        # 候補が見つからない場合のみ15Bステップで1B被りオーバーラップ
        pos += 15
    
    # グローバル配列にアトミック記録
    if local_count > 0:
        base_idx = cuda.atomic.add(row_count, 0, local_count)
        for i in range(local_count):
            if base_idx + i < max_rows:
                row_positions[base_idx + i] = local_positions[i]

def parse_binary_chunk_gpu_ultra_fast_v2(raw_dev, columns, header_size: int = None, *, debug: bool=False):
    """★完全実装版: 方針B + 詳細デバッグで原因特定"""
    if header_size is None:
        header_size = 19
    
    ncols = len(columns)
    # 要件0: 推定行サイズ計算（★実装）
    estimated_row_size = estimate_row_size_from_columns(columns)
    
    # データサイズ
    data_size = raw_dev.size - header_size
    
    # 方針B: GPU制限内で最大密度（元の設計に戻す）
    max_gpu_threads = 65536
    threads_per_block = 256
    max_blocks = max_gpu_threads // threads_per_block
    
    # 元の設計: スレッド間距離 = データサイズ ÷ 最大スレッド数
    thread_stride = data_size // max_gpu_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size  # 最小でも推定行サイズ
    
    # 実際のスレッド数
    num_threads = (data_size + thread_stride - 1) // thread_stride
    num_threads = min(num_threads, max_gpu_threads)
    num_blocks = min((num_threads + threads_per_block - 1) // threads_per_block, max_blocks)
    
    # ★詳細デバッグ: Grid起動問題とカバレッジ特定
    if debug:
        print(f"[DEBUG] ★Ultra Fast v2 (方針B): ncols={ncols}, estimated_row_size={estimated_row_size}")
        print(f"[DEBUG] ★data_size={data_size//1024//1024}MB ({data_size}B)")
        print(f"[DEBUG] ★設計値: threads={num_threads}, blocks={num_blocks}")
        print(f"[DEBUG] ★thread_stride={thread_stride}B")
        
        # カバレッジ計算
        coverage_bytes = num_threads * thread_stride
        coverage_ratio = coverage_bytes / data_size
        print(f"[DEBUG] ★期待カバー範囲: {coverage_bytes//1024//1024}MB ({coverage_bytes}B)")
        print(f"[DEBUG] ★カバレッジ: {coverage_ratio*100:.1f}%")
        
        # GPU制限チェック
        if num_threads == max_gpu_threads:
            print(f"[DEBUG] ★GPU制限に到達: {max_gpu_threads}スレッド")
        if num_blocks == max_blocks:
            print(f"[DEBUG] ★ブロック制限に到達: {max_blocks}ブロック")
    
    # デバイス配列準備
    max_rows = min(2_000_000, (data_size // estimated_row_size) * 2)
    row_positions = cuda.device_array(max_rows, np.int32)
    row_count = cuda.device_array(1, np.int32)
    row_count[0] = 0
    
    # Grid起動状況を確認するためのデバッグ配列
    debug_info = cuda.device_array(3, np.int32)  # [実際のブロック数, 実際のスレッド数, 総スレッド数]
    debug_info[0] = 0
    debug_info[1] = 0
    debug_info[2] = 0
    
    if debug:
        print(f"[DEBUG] ★カーネル起動: detect_rows_optimized[{num_blocks}, {threads_per_block}]")
    
    # デバッグ配列準備（見逃し位置1個 × 5項目）
    debug_info_size = 5  # [位置, スレッドID, ステップ, 候補位置, 検証結果]
    debug_array = cuda.device_array(debug_info_size, np.int32)
    debug_array[:] = -1  # 初期化
    
    # ★全要件統合実行（estimated_row_sizeを引数追加）
    detect_rows_optimized[num_blocks, threads_per_block](
        raw_dev, header_size, thread_stride, estimated_row_size, ncols,
        row_positions, row_count, max_rows, debug_array
    )
    cuda.synchronize()
    
    # デバッグ結果解析
    if debug:
        debug_results = debug_array.copy_to_host()
        target_position = 2948855  # 最初の見逃し位置をデバッグ
        
        print(f"[DEBUG] ★見逃し原因詳細分析 (位置 {target_position}):")
        
        position = debug_results[0]
        thread_id = debug_results[1]
        step = debug_results[2]
        candidate = debug_results[3]
        validation = debug_results[4]
        
        if position == target_position:  # この位置がデバッグ対象として処理された
            step_names = {
                -1: "未処理",
                0: "16B検索で候補なし",
                1: "16B検索で対象位置発見",
                2: "16B検索で別候補発見",
                3: "検証成功・カウント完了",
                4: "検証成功・オーバーラップ領域",
                5: "検証成功・範囲外"
            }
            
            validation_names = {
                -1: "未実行",
                0: "検証失敗",
                1: "検証成功"
            }
            
            print(f"  位置 {target_position}:")
            print(f"    担当スレッド: {thread_id}")
            print(f"    処理結果: {step_names.get(step, f'不明({step})')}")
            if candidate >= 0:
                print(f"    検出候補: {candidate}")
            print(f"    検証結果: {validation_names.get(validation, f'不明({validation})')}")
            
            # 見逃し原因の推定
            if step == 0:
                print(f"    → 見逃し原因: 16B検索で'00 11'パターンが検出されなかった")
            elif step == 2:
                print(f"    → 見逃し原因: 別の候補位置{candidate}が先に検出された")
            elif step == 1 and validation == 0:
                print(f"    → 見逃し原因: 候補は発見されたが検証で失敗")
            elif step >= 3:
                print(f"    → 見逃し原因: 正常処理されているはず（他の問題の可能性）")
            else:
                print(f"    → 見逃し原因: 不明（詳細調査が必要）")
        else:
            print(f"  位置 {target_position}: デバッグ情報なし（この位置を担当するスレッドで処理されなかった可能性）")
            print(f"    実際に記録された位置: {position}")
    
    nrow = int(row_count.copy_to_host()[0])
    print(f"[DEBUG] ★Ultra Fast v2 detected {nrow} rows")
    
    if nrow == 0:
        print("[DEBUG] No rows detected")
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
    
    # 要件4: 累積和処理（ソート形式で高速実装）
    positions_host = row_positions[:nrow].copy_to_host()
    positions_sorted = np.sort(positions_host)  # CPU側ソート = 累積和の効率的実装
    row_offsets = cuda.to_device(positions_sorted)
    
    # 詳細デバッグ情報
    if debug and nrow > 1:
        gaps = np.diff(positions_sorted)
        print(f"[DEBUG] ★rows {nrow} min {gaps.min()} max {gaps.max()} avg {float(gaps.mean()):.2f}")
        small = (gaps < 8).sum()
        print(f"[DEBUG] ★small_rows: {small} (目標: 0)")
        
        # 検出率分析
        expected_rows = 1_000_000
        detection_rate = nrow / expected_rows * 100
        missing_rows = expected_rows - nrow
        print(f"[DEBUG] ★期待: {expected_rows}, 検出: {nrow}, 検出率: {detection_rate:.1f}%")
        print(f"[DEBUG] ★不足: {missing_rows}行 ({missing_rows/expected_rows*100:.1f}%)")
        
        # データ分布分析
        data_start = positions_sorted[0]
        data_end = positions_sorted[-1]
        data_span = data_end - data_start
        print(f"[DEBUG] ★データ分布: {data_start}-{data_end} (範囲: {data_span//1024//1024}MB)")
    
    # フィールド抽出
    field_offsets = cuda.device_array((nrow, ncols), np.int32)
    field_lengths = cuda.device_array((nrow, ncols), np.int32)
    
    extract_blocks = (nrow + threads_per_block - 1) // threads_per_block
    extract_fields[extract_blocks, threads_per_block](
        raw_dev, row_offsets, field_offsets, field_lengths, nrow, ncols
    )
    cuda.synchronize()
    
    return field_offsets, field_lengths

# ───── 下位互換ホストドライバ ─────
def parse_binary_chunk_gpu_ultra_fast(raw_dev, ncols: int, header_size: int = None, first_col_size: int = None, *, debug: bool=False):
    """下位互換ラッパー（従来インターフェース対応）"""
    # 簡易的なcolumns作成
    from ..types import ColumnMeta, INT32
    columns = [ColumnMeta(name=f"col_{i}", pg_oid=23, pg_typmod=0,
                         arrow_id=INT32, elem_size=4) for i in range(ncols)]
    
    return parse_binary_chunk_gpu_ultra_fast_v2(raw_dev, columns, header_size, debug=debug)

__all__ = ["parse_binary_chunk_gpu_ultra_fast", "parse_binary_chunk_gpu_ultra_fast_v2", "estimate_row_size_from_columns"]
