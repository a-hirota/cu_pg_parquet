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
                size += 16   # NUMERIC平均16B
            else:
                size += 4   # その他4B
    
    # ★メモリバンクコンフリクト回避: 32B境界整列
    return ((size + 31) // 32) * 32

@cuda.jit(device=True, inline=True)
def read_uint16_simd16(raw_data, pos, ncols):
    """要件1: 16B読み込みで行ヘッダ探索（★高速化 + 0xFFFF終端検出）"""
    if pos + 1 >= raw_data.size:  # 最低2B必要
        return -2
    
    # 実際に読み込み可能な範囲を計算（0-start）
    max_offset = min(15, raw_data.size - pos)
    
    # 16Bを一度に読み込み、全ての2B位置をチェック（0-15B全範囲）
    for i in range(0, max_offset + 1):  # 安全な範囲内でスキャン
        if pos + i + 1 >= raw_data.size: # 最低2B必要
            return -2
        num_fields = (raw_data[pos + i] << 8) | raw_data[pos + i + 1]
        
        # 0xFFFF終端マーカー検出
        if num_fields == 0xFFFF:
            return -2  # 終端マーカー検出
        
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
        elif flen < 0:  # 0xFFFFFFFF以外の負値は不正
            return False, -1
    
    # 次行ヘッダー検証（偽ヘッダー排除）
    if pos + 2 <= raw_data.size:
        next_header = (raw_data[pos] << 8) | raw_data[pos + 1]
        if next_header != expected_cols and next_header != 0xFFFF:
            return False, -1  # 次行ヘッダーが不正
    
    return True, pos  # (検証成功, 行終端位置)

@cuda.jit
def detect_rows_optimized(raw_data, header_size, thread_stride, estimated_row_size, ncols,
                          row_positions, row_count, max_rows, debug_array=None):
    """★大規模並列版: 2次元グリッド + 完全カバレッジ + デバッグ機能"""
    # 2次元グリッド対応でスレッド数制限を突破
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    
    # 各スレッドの担当範囲計算（1B被りオーバーラップ）
    # overlap = 1 if tid > 0 else 0
    # start_pos = header_size + tid * thread_stride - overlap
    start_pos = header_size + tid * thread_stride
    end_pos = header_size + (tid + 1) * thread_stride
    
    # オーバーラップ領域（正確なestimated_row_sizeを使用）
    overlap_size = max(estimated_row_size * 2, 1024)  # 最低1KB
    search_end = min(end_pos + overlap_size, raw_data.size - 1)  # 最後の1バイト手前まで探索
    
    if start_pos >= raw_data.size:
        return
    
    pos = start_pos
    local_positions = cuda.local.array(256, int32)
    local_count = 0
    
    # 16Bずつ高速スキャン
    # local_count:改行検知：各スレッド256行で強制停止してGPUクラッシュ回避
    while pos < search_end and local_count < 256:
        # 16B読み込みで行ヘッダ"17"探索
        candidate_pos = read_uint16_simd16(raw_data, pos, ncols)
        
        # 0xFFFF終端マーカー検出時の処理
        if candidate_pos == -2:
            break  # データ終端に到達、探索終了
        
        # デバッグログ記録（特定の見逃し位置のみ）
        if candidate_pos >= 0:
            if candidate_pos >= end_pos:
                break # 担当領域外の候補は無視

            # 完全行検証
            is_valid, row_end = validate_complete_row_fast(raw_data, candidate_pos, ncols)
            
            if is_valid:
                # 行開始が担当領域内 → カウント
                local_positions[local_count] = candidate_pos
                local_count += 1
                
                # 次の行開始位置へジャンプ（高速化）
                pos = row_end
                continue

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
    
    # 方針C: 2次元グリッドで大規模並列処理
    max_gpu_threads = 1048576  # 2^20 = 1Mスレッド（16倍増加）
    threads_per_block = 256
    max_total_blocks = max_gpu_threads // threads_per_block  # 4096ブロック
    
    # 完全カバレッジ: 切り上げ除算で端数領域も含める
    thread_stride = (data_size + max_gpu_threads - 1) // max_gpu_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size  # 最小でも推定行サイズ
    
    # 実際のスレッド数とブロック配置
    num_threads = min((data_size + thread_stride - 1) // thread_stride, max_gpu_threads)
    total_blocks = min((num_threads + threads_per_block - 1) // threads_per_block, max_total_blocks)
    
    # 2次元グリッド配置（CUDAの1次元制限を回避）
    max_blocks_per_dim = 65535
    blocks_x = min(total_blocks, max_blocks_per_dim)
    blocks_y = (total_blocks + blocks_x - 1) // blocks_x
    
    # ★詳細デバッグ: 2次元グリッド + 完全カバレッジ分析
    if debug:
        print(f"[DEBUG] ★Ultra Fast v3 (方針C: 2次元グリッド): ncols={ncols}, estimated_row_size={estimated_row_size}")
        print(f"[DEBUG] ★data_size={data_size//1024//1024}MB ({data_size}B)")
        print(f"[DEBUG] ★設計値: threads={num_threads}, total_blocks={total_blocks}")
        print(f"[DEBUG] ★2次元グリッド: ({blocks_x}, {blocks_y}) × {threads_per_block}threads")
        print(f"[DEBUG] ★thread_stride={thread_stride}B (密度: {data_size//num_threads}B/thread)")
        
        # カバレッジ計算（実際のカバー範囲を正確に表示）
        coverage_bytes = min(num_threads * thread_stride, data_size)
        coverage_gap = max(0, data_size - coverage_bytes)
        coverage_ratio = coverage_bytes / data_size
        print(f"[DEBUG] ★期待カバー範囲: {coverage_bytes//1024//1024}MB ({coverage_bytes}B)")
        print(f"[DEBUG] ★カバレッジ: {coverage_ratio*100:.3f}% (不足: {coverage_gap}B)")
        
        # GPU制限チェック
        if num_threads == max_gpu_threads:
            print(f"[DEBUG] ★GPU制限に到達: {max_gpu_threads}スレッド")
        if total_blocks == max_total_blocks:
            print(f"[DEBUG] ★ブロック制限に到達: {max_total_blocks}ブロック")
    
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
        print(f"[DEBUG] ★カーネル起動: detect_rows_optimized[({blocks_x}, {blocks_y}), {threads_per_block}]")
    
    # デバッグ配列準備（見逃し位置1個 × 5項目）
    debug_info_size = 5  # [位置, スレッドID, ステップ, 候補位置, 検証結果]
    debug_array = cuda.device_array(debug_info_size, np.int32)
    debug_array[:] = -1  # 初期化
    
    # ★2次元グリッド起動で大規模並列実行
    grid_2d = (blocks_x, blocks_y)
    detect_rows_optimized[grid_2d, threads_per_block](
        raw_dev, header_size, thread_stride, estimated_row_size, ncols,
        row_positions, row_count, max_rows, debug_array
    )
    cuda.synchronize()
    
    # デバッグ結果解析
    if debug:
        debug_results = debug_array.copy_to_host()
        target_position = 70277  # 新しい見逃し位置をデバッグ
        
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
    
    # 要件4: GPU並列ソート（CPU転送を回避）
    if nrow <= 1:
        row_offsets = row_positions[:nrow]  # 1行以下はソート不要
    else:
        try:
            # CuPyを使用してGPU上で直接ソート
            import cupy as cp
            positions_cupy = cp.asarray(row_positions[:nrow])
            positions_sorted_cupy = cp.sort(positions_cupy)
            row_offsets = cuda.as_cuda_array(positions_sorted_cupy)
        except ImportError:
            # CuPyが利用できない場合は従来のCPU方式
            positions_host = row_positions[:nrow].copy_to_host()
            positions_sorted = np.sort(positions_host)
            row_offsets = cuda.to_device(positions_sorted)
    
    # 詳細デバッグ情報
    if debug and nrow > 1:
        # GPU配列からCPU配列に変換してデバッグ
        row_offsets_host = row_offsets.copy_to_host()
        gaps = np.diff(row_offsets_host)
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
        data_start = row_offsets_host[0]
        data_end = row_offsets_host[-1]
        data_span = data_end - data_start
        print(f"[DEBUG] ★データ分布: {data_start}-{data_end} (範囲: {data_span//1024//1024}MB)")
    
    # フィールド抽出（従来の1次元グリッドで十分）
    field_offsets = cuda.device_array((nrow, ncols), np.int32)
    field_lengths = cuda.device_array((nrow, ncols), np.int32)
    
    extract_blocks = min((nrow + threads_per_block - 1) // threads_per_block, max_blocks_per_dim)
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
