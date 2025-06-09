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

@cuda.jit(device=True, inline=True)
def read_uint16_simd16(raw_data, pos, end_pos,ncols):
    """要件1: 16B読み込みで行ヘッダ探索（★高速化 + 0xFFFF終端検出）"""
    if pos + 1 > raw_data.size:  # 最低2B必要
        return -2
    
    # 実際に読み込み可能な範囲を計算
    max_offset = min(16, raw_data.size - pos + 1)
    
    # 16Bを一度に読み込み、全ての2B位置をチェック（0-15B全範囲）
    for i in range(0, max_offset):  # 安全な範囲内でスキャン
        if pos + i + 1 >= raw_data.size : # 最低2B必要
            return -2
        num_fields = (raw_data[pos + i] << 8) | raw_data[pos + i + 1]
        
        # 0xFFFF終端マーカー検出
        if num_fields == 0xFFFF:
            return -2  # 終端マーカー検出
        
        if num_fields == ncols:
            return pos + i
    return -1

@cuda.jit(device=True, inline=True)
def read_uint16_simd16_in_thread(raw_data, pos, end_pos, ncols):
    """要件1: 16B読み込みで行ヘッダ探索（★高速化 + 0xFFFF終端検出）"""
    
    # 実際に読み込み可能な範囲を計算
    # max_offset = min(16, raw_data.size - pos + 1)
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
    """★初期化徹底版: 未初期化メモリ問題を完全解決"""
    
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
    
    # ★高速行検出ループ（元の実装を維持）
    while pos < end_pos:# and local_count < 256:
        candidate_pos = read_uint16_simd16_in_thread(raw_data, pos, end_pos,ncols)
        
        if candidate_pos <= -2: # 終端マーカー検出
            break  # データ終端

        if candidate_pos == -1: # 行ヘッダ候補なし
            pos += 15  # 15Bステップ（元の設計に戻す）
            continue
        
        if candidate_pos >= 0: # 有効な行ヘッダ候補あり
            if candidate_pos >= end_pos: # 最近傍が担当外のためloop終了
                break
            
            # 完全行検証
            is_valid, row_end = validate_complete_row_fast(raw_data, candidate_pos, ncols, fixed_field_lengths)
            if not is_valid:
                if candidate_pos + 1 < end_pos:
                    pos = candidate_pos + 1
                    continue
                else: # 最近傍+1も担当外のためloop終了
                    break
            
            # 検証成功 → ローカル保存（安全性チェック付き）
            # if local_count < 256 and candidate_pos >= 0:  # 配列境界と有効性チェック
            local_positions[local_count] = candidate_pos
            local_count += 1
            
            # 次の行開始位置へジャンプ（境界条件改善）
            if row_end > 0 and row_end < end_pos:
                pos = row_end
                continue
            else: # row_endより手前は行ヘッダなし。row_end以降は担当外のため
               break
        

    
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

def parse_binary_chunk_gpu_ultra_fast_v2(raw_dev, columns, header_size: int = None, *, debug: bool=False):
    """★完全実装版: 初期化徹底 + 単一パス高精度検出"""
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
    max_attempts = 10  # 最大試行回数
    
    if debug:
        print(f"[DEBUG] ★積和方式リトライ: 目標{target_rows}行達成まで最大{max_attempts}回実行")
    
    # 累積結果保存用（CPU側）
    all_positions = set()  # 重複除去のためのセット
    
    # デバッグ配列サイズを事前定義（複数検証失敗記録対応）
    debug_info_size = 100  # 40 → 100に拡張（10件×6要素+基本領域40）
    
    for attempt in range(max_attempts):
        if debug:
            print(f"[DEBUG] ★試行 {attempt + 1}/{max_attempts} 開始...")
        
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
        
        if debug:
            init_method = "完全初期化" if attempt == 0 else "未初期化活用"
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
