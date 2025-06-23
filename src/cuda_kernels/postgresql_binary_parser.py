"""
PostgreSQL Binary Parser
------------------------
PostgreSQLバイナリ形式の完全パース処理

機能:
- 行ヘッダー検出（16B SIMD並列検索）
- 行検証（完全行チェック）
- フィールド抽出（列データの位置・長さ取得）
"""

from numba import cuda, int32, types
import numpy as np

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

# row_cnt[0]: グローバルメモリ上の行カウント。atomic に加算し row_off[] の書き込み先を確保。
# row_off[]: 各行の開始オフセットを保持。次段階の列抽出カーネルで参照される。
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
def read_uint16_simd16(raw_data, pos, end_pos, ncols):
   
    # 終端位置を超えないようにする
    if pos + 1 > raw_data.size:  # 最低2B必要(終端位置)
        return -2
   
    max_offset = min(16, raw_data.size - pos + 1)
    
    # 16Bを一度に読み込み、全ての2B位置をチェック（0-15B全範囲）
    for i in range(0, max_offset):  # 安全な範囲内でスキャン
        
        # 担当範囲を超えないようにする
        if pos + i+ 1 > end_pos: # 最低2B必要(担当外)
            return -3    
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
                           row_positions, row_count, max_rows, fixed_field_lengths):
    """最適化版行検出カーネル: シンプル化された高速実装"""
    
    # 共有メモリ設定
    MAX_SHARED_ROWS = 2048
    block_positions = cuda.shared.array(MAX_SHARED_ROWS, int32)
    block_count = cuda.shared.array(1, int32)
    
    # スレッドID計算
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    local_tid = cuda.threadIdx.x
    
    # 共有メモリ初期化
    if local_tid == 0:
        block_count[0] = 0
        for i in range(MAX_SHARED_ROWS):
            block_positions[i] = -1
    
    cuda.syncthreads()
    
    # 担当範囲計算
    start_pos = header_size + tid * thread_stride
    end_pos = header_size + (tid + 1) * thread_stride
    
    if start_pos >= raw_data.size:
        return
    
    # ローカル結果保存用
    local_positions = cuda.local.array(256, int32)
    local_count = 0
    pos = start_pos
    
    # 高速行検出ループ
    while pos < end_pos:
        candidate_pos = read_uint16_simd16(raw_data, pos, end_pos, ncols)
        
        if candidate_pos <= -2:  # データ終端
            break

        if candidate_pos == -1:  # 行ヘッダ候補なし
            pos += 15
            continue
        
        if candidate_pos >= 0:  # 有効な行ヘッダ候補
            if candidate_pos >= end_pos:  # 担当外
                break
            
            # 完全行検証
            is_valid, row_end = validate_complete_row_fast(raw_data, candidate_pos, ncols, fixed_field_lengths)
            if not is_valid:
                if candidate_pos + 1 < end_pos:
                    pos = candidate_pos + 1
                    continue
                else:
                    break
            
            # 検証成功
            local_positions[local_count] = candidate_pos
            local_count += 1
            
            # 次の行開始位置へジャンプ（境界処理改善）
            # 行の開始が担当範囲内なら、終了が範囲外でも処理
            if row_end > 0:
                pos = row_end
                continue
            else:
                break
    
    # ブロック単位協調処理
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
    
    # スレッド0による一括グローバル書き込み
    if local_tid == 0 and block_count[0] > 0:
        actual_rows = min(block_count[0], MAX_SHARED_ROWS)
        if actual_rows > 0:
            global_base_idx = cuda.atomic.add(row_count, 0, actual_rows)
            
            for i in range(actual_rows):
                if (global_base_idx + i < max_rows and
                    block_positions[i] >= 0 and
                    block_positions[i] < raw_data.size):
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
    """PostgreSQLバイナリパーサー"""
    if header_size is None:
        header_size = 19
    
    ncols = len(columns)
    
    # 固定長フィールド情報を抽出
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
    data_size = raw_dev.size - header_size
    
    # グリッド配置の最適化
    blocks_x, blocks_y, threads_per_block = calculate_optimal_grid_sm_aware(
        data_size, estimated_row_size
    )
    
    # スレッド数とストライド計算
    actual_threads = blocks_x * blocks_y * threads_per_block
    thread_stride = (data_size + actual_threads - 1) // actual_threads
    if thread_stride < estimated_row_size:
        thread_stride = estimated_row_size
    
    # GPU特性情報
    props = get_device_properties()
    max_blocks_per_dim = props.get('MAX_GRID_DIM_X', 65535)
    
    if debug:
        sm_count = props.get('MULTIPROCESSOR_COUNT', 'Unknown')
        print(f"[DEBUG] PostgreSQL Binary Parser: ncols={ncols}, estimated_row_size={estimated_row_size}")
        print(f"[DEBUG] data_size={data_size//1024//1024}MB, threads={actual_threads}")
        print(f"[DEBUG] grid: ({blocks_x}, {blocks_y}) × {threads_per_block}, SM効率: {(blocks_x * blocks_y) / sm_count:.1f}")
    
    # デバイス配列準備
    # 推定行数の1.5倍のマージンを持たせる（動的拡張対応）
    estimated_rows = data_size // estimated_row_size
    max_rows = min(int(estimated_rows * 1.5), 200_000_000)  # 最大2億行まで
    
    # 1回実行（シンプル化）
    row_positions = cuda.device_array(max_rows, np.int32)
    row_count = cuda.device_array(1, np.int32)
    row_count[0] = 0
    
    cuda.synchronize()
    
    # カーネル実行
    grid_2d = (blocks_x, blocks_y)
    detect_rows_optimized[grid_2d, threads_per_block](
        raw_dev, header_size, thread_stride, estimated_row_size, ncols,
        row_positions, row_count, max_rows, fixed_field_lengths_dev
    )
    cuda.synchronize()
    
    # 結果取得
    nrow = int(row_count.copy_to_host()[0])
    if nrow > 0:
        # 適応的ソート: 大規模データ（50,000行以上）でのみGPUソートを使用
        GPU_SORT_THRESHOLD = 50000
        
        if nrow >= GPU_SORT_THRESHOLD:
            try:
                import cupy as cp
                
                # GPU上で高速ソート処理（CPU転送排除）
                row_positions_gpu = cp.asarray(row_positions[:nrow])
                valid_mask = row_positions_gpu >= 0
                
                if cp.any(valid_mask):
                    row_offsets_valid = row_positions_gpu[valid_mask]
                    row_offsets_sorted = cp.sort(row_offsets_valid)
                    
                    # CuPy配列をNumba CUDA配列に変換
                    row_offsets = cuda.as_cuda_array(row_offsets_sorted)
                    row_offsets_host = row_offsets_sorted.get()  # デバッグ用にCPU配列も保持
                    nrow = len(row_offsets_host)
                    
                    if debug:
                        print(f"[DEBUG] GPU高速ソート完了: {nrow}行（大規模データ最適化）")
                else:
                    row_offsets = cuda.device_array(0, np.int32)
                    row_offsets_host = np.array([], dtype=np.int32)
                    nrow = 0
                    
            except ImportError:
                # CuPy未対応時はCPUソートにフォールバック
                pass
        
        # 小規模データまたはGPUソート未対応時のCPUソート
        if nrow < GPU_SORT_THRESHOLD or 'row_offsets' not in locals():
            if debug and nrow < GPU_SORT_THRESHOLD:
                print(f"[DEBUG] 小規模データ（{nrow}行）のためCPU最適ソートを使用")
            elif debug:
                print("[DEBUG] GPUソート未対応環境、CPUソートを使用")
            
            row_offsets_host = row_positions[:nrow].copy_to_host()
            row_offsets_host = row_offsets_host[row_offsets_host >= 0]  # 有効位置のみ
            row_offsets_host = np.sort(row_offsets_host)
            row_offsets = cuda.to_device(row_offsets_host)
            nrow = len(row_offsets_host)
    else:
        row_offsets = cuda.device_array(0, np.int32)
        row_offsets_host = np.array([], dtype=np.int32)
        nrow = 0
    
    if debug:
        print(f"[DEBUG] 検出完了: {nrow}行")
        if nrow > 1:
            gaps = np.diff(row_offsets_host)
            print(f"[DEBUG] 行間隔 - min: {gaps.min()}, max: {gaps.max()}, avg: {float(gaps.mean()):.2f}")
    
    if nrow == 0:
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
    
    # フィールド抽出
    field_offsets_host = np.zeros((nrow, ncols), dtype=np.int32)
    field_lengths_host = np.full((nrow, ncols), -1, dtype=np.int32)
    field_offsets = cuda.to_device(field_offsets_host)
    field_lengths = cuda.to_device(field_lengths_host)
    
    extract_blocks = min((nrow + threads_per_block - 1) // threads_per_block, max_blocks_per_dim)
    
    try:
        extract_fields[extract_blocks, threads_per_block](
            raw_dev, row_offsets, field_offsets, field_lengths, nrow, ncols
        )
        cuda.synchronize()
    except Exception as e:
        if debug:
            print(f"[DEBUG] フィールド抽出エラー: {e}")
        return cuda.device_array((0, ncols), np.int32), cuda.device_array((0, ncols), np.int32)
    
    return field_offsets, field_lengths

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

# ===== 統合最適化版の追加 =====
def parse_binary_chunk_gpu_ultra_fast_v2_integrated(raw_dev, columns, header_size=None, debug=False):
    """
    統合最適化版: 行検出+フィールド抽出を1回で実行
    
    メモリ最適化:
    - 重複読み込み排除: 218MB × 2回 → 218MB × 1回（50%削減）
    - キャッシュ効率向上: L1/L2キャッシュ最大活用
    
    パフォーマンス向上:
    - 実行時間短縮: 0.6秒 → 0.4秒予想（33%短縮）
    - GPU↔メモリ間トラフィック削減
    """
    try:
        # 軽量版を優先使用（共有メモリ制限対応）
        from .integrated_parser_lite import parse_binary_chunk_gpu_ultra_fast_v2_lite as lite_impl
        return lite_impl(raw_dev, columns, header_size, debug)
    except ImportError:
        try:
            # 標準統合版をフォールバック
            from .integrated_parser import parse_binary_chunk_gpu_ultra_fast_v2_integrated as integrated_impl
            return integrated_impl(raw_dev, columns, header_size, debug)
        except ImportError:
            # 最終フォールバック: 従来版を使用
            if debug:
                print("[WARNING] 統合版が利用できません。従来版を使用します。")
            return parse_binary_chunk_gpu_ultra_fast_v2(raw_dev, columns, header_size, debug=debug)

__all__ = [
    "parse_binary_chunk_gpu_ultra_fast",
    "parse_binary_chunk_gpu_ultra_fast_v2",
    "parse_binary_chunk_gpu_ultra_fast_v2_integrated",  # 統合版を追加
    "detect_pg_header_size",  # ヘッダーサイズ検出を追加
    "estimate_row_size_from_columns",
    "get_device_properties",
    "calculate_optimal_grid_sm_aware"
]