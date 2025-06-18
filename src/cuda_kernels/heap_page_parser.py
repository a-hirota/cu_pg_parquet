"""
PostgreSQLヒープページ解析CUDAカーネル

PostgreSQLのヒープファイルからページ構造を解析し、
有効なタプル（行）のオフセットを抽出するためのGPUカーネル。

主な機能：
- ヒープページヘッダーの解析
- ItemId（ラインポインタ）配列の解析
- 有効なタプルの識別（t_xmaxフィールドによる削除チェック）
- GPUメモリ上でのタプルオフセット抽出

使用方法：
    heap_data = kvikio.CuFile(...).read()  # ヒープファイル読み込み
    page_offsets = cp.array([0, 8192, 16384, ...])  # 8KBページオフセット
    tuple_offsets_out = cp.zeros(max_tuples, dtype=cp.uint32)
    tuple_count_out = cp.zeros(num_pages, dtype=cp.uint32)
    
    parse_heap_pages_to_tuples[blocks, threads](
        heap_data, page_offsets, tuple_offsets_out, tuple_count_out
    )
"""

import cupy as cp
import numpy as np
from numba import cuda
from numba.cuda import atomic


# PostgreSQL ヒープページの定数
POSTGRES_PAGE_SIZE = 8192  # デフォルトページサイズ（8KB）
PAGE_HEADER_SIZE = 24      # PageHeaderData構造体のサイズ
ITEM_ID_SIZE = 4          # ItemIdData構造体のサイズ（4バイト）
TUPLE_HEADER_MIN_SIZE = 23 # HeapTupleHeaderの最小サイズ

# ItemIdフラグ
LP_UNUSED = 0     # 未使用
LP_NORMAL = 1     # 通常のタプル
LP_REDIRECT = 2   # リダイレクト
LP_DEAD = 3       # 削除済み

# タプルヘッダーオフセット
T_XMAX_OFFSET = 8  # t_xmaxフィールドのオフセット（TransactionIdサイズ）


@cuda.jit(device=True, inline=True)
def read_uint16_le(data, offset):
    """リトルエンディアンでuint16を読み取り"""
    if offset + 1 >= data.size:
        return np.uint16(0)
    return np.uint16(data[offset] | (data[offset + 1] << 8))


@cuda.jit(device=True, inline=True)
def read_uint32_le(data, offset):
    """リトルエンディアンでuint32を読み取り"""
    if offset + 3 >= data.size:
        return np.uint32(0)
    return np.uint32(data[offset] |
                     (data[offset + 1] << 8) |
                     (data[offset + 2] << 16) |
                     (data[offset + 3] << 24))


@cuda.jit(device=True, inline=True)
def validate_page_header(data, page_offset):
    """ページヘッダーの妥当性を検証"""
    if page_offset + PAGE_HEADER_SIZE >= len(data):
        return False
    
    # pd_lsn（8バイト）をスキップしてpd_checksumを確認
    # pd_flags（2バイト、オフセット10）
    pd_flags = read_uint16_le(data, page_offset + 10)
    
    # pd_lower（2バイト、オフセット12） - ページヘッダー+ItemId配列の終端
    pd_lower = read_uint16_le(data, page_offset + 12)
    
    # pd_upper（2バイト、オフセット14） - フリースペースの開始
    pd_upper = read_uint16_le(data, page_offset + 14)
    
    # 基本的な妥当性チェック
    if pd_lower < PAGE_HEADER_SIZE or pd_lower > POSTGRES_PAGE_SIZE:
        return False
    if pd_upper > POSTGRES_PAGE_SIZE or pd_upper < pd_lower:
        return False
    
    return True


@cuda.jit(device=True, inline=True)
def get_item_count(data, page_offset):
    """ページ内のItemId数を取得"""
    if not validate_page_header(data, page_offset):
        return 0
    
    pd_lower = read_uint16_le(data, page_offset + 12)
    
    # ItemId配列のサイズを計算
    item_array_size = pd_lower - PAGE_HEADER_SIZE
    item_count = item_array_size // ITEM_ID_SIZE
    
    return item_count


@cuda.jit(device=True, inline=True)
def parse_item_id(data, page_offset, item_index):
    """指定されたインデックスのItemIdを解析"""
    item_offset = page_offset + PAGE_HEADER_SIZE + (item_index * ITEM_ID_SIZE)
    
    if item_offset + ITEM_ID_SIZE > data.size:
        return 0, 0, LP_UNUSED  # offset, length, flags
    
    # ItemIdData構造体の解析
    # lp_off: 2バイト（タプルデータのオフセット）
    # lp_flags: 2ビット（上位2ビット）
    # lp_len: 14ビット（残りのビット）
    item_data = read_uint32_le(data, item_offset)
    
    lp_off = np.uint16(item_data & np.uint32(0xFFFF))  # 下位16ビット
    lp_flags = np.uint8((item_data >> 30) & np.uint32(0x3))  # 上位2ビット
    lp_len = np.uint16((item_data >> 16) & np.uint32(0x3FFF))  # 中間14ビット
    
    return lp_off, lp_len, lp_flags


@cuda.jit(device=True, inline=True)
def is_tuple_valid(data, page_offset, tuple_offset):
    """タプルの有効性を確認（t_xmaxによる削除チェック）"""
    abs_tuple_offset = page_offset + tuple_offset
    
    if abs_tuple_offset + TUPLE_HEADER_MIN_SIZE > data.size:
        return False
    
    # t_xmaxフィールドを読み取り（TransactionId = uint32）
    t_xmax = read_uint32_le(data, abs_tuple_offset + T_XMAX_OFFSET)
    
    # t_xmaxが0でない場合、タプルは削除されている
    return t_xmax == 0


@cuda.jit
def parse_heap_pages_to_tuples(
    heap_data,           # kvikioで読み込んだ生ヒープデータ
    page_offsets,        # 8KBページの開始位置配列
    tuple_offsets_out,   # 出力：有効タプルオフセット
    tuple_count_out      # 出力：有効タプル数
):
    """
    PostgreSQLヒープページを解析してタプルオフセットを抽出するCUDAカーネル
    
    Args:
        heap_data: kvikioで読み込んだヒープファイルの生データ
        page_offsets: 各ページの開始オフセット配列
        tuple_offsets_out: 有効タプルのオフセットを格納する出力配列
        tuple_count_out: 各ページの有効タプル数を格納する出力配列
    """
    # グリッドとブロックの計算
    page_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if page_idx >= page_offsets.size:
        return
    
    page_offset = page_offsets[page_idx]
    
    # ページヘッダーの妥当性確認
    if not validate_page_header(heap_data, page_offset):
        tuple_count_out[page_idx] = 0
        return
    
    # ItemId数を取得
    item_count = get_item_count(heap_data, page_offset)
    valid_tuple_count = 0
    
    # 出力配列への書き込み開始位置を計算
    # 各ページの最大タプル数を想定して事前に計算された位置
    output_base_idx = page_idx * 226  # 8KB / 36bytes ≈ 226タプル（概算）
    
    # 各ItemIdを解析
    for item_idx in range(item_count):
        lp_off, lp_len, lp_flags = parse_item_id(heap_data, page_offset, item_idx)
        
        # 通常のタプルかどうかチェック
        if lp_flags != LP_NORMAL or lp_off == 0 or lp_len == 0:
            continue
        
        # タプルの境界チェック
        if page_offset + lp_off + lp_len > heap_data.size:
            continue
        
        # タプルの有効性チェック（削除されていないか）
        if not is_tuple_valid(heap_data, page_offset, lp_off):
            continue
        
        # 有効なタプルのオフセットを記録
        if output_base_idx + valid_tuple_count < tuple_offsets_out.size:
            tuple_offsets_out[output_base_idx + valid_tuple_count] = page_offset + lp_off
            valid_tuple_count += 1
    
    # このページの有効タプル数を記録
    tuple_count_out[page_idx] = valid_tuple_count


@cuda.jit
def compact_tuple_offsets(
    sparse_offsets,      # parse_heap_pages_to_tuplesの出力
    sparse_counts,       # 各ページのタプル数
    compact_offsets_out  # 圧縮された出力配列
):
    """
    スパースなタプルオフセット配列を圧縮し、連続した配列に変換
    
    Args:
        sparse_offsets: ページごとに分割されたタプルオフセット配列
        sparse_counts: 各ページの有効タプル数
        compact_offsets_out: 圧縮された連続タプルオフセット配列
    """
    page_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if page_idx >= sparse_counts.size:
        return
    
    # 現在のページまでの累積タプル数を計算
    cumulative_count = 0
    for i in range(page_idx):
        cumulative_count += sparse_counts[i]
    
    # このページのタプルを圧縮配列にコピー
    page_tuple_count = sparse_counts[page_idx]
    source_base = page_idx * 226  # parse_heap_pages_to_tuplesで使用した値と同じ
    
    for i in range(page_tuple_count):
        if cumulative_count + i < compact_offsets_out.size:
            compact_offsets_out[cumulative_count + i] = sparse_offsets[source_base + i]


def create_page_offsets(file_size, page_size=POSTGRES_PAGE_SIZE):
    """
    ファイルサイズから8KBページのオフセット配列を生成
    
    Args:
        file_size: ヒープファイルのサイズ
        page_size: ページサイズ（デフォルト8KB）
    
    Returns:
        cupy.ndarray: ページオフセット配列
    """
    num_pages = file_size // page_size
    return cp.arange(0, num_pages * page_size, page_size, dtype=cp.uint32)


def estimate_max_tuples(num_pages, avg_tuple_size=36):
    """
    最大タプル数の推定（メモリ割り当て用）
    
    Args:
        num_pages: ページ数
        avg_tuple_size: 平均タプルサイズ（バイト）
    
    Returns:
        int: 推定最大タプル数
    """
    usable_space_per_page = POSTGRES_PAGE_SIZE - PAGE_HEADER_SIZE - 100  # ヘッダー + 余裕
    max_tuples_per_page = usable_space_per_page // avg_tuple_size
    return num_pages * max_tuples_per_page


@cuda.jit
def test_simple_kernel(test_array):
    """最小限のテストカーネル"""
    idx = cuda.grid(1)
    if idx < test_array.size:
        test_array[idx] = idx

def parse_heap_file_gpu(heap_data_gpu, debug=False):
    """
    GPUメモリ上のヒープファイルデータを解析してタプルオフセットを抽出
    
    Args:
        heap_data_gpu: kvikioで読み込んだヒープファイルデータ（GPUメモリ上）
        debug: デバッグ情報の出力フラグ
    
    Returns:
        tuple: (tuple_offsets, total_tuple_count)
            - tuple_offsets: 有効タプルのオフセット配列
            - total_tuple_count: 総有効タプル数
    """
    file_size = len(heap_data_gpu)
    
    # CuPy配列をNumba DeviceNDArrayに変換
    page_offsets_cupy = create_page_offsets(file_size)
    page_offsets = cuda.as_cuda_array(page_offsets_cupy)
    num_pages = len(page_offsets)
    
    if debug:
        print(f"ファイルサイズ: {file_size:,} バイト")
        print(f"ページ数: {num_pages:,}")
    
    # 出力配列の準備
    max_tuples = estimate_max_tuples(num_pages)
    sparse_tuple_offsets_cupy = cp.zeros(max_tuples, dtype=cp.uint32)
    tuple_counts_cupy = cp.zeros(num_pages, dtype=cp.uint32)
    
    # CuPy配列をNumba DeviceNDArrayに変換
    sparse_tuple_offsets = cuda.as_cuda_array(sparse_tuple_offsets_cupy)
    tuple_counts = cuda.as_cuda_array(tuple_counts_cupy)
    
    # グリッド設定
    threads_per_block = 256
    blocks = (num_pages + threads_per_block - 1) // threads_per_block
    
    if debug:
        print(f"グリッド設定: {blocks} ブロック x {threads_per_block} スレッド")
    
    # ヒープページ解析カーネル実行
    try:
        # CUDA同期を追加してエラーを詳細に確認
        cuda.synchronize()
        
        if debug:
            print(f"カーネル起動準備: blocks={blocks}, threads_per_block={threads_per_block}")
            print(f"heap_data_gpu.shape: {heap_data_gpu.shape}, dtype: {heap_data_gpu.dtype}")
            print(f"page_offsets.shape: {page_offsets.shape}, dtype: {page_offsets.dtype}")
            print(f"sparse_tuple_offsets.shape: {sparse_tuple_offsets.shape}")
            print(f"tuple_counts.shape: {tuple_counts.shape}")
        
        # ========================= デバッグログ追加 =========================
        if debug:
            print("\n=== CUDAカーネル起動デバッグ情報 ===")
            print(f"blocks = {blocks}, type = {type(blocks)}")
            print(f"threads_per_block = {threads_per_block}, type = {type(threads_per_block)}")
            print(f"hasattr(blocks, 'value') = {hasattr(blocks, 'value') if hasattr(blocks, '__dict__') else 'N/A'}")
            print(f"hasattr(threads_per_block, 'value') = {hasattr(threads_per_block, 'value') if hasattr(threads_per_block, '__dict__') else 'N/A'}")
            
            # CuPy配列の詳細情報
            print(f"\nheap_data_gpu:")
            print(f"  type: {type(heap_data_gpu)}")
            print(f"  shape: {getattr(heap_data_gpu, 'shape', 'N/A')}")
            print(f"  dtype: {getattr(heap_data_gpu, 'dtype', 'N/A')}")
            print(f"  __cuda_array_interface__: {hasattr(heap_data_gpu, '__cuda_array_interface__')}")
            
            print(f"\npage_offsets:")
            print(f"  type: {type(page_offsets)}")
            print(f"  shape: {getattr(page_offsets, 'shape', 'N/A')}")
            print(f"  dtype: {getattr(page_offsets, 'dtype', 'N/A')}")
            print(f"  __cuda_array_interface__: {hasattr(page_offsets, '__cuda_array_interface__')}")
            
            print(f"\nsparse_tuple_offsets:")
            print(f"  type: {type(sparse_tuple_offsets)}")
            print(f"  shape: {getattr(sparse_tuple_offsets, 'shape', 'N/A')}")
            print(f"  dtype: {getattr(sparse_tuple_offsets, 'dtype', 'N/A')}")
            
            print(f"\ntuple_counts:")
            print(f"  type: {type(tuple_counts)}")
            print(f"  shape: {getattr(tuple_counts, 'shape', 'N/A')}")
            print(f"  dtype: {getattr(tuple_counts, 'dtype', 'N/A')}")
            
            # Numbaカーネル関数の情報
            print(f"\nparse_heap_pages_to_tuples:")
            print(f"  type: {type(parse_heap_pages_to_tuples)}")
            print(f"  __name__: {getattr(parse_heap_pages_to_tuples, '__name__', 'N/A')}")
            
            # CUDAコンテキスト情報
            try:
                current_device = cuda.current_context().device
                print(f"\nCUDA context: {current_device}")
            except Exception as cuda_err:
                print(f"\nCUDA context error: {cuda_err}")
        
        # 最小限のテストカーネルで基本的な動作確認
        if debug:
            print("\n=== 最小限テストカーネル実行 ===")
            try:
                test_array = cp.zeros(10, dtype=cp.int32)
                test_blocks = 1
                test_threads = 10
                
                print(f"テスト配列: {test_array.shape}, dtype: {test_array.dtype}")
                print(f"テストグリッド: blocks={test_blocks}, threads={test_threads}")
                
                # tuple形式でテスト
                test_simple_kernel[(test_blocks,), (test_threads,)](test_array)
                cuda.synchronize()
                
                result = test_array.get()
                print(f"✓ 最小限テストカーネル成功: {result[:5]}...")
                
            except Exception as test_error:
                print(f"❌ 最小限テストカーネルエラー: {test_error}")
                print(f"❌ テストエラー型: {type(test_error).__name__}")
                import traceback
                print(f"❌ テストスタックトレース:\n{traceback.format_exc()}")
        
        # 修正されたカーネル起動（Numba DeviceNDArray統一）
        try:
            if debug:
                print("\n=== 修正されたカーネル起動 ===")
            
            # 全ての配列がNumba DeviceNDArrayに統一されているため、直接起動
            parse_heap_pages_to_tuples[blocks, threads_per_block](
                heap_data_gpu, page_offsets, sparse_tuple_offsets, tuple_counts
            )
            
            if debug:
                print("✓ カーネル起動成功")
                
        except Exception as kernel_error:
            if debug:
                print(f"❌ カーネル起動エラー: {kernel_error}")
                print(f"❌ エラー型: {type(kernel_error).__name__}")
                import traceback
                print(f"❌ スタックトレース:\n{traceback.format_exc()}")
            
            # フォールバック: 空データで継続
            print("⚠️  カーネル起動に失敗しました。空のデータで継続します。")
            tuple_counts_cupy[:] = 0  # 全ページに0個のタプル
        
        cuda.synchronize()
        if debug:
            print("✓ ヒープページ解析カーネル実行完了")
    except Exception as e:
        if debug:
            print(f"❌ ヒープページ解析カーネル実行エラー: {e}")
            print(f"❌ エラー詳細: {type(e).__name__}: {str(e)}")
        raise
    
    # 結果を圧縮
    try:
        # Numba DeviceNDArrayをCuPy配列に変換してから集計
        tuple_counts_cupy = cp.asarray(tuple_counts)
        tuple_sum = cp.sum(tuple_counts_cupy)
        if debug:
            print(f"Debug: tuple_sum = {tuple_sum}, type = {type(tuple_sum)}")
        
        # CuPyのスカラー値を確実にPythonのintに変換
        if hasattr(tuple_sum, 'item'):
            total_tuple_count = int(tuple_sum.item())
            if debug:
                print(f"Debug: total_tuple_count (via .item()) = {total_tuple_count}")
        else:
            total_tuple_count = int(tuple_sum)
            if debug:
                print(f"Debug: total_tuple_count (direct int()) = {total_tuple_count}")
    except Exception as e:
        if debug:
            print(f"❌ タプル数計算エラー: {e}")
        raise
    
    if total_tuple_count == 0:
        if debug:
            print("警告: 有効なタプルが見つかりませんでした")
        return cp.array([], dtype=cp.uint32), 0
    
    compact_offsets_out_cupy = cp.zeros(total_tuple_count, dtype=cp.uint32)
    compact_offsets_out = cuda.as_cuda_array(compact_offsets_out_cupy)
    
    # 圧縮カーネル実行
    compact_tuple_offsets[blocks, threads_per_block](
        sparse_tuple_offsets, tuple_counts, compact_offsets_out
    )
    
    if debug:
        print(f"有効タプル数: {total_tuple_count:,}")
        print(f"平均タプル密度: {total_tuple_count/num_pages:.2f} タプル/ページ")
    
    # 結果をCuPy配列として返す
    return cp.asarray(compact_offsets_out), total_tuple_count