"""
ヒープページパーサーCUDAカーネルのテスト

PostgreSQLヒープページ解析カーネルの基本的な動作確認を行う。
実際のヒープファイルがない場合のためのモックデータテスト。
"""

import numpy as np
import cupy as cp
from numba import cuda
import pytest

# プロジェクトのCUDAカーネルをインポート
try:
    from src.cuda_kernels.heap_page_parser import (
        parse_heap_pages_to_tuples,
        compact_tuple_offsets,
        parse_heap_file_gpu,
        create_page_offsets,
        estimate_max_tuples,
        POSTGRES_PAGE_SIZE,
        PAGE_HEADER_SIZE,
        ITEM_ID_SIZE,
        LP_NORMAL,
        LP_UNUSED
    )
    print("✓ ヒープページパーサーモジュールのインポートに成功")
except ImportError as e:
    print(f"✗ インポートエラー: {e}")
    raise


def create_mock_heap_page():
    """
    テスト用のモックヒープページデータを作成
    
    Returns:
        numpy.ndarray: 模擬ヒープページデータ（8KB）
    """
    page_data = np.zeros(POSTGRES_PAGE_SIZE, dtype=np.uint8)
    
    # PageHeaderData構造体の設定
    # pd_lsn (8バイト) - LSN番号
    page_data[0:8] = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    
    # pd_checksum (2バイト) - チェックサム
    page_data[8:10] = [0x00, 0x00]
    
    # pd_flags (2バイト) - フラグ
    page_data[10:12] = [0x00, 0x00]
    
    # pd_lower (2バイト) - ItemId配列の終端（ページヘッダー + 3つのItemId）
    pd_lower = PAGE_HEADER_SIZE + (3 * ITEM_ID_SIZE)  # 24 + 12 = 36
    page_data[12:14] = [pd_lower & 0xFF, (pd_lower >> 8) & 0xFF]
    
    # pd_upper (2バイト) - フリースペースの開始
    pd_upper = POSTGRES_PAGE_SIZE - 200  # データ領域の開始
    page_data[14:16] = [pd_upper & 0xFF, (pd_upper >> 8) & 0xFF]
    
    # pd_special (2バイト) - 特殊領域へのポインタ
    page_data[16:18] = [0x00, 0x20]  # POSTGRES_PAGE_SIZE
    
    # pd_pagesize_version (2バイト) - ページサイズとバージョン
    page_data[18:20] = [0x00, 0x20]  # 8192
    
    # pd_prune_xid (4バイト) - プルーニングXID
    page_data[20:24] = [0x00, 0x00, 0x00, 0x00]
    
    # ItemId配列の設定（3つのアイテム）
    item_offset = PAGE_HEADER_SIZE
    
    # ItemId 1: 有効なタプル
    tuple1_offset = pd_upper
    tuple1_length = 50
    item1_data = tuple1_offset | ((tuple1_length & 0x3FFF) << 16) | (LP_NORMAL << 30)
    page_data[item_offset:item_offset+4] = [
        item1_data & 0xFF,
        (item1_data >> 8) & 0xFF,
        (item1_data >> 16) & 0xFF,
        (item1_data >> 24) & 0xFF
    ]
    
    # ItemId 2: 有効なタプル
    tuple2_offset = pd_upper + 60
    tuple2_length = 45
    item2_data = tuple2_offset | ((tuple2_length & 0x3FFF) << 16) | (LP_NORMAL << 30)
    page_data[item_offset+4:item_offset+8] = [
        item2_data & 0xFF,
        (item2_data >> 8) & 0xFF,
        (item2_data >> 16) & 0xFF,
        (item2_data >> 24) & 0xFF
    ]
    
    # ItemId 3: 未使用
    page_data[item_offset+8:item_offset+12] = [0x00, 0x00, 0x00, 0x00]
    
    # タプルデータの設定
    # タプル1のヘッダー（HeapTupleHeader）
    tuple1_start = tuple1_offset
    # t_xmin (4バイト) - 作成トランザクションID
    page_data[tuple1_start:tuple1_start+4] = [0x01, 0x00, 0x00, 0x00]
    # t_xmax (4バイト) - 削除トランザクションID（0=有効）
    page_data[tuple1_start+4:tuple1_start+8] = [0x00, 0x00, 0x00, 0x00]
    # t_field3 (4バイト) - ctid等
    page_data[tuple1_start+8:tuple1_start+12] = [0x00, 0x00, 0x00, 0x00]
    
    # タプル2のヘッダー
    tuple2_start = tuple2_offset
    # t_xmin (4バイト)
    page_data[tuple2_start:tuple2_start+4] = [0x02, 0x00, 0x00, 0x00]
    # t_xmax (4バイト) - 削除済み（非ゼロ）
    page_data[tuple2_start+4:tuple2_start+8] = [0x03, 0x00, 0x00, 0x00]
    # t_field3 (4バイト)
    page_data[tuple2_start+8:tuple2_start+12] = [0x00, 0x00, 0x01, 0x00]
    
    return page_data


def test_cuda_kernel_compilation():
    """CUDAカーネルのコンパイルテスト"""
    print("\n=== CUDAカーネルコンパイルテスト ===")
    
    try:
        # ダミーデータでカーネルをコンパイル
        dummy_heap = cp.zeros(POSTGRES_PAGE_SIZE, dtype=cp.uint8)
        dummy_offsets = cp.array([0], dtype=cp.uint32)
        dummy_tuple_out = cp.zeros(100, dtype=cp.uint32)
        dummy_count_out = cp.zeros(1, dtype=cp.uint32)
        
        # カーネルのコンパイルを試行
        parse_heap_pages_to_tuples[1, 1](
            dummy_heap, dummy_offsets, dummy_tuple_out, dummy_count_out
        )
        print("✓ parse_heap_pages_to_tuples カーネルのコンパイルに成功")
        
        compact_tuple_offsets[1, 1](
            dummy_tuple_out, dummy_count_out, dummy_tuple_out[:50]
        )
        print("✓ compact_tuple_offsets カーネルのコンパイルに成功")
        
    except Exception as e:
        print(f"✗ カーネルコンパイルエラー: {e}")
        raise


def test_mock_heap_page_parsing():
    """モックヒープページの解析テスト"""
    print("\n=== モックヒープページ解析テスト ===")
    
    # モックページデータを作成
    mock_page = create_mock_heap_page()
    print(f"モックページサイズ: {len(mock_page)} バイト")
    
    # GPUメモリにコピー
    heap_data_gpu = cp.asarray(mock_page)
    page_offsets = cp.array([0], dtype=cp.uint32)
    
    # 出力配列の準備
    max_tuples = 10
    tuple_offsets_out = cp.zeros(max_tuples, dtype=cp.uint32)
    tuple_count_out = cp.zeros(1, dtype=cp.uint32)
    
    try:
        # カーネル実行
        parse_heap_pages_to_tuples[1, 1](
            heap_data_gpu, page_offsets, tuple_offsets_out, tuple_count_out
        )
        
        # 結果の確認
        tuple_count = int(tuple_count_out[0])
        print(f"検出された有効タプル数: {tuple_count}")
        
        if tuple_count > 0:
            valid_offsets = tuple_offsets_out[:tuple_count].get()
            print(f"有効タプルオフセット: {valid_offsets}")
            
            # 期待値の確認（タプル1は有効、タプル2は削除済み）
            expected_valid_tuples = 1  # タプル1のみ有効
            if tuple_count == expected_valid_tuples:
                print("✓ 期待通りの結果：削除済みタプルが正しく除外されました")
            else:
                print(f"⚠ 期待値と異なる結果：期待{expected_valid_tuples}、実際{tuple_count}")
        else:
            print("⚠ 有効なタプルが検出されませんでした")
            
    except Exception as e:
        print(f"✗ カーネル実行エラー: {e}")
        raise


def test_utility_functions():
    """ユーティリティ関数のテスト"""
    print("\n=== ユーティリティ関数テスト ===")
    
    try:
        # create_page_offsets のテスト
        file_size = POSTGRES_PAGE_SIZE * 3  # 3ページ
        offsets = create_page_offsets(file_size)
        expected_offsets = [0, 8192, 16384]
        
        if np.array_equal(offsets.get(), expected_offsets):
            print("✓ create_page_offsets: 正しいページオフセットが生成されました")
        else:
            print(f"✗ create_page_offsets: 期待値{expected_offsets}、実際{offsets.get()}")
        
        # estimate_max_tuples のテスト
        num_pages = 10
        max_tuples = estimate_max_tuples(num_pages, avg_tuple_size=40)
        expected_range = (1500, 2500)  # 概算範囲
        
        if expected_range[0] <= max_tuples <= expected_range[1]:
            print(f"✓ estimate_max_tuples: 妥当な推定値 {max_tuples}")
        else:
            print(f"⚠ estimate_max_tuples: 推定値が範囲外 {max_tuples}")
            
    except Exception as e:
        print(f"✗ ユーティリティ関数エラー: {e}")
        raise


def test_high_level_api():
    """高レベルAPI（parse_heap_file_gpu）のテスト"""
    print("\n=== 高レベルAPIテスト ===")
    
    try:
        # 複数ページのモックデータを作成
        num_pages = 2
        full_heap_data = np.concatenate([
            create_mock_heap_page() for _ in range(num_pages)
        ])
        
        heap_data_gpu = cp.asarray(full_heap_data)
        print(f"テストデータサイズ: {len(full_heap_data):,} バイト ({num_pages} ページ)")
        
        # 高レベルAPI実行
        tuple_offsets, total_count = parse_heap_file_gpu(heap_data_gpu, debug=True)
        
        print(f"解析結果:")
        print(f"  総有効タプル数: {total_count}")
        print(f"  オフセット配列サイズ: {len(tuple_offsets)}")
        
        if total_count > 0:
            print(f"  最初の5つのオフセット: {tuple_offsets[:min(5, total_count)].get()}")
            print("✓ 高レベルAPI実行成功")
        else:
            print("⚠ 有効なタプルが検出されませんでした")
            
    except Exception as e:
        print(f"✗ 高レベルAPIエラー: {e}")
        raise


if __name__ == "__main__":
    print("PostgreSQL ヒープページパーサー CUDAカーネル テスト")
    print("=" * 60)
    
    try:
        # CUDAデバイスの確認
        if not cuda.is_available():
            print("✗ CUDAが利用できません")
            exit(1)
        
        print(f"✓ CUDA利用可能 (デバイス数: {cuda.gpus.count})")
        
        # 各テストの実行
        test_cuda_kernel_compilation()
        test_mock_heap_page_parsing()
        test_utility_functions()
        test_high_level_api()
        
        print("\n" + "=" * 60)
        print("✓ すべてのテストが完了しました")
        
    except Exception as e:
        print(f"\n✗ テスト実行中にエラーが発生: {e}")
        exit(1)