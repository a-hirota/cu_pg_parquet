#!/usr/bin/env python3
"""
CUDAカーネル起動エラーのデバッグテスト

「'int' object has no attribute 'value'」エラーの根本原因を特定するための
最小限のテストスクリプト。
"""

import os
import sys
import cupy as cp
import numpy as np
from numba import cuda

# プロジェクトパスを追加
sys.path.insert(0, '/home/ubuntu/gpupgparser')

def test_basic_cuda_setup():
    """基本的なCUDA環境のテスト"""
    print("=== 基本CUDA環境テスト ===")
    
    try:
        # CUDA初期化確認
        cuda.current_context()
        print("✅ CUDA context OK")
        
        # デバイス情報
        device = cuda.get_current_device()
        print(f"✅ CUDA device: {device.name}")
        
        return True
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        return False

def test_cupy_array_creation():
    """CuPy配列作成のテスト"""
    print("\n=== CuPy配列作成テスト ===")
    
    try:
        # 基本的なCuPy配列
        test_array = cp.zeros(100, dtype=cp.uint8)
        print(f"✅ CuPy配列作成成功: shape={test_array.shape}, dtype={test_array.dtype}")
        
        # CUDA Array Interface確認
        if hasattr(test_array, '__cuda_array_interface__'):
            print("✅ __cuda_array_interface__ 対応")
        else:
            print("❌ __cuda_array_interface__ 未対応")
        
        return test_array
    except Exception as e:
        print(f"❌ CuPy配列作成失敗: {e}")
        return None

def test_numba_kernel_basic():
    """基本的なNumbaカーネルのテスト"""
    print("\n=== 基本Numbaカーネルテスト ===")
    
    try:
        @cuda.jit
        def simple_kernel(arr):
            idx = cuda.grid(1)
            if idx < arr.size:
                arr[idx] = idx
        
        # CuPy配列で実行
        test_array = cp.zeros(10, dtype=cp.int32)
        blocks = 1
        threads = 10
        
        print(f"実行前: blocks={blocks} ({type(blocks)}), threads={threads} ({type(threads)})")
        print(f"配列: {test_array.shape}, {test_array.dtype}")
        
        # カーネル起動
        simple_kernel[(blocks,), (threads,)](test_array)
        cuda.synchronize()
        
        result = test_array.get()
        print(f"✅ 基本カーネル成功: {result}")
        
        return True
    except Exception as e:
        print(f"❌ 基本カーネル失敗: {e}")
        import traceback
        print(f"スタックトレース:\n{traceback.format_exc()}")
        return False

def test_heap_parser_import():
    """ヒープパーサーのインポートテスト"""
    print("\n=== ヒープパーサーインポートテスト ===")
    
    try:
        from src.cuda_kernels.heap_page_parser import parse_heap_file_gpu, test_simple_kernel
        print("✅ ヒープパーサーインポート成功")
        
        # テストカーネルの実行
        test_array = cp.zeros(10, dtype=cp.int32)
        blocks = 1
        threads = 10
        
        print(f"テストカーネル実行: blocks={blocks}, threads={threads}")
        test_simple_kernel[(blocks,), (threads,)](test_array)
        cuda.synchronize()
        
        result = test_array.get()
        print(f"✅ テストカーネル成功: {result}")
        
        return True
    except Exception as e:
        print(f"❌ ヒープパーサーテスト失敗: {e}")
        import traceback
        print(f"スタックトレース:\n{traceback.format_exc()}")
        return False

def test_kvikio_pipeline_minimal():
    """最小限のkvikioパイプラインテスト"""
    print("\n=== 最小限kvikioパイプラインテスト ===")
    
    try:
        # ダミーのヒープデータを作成
        dummy_heap_data = cp.zeros(8192, dtype=cp.uint8)  # 1ページ分
        
        from src.cuda_kernels.heap_page_parser import parse_heap_file_gpu
        
        print("parse_heap_file_gpu呼び出し開始（debug=True）...")
        tuple_offsets, total_count = parse_heap_file_gpu(dummy_heap_data, debug=True)
        
        print(f"✅ parse_heap_file_gpu成功: {total_count}個のタプル")
        return True
        
    except Exception as e:
        print(f"❌ kvikioパイプラインテスト失敗: {e}")
        import traceback
        print(f"スタックトレース:\n{traceback.format_exc()}")
        return False

def main():
    """メインテスト実行"""
    print("CUDAカーネル起動エラーのデバッグテスト開始\n")
    
    # 段階的テスト実行
    tests = [
        ("基本CUDA環境", test_basic_cuda_setup),
        ("CuPy配列作成", test_cupy_array_creation),
        ("基本Numbaカーネル", test_numba_kernel_basic),
        ("ヒープパーサーインポート", test_heap_parser_import),
        ("最小限kvikioパイプライン", test_kvikio_pipeline_minimal),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "成功" if result else "失敗"
        except Exception as e:
            results[test_name] = f"例外: {e}"
        
        print()  # 改行
    
    # 結果サマリー
    print("=== テスト結果サマリー ===")
    for test_name, result in results.items():
        status = "✅" if result == "成功" else "❌"
        print(f"{status} {test_name}: {result}")
    
    # 失敗があった場合の推奨アクション
    failed_tests = [name for name, result in results.items() if result != "成功"]
    if failed_tests:
        print(f"\n⚠️  失敗したテスト: {', '.join(failed_tests)}")
        print("推奨アクション:")
        print("1. CUDA環境の確認（nvidia-smi, nvcc --version）")
        print("2. NumbaとCuPyのバージョン確認")
        print("3. 環境変数の確認（CUDA_PATH, LD_LIBRARY_PATH）")
    else:
        print("\n🎉 全テスト成功！")

if __name__ == "__main__":
    main()