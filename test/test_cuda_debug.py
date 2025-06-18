"""
CUDA環境デバッグテスト

CUDA環境の詳細な状態を確認し、問題を特定する
"""

import sys
import traceback

def test_basic_imports():
    """基本ライブラリのインポートテスト"""
    print("=== 基本ライブラリインポートテスト ===")
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy インポートエラー: {e}")
        return False
    
    try:
        import cupy as cp
        print(f"✓ CuPy: {cp.__version__}")
    except Exception as e:
        print(f"✗ CuPy インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    try:
        from numba import cuda
        print(f"✓ Numba CUDA インポート成功")
    except Exception as e:
        print(f"✗ Numba CUDA インポートエラー: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_cupy_basic():
    """CuPy基本機能テスト"""
    print("\n=== CuPy基本機能テスト ===")
    
    try:
        import cupy as cp
        
        # メモリプール無効化（互換性問題回避）
        mempool = cp.get_default_memory_pool()
        print(f"メモリプール使用量: {mempool.used_bytes()} bytes")
        
        # 基本的な配列操作
        print("基本配列作成テスト...")
        a = cp.array([1, 2, 3])
        print(f"✓ 配列作成成功: {a}")
        
        # 算術演算
        print("算術演算テスト...")
        b = a + 1
        print(f"✓ 算術演算成功: {b}")
        
        # GPU->CPU転送
        print("GPU->CPU転送テスト...")
        c = b.get()
        print(f"✓ データ転送成功: {c}")
        
        return True
    except Exception as e:
        print(f"✗ CuPy基本機能エラー: {e}")
        traceback.print_exc()
        return False

def test_numba_cuda():
    """Numba CUDA基本テスト"""
    print("\n=== Numba CUDA基本テスト ===")
    
    try:
        from numba import cuda
        
        # デバイス情報
        print("CUDA デバイス情報:")
        print(f"  利用可能: {cuda.is_available()}")
        if cuda.is_available():
            print(f"  デバイス数: {cuda.gpus.count}")
            for i, gpu in enumerate(cuda.gpus):
                print(f"  GPU {i}: {gpu.name}")
        
        # 簡単なカーネルテスト
        @cuda.jit
        def simple_kernel(arr):
            idx = cuda.grid(1)
            if idx < len(arr):
                arr[idx] = idx * 2
        
        # テストデータ
        import cupy as cp
        test_data = cp.zeros(10, dtype=cp.int32)
        
        # カーネル実行
        simple_kernel[1, 10](test_data)
        result = test_data.get()
        
        print(f"✓ 簡単なカーネル実行成功: {result}")
        return True
        
    except Exception as e:
        print(f"✗ Numba CUDA エラー: {e}")
        traceback.print_exc()
        return False

def test_project_import():
    """プロジェクトモジュールインポートテスト"""
    print("\n=== プロジェクトモジュールインポートテスト ===")
    
    try:
        # 段階的にインポート
        print("src モジュールインポート...")
        import src
        print("✓ src インポート成功")
        
        print("src.cuda_kernels インポート...")
        import src.cuda_kernels
        print("✓ src.cuda_kernels インポート成功")
        
        print("heap_page_parser インポート...")
        from src.cuda_kernels import heap_page_parser
        print("✓ heap_page_parser インポート成功")
        
        print("個別関数インポート...")
        from src.cuda_kernels.heap_page_parser import parse_heap_pages_to_tuples
        print("✓ parse_heap_pages_to_tuples インポート成功")
        
        return True
    except Exception as e:
        print(f"✗ プロジェクトモジュールエラー: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("CUDA環境デバッグテスト")
    print("=" * 50)
    print(f"Python バージョン: {sys.version}")
    print()
    
    tests = [
        test_basic_imports,
        test_cupy_basic,
        test_numba_cuda,
        test_project_import
    ]
    
    for test_func in tests:
        if not test_func():
            print(f"\n✗ {test_func.__name__} で失敗")
            break
    else:
        print("\n" + "=" * 50)
        print("✓ すべてのデバッグテスト完了")