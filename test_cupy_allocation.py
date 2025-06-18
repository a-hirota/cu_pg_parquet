#!/usr/bin/env python
"""CuPy配列による大規模メモリ確保・解放テスト"""
import cupy as cp
import numpy as np
from numba import cuda
import gc
import time

print("=== CuPy配列 大規模アロケーションテスト ===")

# テストサイズ（GB）
test_sizes = [1, 5, 10, 13, 13, 13]  # 13GBを3回連続で確保・解放

for i, size_gb in enumerate(test_sizes):
    size_bytes = size_gb * 1024**3
    print(f"\n{i+1}. {size_gb}GB確保テスト...")
    
    try:
        # メモリ使用状況（確保前）
        mempool = cp.get_default_memory_pool()
        used_before = mempool.used_bytes() / (1024**3)
        print(f"   確保前メモリ使用: {used_before:.2f} GB")
        
        # CuPy配列として確保
        start_time = time.time()
        gpu_array = cp.zeros(size_bytes, dtype=cp.uint8)
        alloc_time = time.time() - start_time
        print(f"✅ {size_gb}GB確保成功 ({alloc_time:.3f}秒)")
        
        # メモリポインタ取得
        gpu_ptr = gpu_array.data.ptr
        print(f"   GPUメモリアドレス: 0x{gpu_ptr:016x}")
        
        # CUDAカーネル実行テスト
        print(f"   CUDAカーネル実行テスト...")
        @cuda.jit
        def simple_kernel(data):
            idx = cuda.grid(1)
            if idx < data.size:
                data[idx] = idx % 256
        
        # Numbaで使用できる配列に変換
        numba_array = cuda.as_cuda_array(gpu_array)
        
        # カーネル実行
        threads = 256
        blocks = min((size_bytes + threads - 1) // threads, 65535)
        simple_kernel[blocks, threads](numba_array)
        cuda.synchronize()
        print(f"   ✅ CUDAカーネル実行成功")
        
        # データ検証（最初の10要素）
        result = gpu_array[:10].get()
        print(f"   データ検証: {result}")
        
        # メモリ使用状況（確保後）
        used_after = mempool.used_bytes() / (1024**3)
        print(f"   確保後メモリ使用: {used_after:.2f} GB")
        
        # メモリ解放
        print(f"   メモリ解放中...")
        del gpu_array
        del numba_array
        gpu_array = None
        numba_array = None
        
        # ガベージコレクション強制実行
        gc.collect()
        
        # メモリプールをクリア
        mempool.free_all_blocks()
        
        # CUDA同期
        cuda.synchronize()
        
        # 解放後のメモリ使用状況
        time.sleep(0.1)  # 少し待つ
        used_after_free = mempool.used_bytes() / (1024**3)
        print(f"   ✅ メモリ解放完了")
        print(f"   解放後メモリ使用: {used_after_free:.2f} GB")
        
        if used_after_free > 0.1:  # 100MB以上残っている場合
            print(f"   ⚠️  警告: メモリが完全に解放されていない可能性")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        # エラー時もメモリクリアを試みる
        mempool.free_all_blocks()
        gc.collect()
        cuda.synchronize()

# 最終メモリ状況
print("\n=== 最終メモリ状況 ===")
mempool = cp.get_default_memory_pool()
print(f"メモリプール使用量: {mempool.used_bytes() / (1024**3):.2f} GB")
print(f"メモリプール合計: {mempool.total_bytes() / (1024**3):.2f} GB")

print("\n=== テスト完了 ===")