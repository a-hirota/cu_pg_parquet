#!/usr/bin/env python
"""RMM 大規模アロケーションテスト"""
import rmm
import cupy as cp
from numba import cuda
import numpy as np

print("=== RMM 大規模アロケーションテスト ===")

# RMM 22GB初期化
print("\n1. RMM 22GB初期化...")
try:
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=22*1024**3,  # 22GB
        maximum_pool_size=22*1024**3   # 22GB
    )
    print("✅ 22GB初期化成功")
except Exception as e:
    print(f"❌ 初期化エラー: {e}")
    exit(1)

# 段階的にメモリ確保テスト
test_sizes = [1, 5, 10, 13, 15, 20]  # GB

for size_gb in test_sizes:
    size_bytes = size_gb * 1024**3
    print(f"\n2. {size_gb}GB確保テスト...")
    try:
        buffer = rmm.DeviceBuffer(size=size_bytes)
        print(f"✅ {size_gb}GB確保成功")
        
        # CUDAカーネル実行テスト
        print(f"   CUDAカーネル実行テスト...")
        @cuda.jit
        def simple_kernel(data):
            idx = cuda.grid(1)
            if idx < data.size:
                data[idx] = idx
        
        # numba配列として使用
        arr = cuda.as_cuda_array(buffer).view(dtype=np.uint8)
        simple_kernel[1024, 256](arr[:1024*256])
        cuda.synchronize()
        print(f"   ✅ CUDAカーネル実行成功")
        
        # メモリ解放
        del buffer
        del arr
        cuda.synchronize()
        print(f"   ✅ メモリ解放完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        break

print("\n=== テスト完了 ===")