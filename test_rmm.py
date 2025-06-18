#!/usr/bin/env python
"""RMM メモリプール状況確認スクリプト"""
import rmm
import cupy as cp
from numba import cuda

print("=== RMM/GPU メモリ状況確認 ===")

# RMM状況
if rmm.is_initialized():
    print("RMM: 初期化済み")
    try:
        mr = rmm.mr.get_current_device_resource()
        print(f"現在のメモリリソース: {mr}")
    except:
        pass
else:
    print("RMM: 未初期化")

# GPU情報
device = cuda.get_current_device()
print(f"\nGPUデバイス: {device.name}")

# CuPyメモリプール
mempool = cp.get_default_memory_pool()
print(f"\nCuPy メモリプール:")
print(f"  使用中: {mempool.used_bytes() / (1024**3):.2f} GB")
print(f"  合計: {mempool.total_bytes() / (1024**3):.2f} GB")

# 強制的に22GBで再初期化
print("\n=== RMM 22GB再初期化テスト ===")
try:
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=22*1024**3,  # 22GB
        maximum_pool_size=22*1024**3   # 22GB
    )
    print("✅ 22GB再初期化成功")
    
    # 大きなバッファ確保テスト
    test_size = 10 * 1024**3  # 10GB
    print(f"\n10GBバッファ確保テスト...")
    buffer = rmm.DeviceBuffer(size=test_size)
    print("✅ 10GB確保成功")
    del buffer
    
except Exception as e:
    print(f"❌ エラー: {e}")