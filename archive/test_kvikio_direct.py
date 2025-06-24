#!/usr/bin/env python3
"""
kvikio直接転送テスト
/dev/shmからGPUへの直接転送を検証
"""

import os
import time
import numpy as np
import cupy as cp
import kvikio
from numba import cuda
import rmm

def test_traditional_method(file_path, file_size):
    """従来方法: CPU経由でGPU転送"""
    print("\n=== 従来方法: CPU経由 ===")
    
    # ファイル読み込み（CPUメモリへ）
    read_start = time.time()
    with open(file_path, 'rb') as f:
        data = f.read()
    raw_host = np.frombuffer(data, dtype=np.uint8)
    del data
    read_time = time.time() - read_start
    
    # GPU転送
    transfer_start = time.time()
    raw_dev = cuda.to_device(raw_host)
    transfer_time = time.time() - transfer_start
    
    total_time = read_time + transfer_time
    throughput = file_size / total_time / 1024**3
    
    print(f"  ファイル読み込み: {read_time:.4f}秒")
    print(f"  GPU転送: {transfer_time:.4f}秒")
    print(f"  合計時間: {total_time:.4f}秒")
    print(f"  スループット: {throughput:.2f} GB/秒")
    
    return raw_dev, total_time

def test_kvikio_direct(file_path, file_size):
    """kvikio: GPUDirect転送"""
    print("\n=== kvikio: GPUDirect ===")
    
    # kvikioの設定確認（新しいAPI）
    try:
        # 互換モード確認
        is_compat = os.environ.get("KVIKIO_COMPAT_MODE", "").lower() in ["on", "1", "true"]
        print(f"  KVIKIO_COMPAT_MODE: {is_compat}")
    except:
        pass
    
    # GPU用バッファ確保
    gpu_buffer = cp.empty(file_size, dtype=cp.uint8)
    
    # kvikioで直接読み込み
    transfer_start = time.time()
    try:
        with kvikio.CuFile(file_path, "rb") as f:
            # read: ファイル全体を読み込み
            bytes_read = f.read(gpu_buffer)
    except Exception as e:
        print(f"  kvikio.CuFile エラー: {e}")
        # フォールバック: 通常の読み込み
        with open(file_path, 'rb') as f:
            data = f.read()
        gpu_buffer[:] = cp.asarray(np.frombuffer(data, dtype=np.uint8))
        bytes_read = len(data)
        del data
    transfer_time = time.time() - transfer_start
    
    throughput = file_size / transfer_time / 1024**3
    
    print(f"  読み込みバイト数: {bytes_read:,}")
    print(f"  転送時間: {transfer_time:.4f}秒")
    print(f"  スループット: {throughput:.2f} GB/秒")
    
    # numba cuda配列に変換（ゼロコピー）
    raw_dev = cuda.as_cuda_array(gpu_buffer)
    
    return raw_dev, transfer_time

def test_kvikio_rmm(file_path, file_size):
    """kvikio + RMM: より効率的な転送"""
    print("\n=== kvikio + RMM ===")
    
    # RMM DeviceBufferを使用
    gpu_buffer = rmm.DeviceBuffer(size=file_size)
    
    # kvikioで直接読み込み
    transfer_start = time.time()
    try:
        with kvikio.CuFile(file_path, "rb") as f:
            # RMM DeviceBufferはCuPy配列にラップする必要がある
            gpu_array = cp.asarray(gpu_buffer).view(dtype=cp.uint8)
            bytes_read = f.read(gpu_array)
    except Exception as e:
        print(f"  kvikio.CuFile エラー: {e}")
        # フォールバック
        with open(file_path, 'rb') as f:
            data = f.read()
        gpu_buffer.copy_from_host(data)
        bytes_read = len(data)
        del data
    transfer_time = time.time() - transfer_start
    
    throughput = file_size / transfer_time / 1024**3
    
    print(f"  読み込みバイト数: {bytes_read:,}")
    print(f"  転送時間: {transfer_time:.4f}秒")
    print(f"  スループット: {throughput:.2f} GB/秒")
    
    # numba cuda配列に変換（ゼロコピー）
    raw_dev = cuda.as_cuda_array(gpu_buffer).view(dtype=np.uint8)
    
    return raw_dev, transfer_time

def main():
    print("=== kvikio直接転送テスト ===")
    
    # テストデータ作成（100MB）
    test_size = 100 * 1024 * 1024
    test_file = "/dev/shm/kvikio_test.bin"
    
    print(f"\nテストファイル作成中: {test_size / 1024**2:.0f} MB")
    test_data = np.random.randint(0, 256, test_size, dtype=np.uint8)
    with open(test_file, 'wb') as f:
        f.write(test_data.tobytes())
    
    # RMM初期化
    if not rmm.is_initialized():
        rmm.reinitialize(pool_allocator=True, initial_pool_size=2*1024**3)
    
    try:
        # 1. 従来方法テスト
        dev1, time1 = test_traditional_method(test_file, test_size)
        del dev1
        
        # 2. kvikio直接転送テスト
        dev2, time2 = test_kvikio_direct(test_file, test_size)
        del dev2
        
        # 3. kvikio + RMM テスト
        dev3, time3 = test_kvikio_rmm(test_file, test_size)
        del dev3
        
        # 結果比較
        print("\n=== 結果比較 ===")
        print(f"従来方法: {time1:.4f}秒")
        print(f"kvikio直接: {time2:.4f}秒 ({time1/time2:.1f}倍高速)")
        print(f"kvikio+RMM: {time3:.4f}秒 ({time1/time3:.1f}倍高速)")
        
        # GPUDirect有効性確認
        is_compat = os.environ.get("KVIKIO_COMPAT_MODE", "").lower() in ["on", "1", "true"]
        if is_compat:
            print("\n⚠️  互換モードで動作中（GPUDirect未使用）")
            print("GPUDirectを有効にするには:")
            print("  export KVIKIO_COMPAT_MODE=OFF")
            print("  nvidia-fsドライバが必要です")
        else:
            print("\n✅ GPUDirect Storage有効の可能性")
            print("  nvidia-fsドライバの状態を確認してください:")
            print("  lsmod | grep nvidia_fs")
            
    finally:
        # テストファイル削除
        if os.path.exists(test_file):
            os.remove(test_file)
        print("\nテスト完了")

if __name__ == "__main__":
    main()