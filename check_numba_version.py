import numba
import numba.cuda
import sys

print("=== Numba・CUDA環境情報確認 ===")
print(f"Python version: {sys.version}")
print(f"Numba version: {numba.__version__}")

try:
    print(f"Numba-CUDA version: {numba.cuda.__version__}")
except:
    print("Numba-CUDA version: 情報取得不可")

print(f"CUDA利用可能: {numba.cuda.is_available()}")

if numba.cuda.is_available():
    try:
        print(f"CUDA GPUs検出数: {len(numba.cuda.gpus)}")
        for i, gpu in enumerate(numba.cuda.gpus):
            print(f"  GPU {i}: {gpu.name} (Compute Capability: {gpu.compute_capability})")
    except Exception as e:
        print(f"GPU情報取得エラー: {e}")

print("\n=== インポートテスト ===")
modules_to_test = [
    'numba.cuda',
    'cudf',
    'pyarrow',
    'cupy'
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"{module}: 成功")
    except Exception as e:
        print(f"{module}: 失敗 - {e}")

print("\n=== CUDA Kernel簡易テスト ===")
try:
    from numba import cuda
    import numpy as np
    
    @cuda.jit
    def test_kernel(arr):
        idx = cuda.grid(1)
        if idx < arr.size:
            arr[idx] = idx * 2
    
    # テスト実行
    host_array = np.zeros(10, dtype=np.int32)
    device_array = cuda.to_device(host_array)
    
    test_kernel[1, 10](device_array)
    cuda.synchronize()
    
    result = device_array.copy_to_host()
    print(f"CUDA Kernel実行: 成功")
    print(f"結果: {result[:5]}...")
    
except Exception as e:
    print(f"CUDA Kernel実行: 失敗 - {e}")