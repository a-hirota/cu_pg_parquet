import cudf
import pyarrow as pa
import pandas as pd
import numpy as np

print("=== 環境情報確認 ===")
print(f"CuDF version: {cudf.__version__}")
print(f"PyArrow version: {pa.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

print("\n=== CuDF Decimal128サポート確認 ===")
try:
    # PyArrow Decimal128配列作成テスト
    decimal_type = pa.decimal128(precision=10, scale=2)
    test_values = [123, 456, 789]
    arrow_array = pa.array(test_values, type=decimal_type)
    print(f"PyArrow Decimal128配列作成: 成功")
    print(f"Arrow配列: {arrow_array}")
    
    # CuDF変換テスト
    try:
        cudf_series = cudf.Series.from_arrow(arrow_array)
        print(f"CuDF Decimal128変換: 成功")
        print(f"CuDF Series dtype: {cudf_series.dtype}")
        print(f"CuDF Series: {cudf_series}")
    except Exception as e:
        print(f"CuDF Decimal128変換: 失敗 - {e}")
        
        # フォールバック: int64として処理
        int_values = [int(val) for val in test_values]
        cudf_series = cudf.Series(int_values, dtype='int64')
        print(f"フォールバック int64変換: 成功")
        print(f"CuDF Series dtype: {cudf_series.dtype}")
        print(f"CuDF Series: {cudf_series}")
        
except Exception as e:
    print(f"PyArrow Decimal128配列作成: 失敗 - {e}")

print("\n=== CuDF利用可能型確認 ===")
try:
    # 基本型テスト
    test_types = {
        'int32': np.int32,
        'int64': np.int64,
        'float32': np.float32,
        'float64': np.float64,
        'string': 'string'
    }
    
    for name, dtype in test_types.items():
        try:
            if name == 'string':
                series = cudf.Series(['test1', 'test2'], dtype=dtype)
            else:
                series = cudf.Series([1, 2, 3], dtype=dtype)
            print(f"{name}: 成功 (dtype: {series.dtype})")
        except Exception as e:
            print(f"{name}: 失敗 - {e}")
            
except Exception as e:
    print(f"型テストエラー: {e}")