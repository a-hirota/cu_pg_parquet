"""
ZeroCopy修正のテストプログラム

Decimal128と文字列の変換を個別にテストして問題を特定・修正します。
"""

import numpy as np
import cupy as cp
import cudf
import warnings
from numba import cuda
import sys
import os

# パッケージのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_decimal128_conversion():
    """Decimal128変換のテスト"""
    print("=== Decimal128変換テスト ===")
    
    try:
        # テスト用の整数値（PostgreSQL Numeric相当）
        test_values = [12345678, -987654, 0, 999999999999]
        scale = 4
        
        # 方法1: numpy配列 + astype方式（修正版）
        print("方法1: numpy配列 + astype")
        decimal_array = np.array(test_values, dtype=np.int64)
        decimal_dtype = cudf.Decimal128Dtype(precision=38, scale=scale)
        series1 = cudf.Series(decimal_array).astype(decimal_dtype)
        print(f"✅ 成功: {series1}")
        
        # 方法2: PyArrow経由（既存の動作確認済み方式）
        print("\n方法2: PyArrow経由")
        import pyarrow as pa
        arrow_decimal_type = pa.decimal128(precision=38, scale=scale)
        arrow_array = pa.array(test_values, type=arrow_decimal_type)
        series2 = cudf.Series.from_arrow(arrow_array)
        print(f"✅ 成功: {series2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Decimal128変換エラー: {e}")
        return False

def test_string_conversion():
    """文字列変換のテスト"""
    print("\n=== 文字列変換テスト ===")
    
    try:
        # テスト用データ
        test_strings = ["Hello", "World", "cuDF", "Test"]
        
        # データとオフセットを準備
        data_bytes = "".join(test_strings).encode('utf-8')
        offsets = np.zeros(len(test_strings) + 1, dtype=np.int32)
        
        current_offset = 0
        for i, s in enumerate(test_strings):
            offsets[i] = current_offset
            current_offset += len(s.encode('utf-8'))
        offsets[-1] = current_offset
        
        print(f"データバイト: {data_bytes}")
        print(f"オフセット: {offsets}")
        
        # GPU上にデータを配置
        data_cupy = cp.asarray(np.frombuffer(data_bytes, dtype=np.uint8))
        offsets_cupy = cp.asarray(offsets)
        
        print(f"CuPyデータ形状: {data_cupy.shape}")
        print(f"CuPyオフセット形状: {offsets_cupy.shape}")
        
        # 方法1: pylibcudf方式のテスト
        print("\n方法1: pylibcudf方式")
        try:
            import pylibcudf as plc
            import rmm
            
            # CuPy → pylibcudf変換
            try:
                # 正しいAPI: from_cuda_array_interface_obj を使用
                offsets_col = plc.column.Column.from_cuda_array_interface_obj(
                    offsets_cupy.__cuda_array_interface__
                )
                
                # 文字データバッファ
                chars_buf = rmm.DeviceBuffer(
                    data=data_cupy.data.ptr,
                    size=data_cupy.nbytes
                )
                
                # 文字列列を生成
                str_col_cpp = plc.strings.make_strings_column(
                    len(test_strings),
                    offsets_col,
                    chars_buf,
                    0,
                    rmm.DeviceBuffer()
                )
                
                # Python Series化
                series1 = cudf.Series._from_pylibcudf(str_col_cpp)
                print(f"✅ pylibcudf成功: {series1}")
                
            except Exception as e:
                print(f"❌ pylibcudf変換エラー: {e}")
                raise
                
        except ImportError:
            print("⚠️ pylibcudf が利用できません")
        
        # 方法2: PyArrow経由（フォールバック）
        print("\n方法2: PyArrow経由（フォールバック）")
        host_data = data_cupy.get()
        host_offsets = offsets_cupy.get()
        
        import pyarrow as pa
        pa_string_array = pa.StringArray.from_buffers(
            length=len(test_strings),
            value_offsets=pa.py_buffer(host_offsets),
            data=pa.py_buffer(host_data),
            null_bitmap=None
        )
        series2 = cudf.Series.from_arrow(pa_string_array)
        print(f"✅ PyArrow成功: {series2}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文字列変換エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cudf_version_info():
    """cuDFとpylibcudfのバージョン情報"""
    print("\n=== バージョン情報 ===")
    
    try:
        print(f"cuDF バージョン: {cudf.__version__}")
        
        try:
            import pylibcudf as plc
            print(f"pylibcudf インポート成功")
            print(f"pylibcudf.column 属性: {dir(plc.column)}")
            if hasattr(plc, '__version__'):
                print(f"pylibcudf バージョン: {plc.__version__}")
        except ImportError as e:
            print(f"pylibcudf インポートエラー: {e}")
            
        try:
            import rmm
            print(f"RMM バージョン: {rmm.__version__}")
        except Exception as e:
            print(f"RMM情報取得エラー: {e}")
            
    except Exception as e:
        print(f"バージョン情報エラー: {e}")

def main():
    """メインテスト実行"""
    print("🔬 ZeroCopy修正テストプログラム")
    print("=" * 50)
    
    # バージョン情報確認
    test_cudf_version_info()
    
    # Decimal128テスト
    decimal_success = test_decimal128_conversion()
    
    # 文字列テスト
    string_success = test_string_conversion()
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("🏁 テスト結果サマリー")
    print(f"Decimal128変換: {'✅ 成功' if decimal_success else '❌ 失敗'}")
    print(f"文字列変換: {'✅ 成功' if string_success else '❌ 失敗'}")
    
    if decimal_success and string_success:
        print("🎉 全テスト成功！")
        return 0
    else:
        print("⚠️ 一部テスト失敗")
        return 1

if __name__ == "__main__":
    exit(main())