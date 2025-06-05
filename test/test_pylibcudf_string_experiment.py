#!/usr/bin/env python3
"""
pylibcudf STRING型の正しい構築方法の実験プログラム

RMM DeviceBuffer + Column.from_rmm_buffer() を使った
正しい文字列カラムの作成方法を検証します。
"""

import os
import sys
import numpy as np
import cupy as cp
import cudf
import rmm
import pylibcudf as plc
from pylibcudf.types import DataType, TypeId

def test_pylibcudf_string_construction():
    """pylibcudf STRING型の正しい構築テスト"""
    
    print("=== pylibcudf STRING型構築実験 ===")
    
    # RMM初期化
    try:
        rmm.reinitialize(pool_allocator=True)
        print("✅ RMM初期化完了")
    except Exception as e:
        print(f"⚠️ RMM初期化警告: {e}")
    
    # テストデータ
    test_strings = ["ABC", "DEF", "GHIJK"]
    print(f"テスト文字列: {test_strings}")
    
    # 1. UTF-8バイト列の準備
    utf8_bytes = b"".join(s.encode('utf-8') for s in test_strings)
    print(f"結合UTF-8バイト: {utf8_bytes} (長さ: {len(utf8_bytes)})")
    
    # 2. オフセット配列の準備
    offsets = [0]
    for s in test_strings:
        offsets.append(offsets[-1] + len(s.encode('utf-8')))
    offsets_array = np.array(offsets, dtype=np.int32)
    print(f"オフセット配列: {offsets_array}")
    
    try:
        # 3. RMM DeviceBufferの作成
        print("\n--- RMM DeviceBuffer作成 ---")
        
        # chars buffer (INT8)
        chars_buf = rmm.DeviceBuffer.to_device(utf8_bytes)
        print(f"✅ chars_buf作成完了: {len(utf8_bytes)} bytes")
        
        # offsets buffer (INT32) - バイト配列として変換
        offsets_bytes = offsets_array.tobytes()
        offsets_buf = rmm.DeviceBuffer.to_device(offsets_bytes)
        print(f"✅ offsets_buf作成完了: {len(offsets_array)} elements")
        
        # 4. 利用可能なAPIの確認
        print("\n--- pylibcudf API確認 ---")
        print("Column available methods:")
        print([method for method in dir(plc.Column) if not method.startswith('_')])
        
        # 5. gpumemoryviewを使った方法を試行
        print("\n--- gpumemoryview使用方式 ---")
        
        # offsets column (INT32)
        offsets_mv = plc.gpumemoryview(offsets_buf)
        offsets_col = plc.Column(
            DataType(TypeId.INT32),
            len(offsets_array),
            offsets_mv,
            None,  # mask
            0,     # null_count
            0,     # offset
            []     # children
        )
        print(f"✅ offsets_col作成完了: type={offsets_col.type()}")
        
        # chars column (INT8)
        chars_mv = plc.gpumemoryview(chars_buf)
        chars_col = plc.Column(
            DataType(TypeId.INT8),
            len(utf8_bytes),
            chars_mv,
            None,  # mask
            0,     # null_count
            0,     # offset
            []     # children
        )
        print(f"✅ chars_col作成完了: type={chars_col.type()}")
        
        # 6. 親STRING カラムの作成
        print("\n--- 親STRING カラム作成 ---")
        
        # 親カラム用の空バッファ（未使用）
        empty_buf = rmm.DeviceBuffer(size=0)
        empty_mv = plc.gpumemoryview(empty_buf)
        
        string_col = plc.Column(
            DataType(TypeId.STRING),
            len(test_strings),  # 文字列の本数
            plc.gpumemoryview(chars_buf),  # chars buffer
            None,   # mask
            0,      # null_count
            0,      # offset
            [offsets_col]  # offset column
        )
        print(f"✅ string_col作成完了: type={string_col.type()}, size={string_col.size()}")
        print(f"   子カラム数: {string_col.num_children()}")
        
        # 7. cuDF Seriesへの変換
        print("\n--- cuDF Series変換 ---")
        
        cudf_series = cudf.Series.from_pylibcudf(string_col)
        print(f"✅ cuDF変換成功!")
        print(f"   dtype: {cudf_series.dtype}")
        print(f"   size: {len(cudf_series)}")
        
        # 7. 結果の確認
        print("\n--- 結果確認 ---")
        print("cuDF Series内容:")
        for i, val in enumerate(cudf_series.to_arrow().to_pylist()):
            print(f"[{i}]: '{val}'")
        
        # 8. 期待値との比較
        print("\n--- 検証 ---")
        success = True
        for i, expected in enumerate(test_strings):
            actual = str(cudf_series[i])
            if actual == expected:
                print(f"✅ [{i}]: '{actual}' == '{expected}'")
            else:
                print(f"❌ [{i}]: '{actual}' != '{expected}'")
                success = False
        
        if success:
            print("\n🎉 全テスト成功！正しいSTRING型構築が確認できました！")
            return True
        else:
            print("\n⚠️ 一部テスト失敗")
            return False
            
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        print(f"エラー型: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("pylibcudf STRING型構築実験を開始...")
    
    success = test_pylibcudf_string_construction()
    
    if success:
        print("\n✅ 実験完了: 本体コードへの組み込み準備完了！")
    else:
        print("\n❌ 実験失敗: 問題を修正してから再試行してください")
