#!/usr/bin/env python3
"""
date型変換のユニットテスト
"""

import numpy as np
import cupy as cp
import cudf
from numba import cuda
from src.types import TS64_S

def test_date_conversion_kernel():
    """PostgreSQL date型(4バイト)からdatetime64[s](8バイト)への変換をテスト"""
    
    # テスト用のカーネル関数を定義
    @cuda.jit
    def convert_date_kernel(raw_data, output_buffer):
        """PostgreSQL date型をdatetime64[s]に変換するカーネル"""
        tid = cuda.grid(1)
        
        if tid == 0:  # 1つのデータだけ処理
            src_offset = 0
            dst_offset = 0
            
            # 4バイトのビッグエンディアン整数を読み取り（符号付き）
            days_since_2000_unsigned = (raw_data[src_offset] << 24) | \
                                     (raw_data[src_offset + 1] << 16) | \
                                     (raw_data[src_offset + 2] << 8) | \
                                     raw_data[src_offset + 3]
            
            # 符号付き32ビット整数として解釈
            if days_since_2000_unsigned >= 0x80000000:
                days_since_2000 = days_since_2000_unsigned - 0x100000000
            else:
                days_since_2000 = days_since_2000_unsigned
            
            # 2000-01-01 00:00:00 UTCのUnix timestampは946684800秒
            # 日数を秒数に変換して、2000-01-01のタイムスタンプを加算
            seconds_since_epoch = np.int64(946684800 + days_since_2000 * 86400)
            
            # 8バイトのリトルエンディアンとして書き込み
            for i in range(8):
                output_buffer[dst_offset + i] = (seconds_since_epoch >> (i * 8)) & 0xFF
    
    # テストケース
    test_cases = [
        # (日数, 説明)
        (0, "2000-01-01 (PostgreSQL date epoch)"),
        (-10957, "1970-01-01 (Unix epoch)"),  # 2000-01-01から1970-01-01までは-10957日
        (9125, "2024-12-25 (Christmas 2024)"),  # 2000-01-01から2024-12-25までは9125日
        (-1, "1999-12-31 (Day before 2000)"),
    ]
    
    for days_since_2000, description in test_cases:
        print(f"\nテスト: {description}")
        print(f"PostgreSQL date値 (2000-01-01からの日数): {days_since_2000}")
        
        # PostgreSQLのバイナリフォーマット（4バイト、ビッグエンディアン）を作成
        raw_data = np.zeros(4, dtype=np.uint8)
        value = np.int32(days_since_2000)
        raw_data[0] = (value >> 24) & 0xFF
        raw_data[1] = (value >> 16) & 0xFF
        raw_data[2] = (value >> 8) & 0xFF
        raw_data[3] = value & 0xFF
        
        # GPUメモリにコピー
        d_raw_data = cuda.to_device(raw_data)
        d_output = cuda.device_array(8, dtype=np.uint8)
        
        # カーネル実行
        convert_date_kernel[1, 1](d_raw_data, d_output)
        
        # 結果を取得
        output = d_output.copy_to_host()
        
        # 8バイトのリトルエンディアンからint64に変換
        seconds_since_epoch = np.int64(0)
        for i in range(8):
            seconds_since_epoch |= np.int64(output[i]) << (i * 8)
        
        # numpyのdatetime64に変換して確認
        datetime_value = np.datetime64(int(seconds_since_epoch), 's')
        print(f"変換結果: {datetime_value}")
        
        # 期待値の計算
        expected_seconds = 946684800 + days_since_2000 * 86400
        expected_datetime = np.datetime64(int(expected_seconds), 's')
        print(f"期待値: {expected_datetime}")
        
        if datetime_value == expected_datetime:
            print("✓ 正しく変換されました")
        else:
            print("✗ 変換エラー")

def test_cudf_series_creation():
    """cuDFでdatetime64[s]のSeriesが作成できるかテスト"""
    print("\n\ncuDF Series作成テスト:")
    
    # Unix epochからの秒数のテストデータ
    seconds_data = np.array([
        946684800,      # 2000-01-01
        0,              # 1970-01-01
        1735084800,     # 2024-12-25
        946598400,      # 1999-12-31
    ], dtype=np.int64)
    
    # CuPy配列として作成
    cupy_data = cp.array(seconds_data)
    
    # datetime64[s]のSeriesを作成
    series = cudf.Series(cupy_data, dtype='datetime64[s]')
    
    print(series)
    print(f"データ型: {series.dtype}")

if __name__ == "__main__":
    print("=== PostgreSQL date型 → datetime64[s] 変換テスト ===")
    test_date_conversion_kernel()
    test_cudf_series_creation()
    print("\n\nテスト完了")