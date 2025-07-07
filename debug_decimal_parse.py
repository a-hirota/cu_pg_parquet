#!/usr/bin/env python3
"""
Decimalパースのデバッグ
PostgreSQLからバイナリデータを取得して解析
"""

import os
import psycopg
import struct

def debug_decimal_binary(value):
    """PostgreSQL NUMERICのバイナリ形式を解析"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # COPY BINARYでデータを取得
            with cur.copy(f"COPY (SELECT {value}::numeric) TO STDOUT WITH (FORMAT BINARY)") as copy:
                binary_data = b''
                for data in copy:
                    binary_data += data
            
            # バイナリデータを解析
            print(f"\n=== {value}のバイナリ解析 ===")
            print(f"バイナリ長: {len(binary_data)}バイト")
            print(f"バイナリデータ: {binary_data.hex()}")
            
            # COPY BINARYヘッダをスキップ（19バイト）
            data = binary_data[19:]
            
            # フィールド数（2バイト）
            field_count = struct.unpack('>H', data[0:2])[0]
            print(f"フィールド数: {field_count}")
            
            # フィールド長（4バイト）
            field_len = struct.unpack('>I', data[2:6])[0]
            print(f"フィールド長: {field_len}")
            
            # NUMERICヘッダ（8バイト）
            numeric_data = data[6:6+field_len]
            nd = struct.unpack('>H', numeric_data[0:2])[0]       # 桁数
            weight = struct.unpack('>h', numeric_data[2:4])[0]  # 重み（符号付き）
            sign = struct.unpack('>H', numeric_data[4:6])[0]    # 符号
            dscale = struct.unpack('>H', numeric_data[6:8])[0]  # 小数点以下桁数
            
            print(f"\nNUMERICヘッダ:")
            print(f"  nd (桁数): {nd}")
            print(f"  weight: {weight}")
            print(f"  sign: 0x{sign:04X}")
            print(f"  dscale: {dscale}")
            
            # 桁データ
            digits = []
            for i in range(nd):
                digit = struct.unpack('>H', numeric_data[8+i*2:10+i*2])[0]
                digits.append(digit)
            
            print(f"\n基数10000の桁: {digits}")
            
            # 実際の値を計算
            value_calc = 0
            for i, digit in enumerate(digits):
                # weightから各桁の位置を計算
                power = weight - i
                value_calc += digit * (10000 ** power)
            
            print(f"\n計算結果: {value_calc}")
            print(f"期待値: {value}")
            
            # 現在の実装の問題を示す
            print(f"\n【現在の実装の問題】")
            print(f"weightを無視すると:")
            wrong_value = 0
            for digit in digits:
                wrong_value = wrong_value * 10000 + digit
            print(f"  誤った値: {wrong_value}")
            
            return nd, weight, sign, dscale, digits

# テストケース
test_values = [877, 8770000, 535, 5350000, 1033, 10330000]

for val in test_values:
    try:
        debug_decimal_binary(val)
    except Exception as e:
        print(f"\nエラー ({val}): {e}")