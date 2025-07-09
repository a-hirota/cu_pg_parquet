#!/usr/bin/env python3
"""欠落している15行の実際のデータをPostgreSQLから取得"""

import psycopg
import os
import pandas as pd
import numpy as np

def get_missing_keys():
    """Parquetファイルから欠落キーを特定"""
    print("=== 欠落キーの特定 ===\n")
    
    # Parquetファイルを読み込み
    df = pd.read_parquet("output/customer_chunk_0_queue.parquet")
    # Decimal型をintに変換
    df['c_custkey'] = df['c_custkey'].astype('int64')
    all_keys = sorted(df['c_custkey'].unique())
    
    # キーのギャップを検出
    missing_keys = []
    for i in range(1, len(all_keys)):
        expected = all_keys[i-1] + 1
        actual = all_keys[i]
        
        if actual > expected:
            # ギャップがある
            for key in range(expected, actual):
                missing_keys.append(key)
    
    print(f"欠落キー数: {len(missing_keys)}")
    
    # 最初の20個を表示
    if missing_keys:
        print("\n欠落キー（最初の20個）:")
        for i, key in enumerate(missing_keys[:20]):
            print(f"  {i+1:2d}. {key}")
    
    return missing_keys

def get_row_data_from_postgres(keys):
    """PostgreSQLから指定キーの行データを取得"""
    print("\n\n=== PostgreSQLからデータ取得 ===\n")
    
    dsn = os.environ.get("GPUPASER_PG_DSN", "host=localhost dbname=postgres user=postgres")
    
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                # 最初の5個のキーの詳細を取得
                for key in keys[:5]:
                    print(f"\n--- c_custkey = {key} ---")
                    
                    # 行データを取得
                    cur.execute("""
                        SELECT 
                            c_custkey,
                            c_name,
                            c_address,
                            c_city,
                            c_nation,
                            c_region,
                            c_phone,
                            c_mktsegment,
                            length(c_name) as name_len,
                            length(c_address) as addr_len,
                            length(c_city) as city_len,
                            length(c_nation) as nation_len,
                            length(c_region) as region_len,
                            length(c_phone) as phone_len,
                            length(c_mktsegment) as mkt_len
                        FROM customer 
                        WHERE c_custkey = %s
                    """, (key,))
                    
                    row = cur.fetchone()
                    if row:
                        print(f"  c_name: '{row[1]}' (長さ: {row[8]})")
                        print(f"  c_address: '{row[2]}' (長さ: {row[9]})")
                        print(f"  c_city: '{row[3]}' (長さ: {row[10]})")
                        print(f"  c_nation: '{row[4]}' (長さ: {row[11]})")
                        print(f"  c_region: '{row[5]}' (長さ: {row[12]})")
                        print(f"  c_phone: '{row[6]}' (長さ: {row[13]})")
                        print(f"  c_mktsegment: '{row[7]}' (長さ: {row[14]})")
                        
                        # 合計バイト数を推定
                        total_bytes = 4 + 8  # custkey(4) + 列数(2) + 各フィールドヘッダ(4*8)
                        total_bytes += sum([row[8], row[9], row[10], row[11], row[12], row[13], row[14]])
                        print(f"  推定行サイズ: {total_bytes} bytes")
                    else:
                        print(f"  ⚠️ データが見つかりません")
                
                # バイナリ形式で1行取得してサイズを確認
                print("\n\n=== バイナリ形式でのサイズ確認 ===")
                
                cur.execute("""
                    COPY (SELECT * FROM customer WHERE c_custkey = %s) 
                    TO STDOUT (FORMAT BINARY)
                """, (keys[0],))
                
                binary_data = cur.read()
                if binary_data:
                    # バイナリデータを解析
                    print(f"\nc_custkey = {keys[0]}のバイナリデータ:")
                    print(f"  総バイト数: {len(binary_data)} bytes")
                    
                    # 最初の50バイトを16進表示
                    print(f"  最初の50バイト:")
                    for i in range(0, min(50, len(binary_data)), 16):
                        hex_str = ' '.join(f'{b:02X}' for b in binary_data[i:i+16])
                        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in binary_data[i:i+16])
                        print(f"    {i:04X}: {hex_str:<48} {ascii_str}")
                
                # 特殊文字の確認
                print("\n\n=== 特殊文字の確認 ===")
                
                for key in keys[:5]:
                    cur.execute("""
                        SELECT 
                            c_custkey,
                            c_name,
                            c_address
                        FROM customer 
                        WHERE c_custkey = %s
                    """, (key,))
                    
                    row = cur.fetchone()
                    if row:
                        # 特殊文字をチェック
                        special_chars = []
                        for field_name, field_value in [('c_name', row[1]), ('c_address', row[2])]:
                            for i, char in enumerate(field_value):
                                if ord(char) < 32 or ord(char) > 126:
                                    special_chars.append(f"{field_name}[{i}]: 0x{ord(char):02X}")
                        
                        if special_chars:
                            print(f"\nc_custkey = {key}:")
                            print(f"  特殊文字が含まれています:")
                            for sc in special_chars:
                                print(f"    {sc}")
                
    except Exception as e:
        print(f"エラー: {e}")

def analyze_missing_pattern():
    """欠落パターンを分析"""
    print("\n\n=== 欠落パターンの分析 ===\n")
    
    missing_keys = get_missing_keys()
    
    if not missing_keys:
        print("欠落キーがありません")
        return
    
    # 連続性を確認
    gaps = []
    current_start = missing_keys[0]
    current_end = missing_keys[0]
    
    for i in range(1, len(missing_keys)):
        if missing_keys[i] == current_end + 1:
            current_end = missing_keys[i]
        else:
            gaps.append((current_start, current_end))
            current_start = missing_keys[i]
            current_end = missing_keys[i]
    
    gaps.append((current_start, current_end))
    
    print(f"欠落グループ数: {len(gaps)}")
    print("\n欠落グループ:")
    for start, end in gaps:
        if start == end:
            print(f"  {start}")
        else:
            print(f"  {start} - {end} ({end - start + 1}個)")
    
    return missing_keys

if __name__ == "__main__":
    missing_keys = analyze_missing_pattern()
    if missing_keys:
        get_row_data_from_postgres(missing_keys)