#!/usr/bin/env python
"""
バイナリデータ検査用デバッグスクリプト
"""

import sys
import os
import numpy as np
import psycopg2

def examine_binary_data(data_path='output_debug.bin', max_bytes=100):
    """バイナリデータファイルの内容を検査"""
    try:
        with open(data_path, 'rb') as f:
            binary_data = f.read()
            
        print(f"バイナリファイルサイズ: {len(binary_data)} バイト")
        print(f"最初の{max_bytes}バイトを表示:")
        
        # NumPy配列に変換
        data_array = np.frombuffer(binary_data, dtype=np.uint8)
        
        # ヘッダー部分を表示
        for i in range(min(max_bytes, len(data_array))):
            print(f"Pos {i}: 0x{data_array[i]:02x} ({data_array[i]})")
            
        # PostgreSQLバイナリヘッダー "PGCOPY\n\377\r\n\0" の確認
        if len(data_array) >= 11:
            header_pattern = np.array([80,71,67,79,80,89,10,255,13,10,0], dtype=np.uint8)
            header_match = True
            
            for i in range(11):
                if i >= len(data_array) or data_array[i] != header_pattern[i]:
                    header_match = False
                    break
                    
            if header_match:
                print("\nPostgreSQLバイナリヘッダーを確認: PGCOPY\\n\\377\\r\\n\\0")
                
                # フラグとヘッダー拡張情報を表示
                if len(data_array) >= 19:  # ヘッダー(11) + フラグ(4) + 拡張長(4)
                    flags = ((data_array[11] << 24) | (data_array[12] << 16) | 
                           (data_array[13] << 8) | data_array[14])
                    ext_len = ((data_array[15] << 24) | (data_array[16] << 16) | 
                             (data_array[17] << 8) | data_array[18])
                    
                    print(f"フラグ: 0x{flags:08x}")
                    print(f"拡張長: {ext_len}")
                    
                    # 拡張データをスキップした後の位置
                    pos = 11 + 8 + ext_len
                    
                    # 最初の行のフィールド数を確認
                    if len(data_array) >= pos + 2:
                        num_fields = ((data_array[pos] << 8) | data_array[pos + 1])
                        print(f"\n最初の行のフィールド数: {num_fields}")
                        
                        # フィールド長を確認
                        pos += 2
                        for field_idx in range(min(num_fields, 10)):  # 最初の10フィールドまで
                            if pos + 4 <= len(data_array):
                                field_len = ((data_array[pos] << 24) | (data_array[pos+1] << 16) | 
                                           (data_array[pos+2] << 8) | data_array[pos+3])
                                
                                # 符号付き整数に変換
                                if field_len & 0x80000000:
                                    field_len = -((~field_len + 1) & 0xFFFFFFFF)
                                
                                print(f"フィールド {field_idx+1}: 長さ={field_len}")
                                pos += 4
                                
                                # NULL値以外ならデータもスキップ
                                if field_len != -1:
                                    pos += field_len
            else:
                print("\n警告: PostgreSQLバイナリヘッダーが見つかりません")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False

def get_fresh_binary_data(table_name='customer', limit=100):
    """PostgreSQLから新しいバイナリデータを取得"""
    try:
        # PostgreSQLに接続
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        
        # バイナリデータを取得
        query = f"COPY (SELECT * FROM {table_name} LIMIT {limit}) TO STDOUT WITH (FORMAT binary)"
        
        with open('fresh_debug.bin', 'wb') as f:
            conn.cursor().copy_expert(query, f)
            
        conn.close()
        
        print(f"{table_name}テーブルから{limit}行のバイナリデータを取得しました")
        print("ファイル 'fresh_debug.bin' に保存しました")
        
        # 取得したデータを検査
        examine_binary_data('fresh_debug.bin')
        
        return True
    except Exception as e:
        print(f"データ取得エラー: {e}")
        return False

if __name__ == "__main__":
    # 既存のバイナリファイルを検査
    print("\n=== 既存のバイナリファイル検査 ===")
    examine_binary_data()
    
    # 新しいバイナリデータを取得して検査
    print("\n=== 新しいバイナリデータを取得 ===")
    get_fresh_binary_data(limit=10)
