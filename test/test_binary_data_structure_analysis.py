#!/usr/bin/env python3
"""
バイナリデータ構造詳細解析ツール
===============================

目的: データ終端近くでの行検出失敗の原因を特定
方法: バイナリデータの実際の構造を16進ダンプで確認
"""

import os
import sys
import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.types import ColumnMeta, INT32, UTF8

def create_detailed_test_data():
    """詳細解析用テストデータ（正確に100行）"""
    
    # ヘッダ（19バイト）
    header = bytearray(19)
    header[:11] = b"PGCOPY\n\xff\r\n\x00"  # COPY signature
    header[11:15] = (0).to_bytes(4, 'big')  # flags
    header[15:19] = (0).to_bytes(4, 'big')  # header extension length
    
    # データ部（正確に100行生成）
    data = bytearray()
    ncols = 3  # シンプルな3列
    
    row_positions = []  # 各行の開始位置を記録
    
    for row_id in range(100):
        # 行開始位置を記録
        row_positions.append(len(header) + len(data))
        
        # 行ヘッダ: フィールド数（2バイト）
        data.extend(ncols.to_bytes(2, 'big'))
        
        # フィールド1: INT32（固定4バイト）
        data.extend((4).to_bytes(4, 'big'))
        data.extend(row_id.to_bytes(4, 'big'))
        
        # フィールド2: INT32（固定4バイト）
        data.extend((4).to_bytes(4, 'big'))
        data.extend((row_id * 2).to_bytes(4, 'big'))
        
        # フィールド3: 文字列（可変長）
        if row_id % 10 == 9:  # 10行に1回NULL
            # NULL値
            data.extend((0xFFFFFFFF).to_bytes(4, 'big'))
        else:
            # 通常の文字列
            field_data = f"ROW{row_id:03d}".encode('utf-8')
            data.extend(len(field_data).to_bytes(4, 'big'))
            data.extend(field_data)
    
    # PostgreSQL終端マーカー追加
    terminator_pos = len(header) + len(data)
    data.extend((0xFFFF).to_bytes(2, 'big'))
    
    return bytes(header + data), row_positions, terminator_pos

def hex_dump(data, start_pos=0, length=256, width=16):
    """16進ダンプ表示"""
    
    end_pos = min(start_pos + length, len(data))
    
    print(f"16進ダンプ (位置 {start_pos:04X} - {end_pos-1:04X}):")
    print("=" * 80)
    
    for i in range(start_pos, end_pos, width):
        # アドレス表示
        addr = f"{i:04X}: "
        
        # 16進数表示
        hex_part = ""
        ascii_part = ""
        
        for j in range(width):
            if i + j < end_pos:
                byte_val = data[i + j]
                hex_part += f"{byte_val:02X} "
                ascii_part += chr(byte_val) if 32 <= byte_val <= 126 else "."
            else:
                hex_part += "   "
                ascii_part += " "
        
        print(f"{addr}{hex_part} | {ascii_part}")

def analyze_row_structure(data, row_positions, ncols=3):
    """各行の構造を詳細分析"""
    
    print(f"\n📊 行構造詳細分析")
    print("=" * 60)
    
    for i, row_start in enumerate(row_positions):
        if i >= 20 and i < 90:  # 中間は省略
            continue
        
        print(f"\n行 {i:3d} (位置 {row_start:04X}):")
        
        # 行ヘッダ確認
        if row_start + 2 > len(data):
            print("  ❌ ヘッダ読み取り不可（データ終端）")
            continue
        
        num_fields = (data[row_start] << 8) | data[row_start + 1]
        print(f"  フィールド数: {num_fields} (期待値: {ncols})")
        
        if num_fields != ncols:
            print("  ❌ フィールド数不一致")
            continue
        
        pos = row_start + 2
        row_size = 2  # ヘッダ分
        
        # 各フィールド分析
        for field_idx in range(num_fields):
            if pos + 4 > len(data):
                print(f"    フィールド{field_idx}: ❌ 長さ読み取り不可")
                break
            
            field_len = (
                data[pos] << 24 | data[pos+1] << 16 |
                data[pos+2] << 8 | data[pos+3]
            )
            
            if field_len == 0xFFFFFFFF:
                print(f"    フィールド{field_idx}: NULL")
                pos += 4
                row_size += 4
            else:
                if pos + 4 + field_len > len(data):
                    print(f"    フィールド{field_idx}: ❌ データ読み取り不可 (長さ:{field_len})")
                    break
                
                if field_idx < 2:  # INT32フィールド
                    value = int.from_bytes(data[pos+4:pos+8], 'big')
                    print(f"    フィールド{field_idx}: INT32 = {value}")
                else:  # 文字列フィールド
                    text = data[pos+4:pos+4+field_len].decode('utf-8')
                    print(f"    フィールド{field_idx}: TEXT = '{text}'")
                
                pos += 4 + field_len
                row_size += 4 + field_len
        
        print(f"  行サイズ: {row_size}B")
        
        # 次行ヘッダ確認
        if pos + 2 <= len(data):
            next_header = (data[pos] << 8) | data[pos + 1]
            if next_header == ncols:
                print(f"  次行ヘッダ: {next_header:04X} ✅")
            elif next_header == 0xFFFF:
                print(f"  終端マーカー: {next_header:04X} ✅")
            else:
                print(f"  次行ヘッダ: {next_header:04X} ❌")

def analyze_problem_area(data, row_positions):
    """問題領域（終端近く）の詳細分析"""
    
    print(f"\n🔍 問題領域分析（データ終端近く）")
    print("=" * 60)
    
    # スレッド90が担当する範囲 (2719-2749)
    problem_start = 2719
    problem_end = 2749
    
    print(f"スレッド90担当範囲: {problem_start} - {problem_end}")
    
    # この範囲の16進ダンプ
    hex_dump(data, problem_start - 50, 150)  # 前後50バイト含む
    
    # 最後の数行の構造確認
    print(f"\n📋 最後の10行分析:")
    for i in range(90, min(100, len(row_positions))):
        row_start = row_positions[i]
        print(f"  行{i}: 位置{row_start} (0x{row_start:04X})")
        
        if row_start >= problem_start and row_start < problem_end:
            print(f"    ★ スレッド90担当範囲内")

def main():
    """メイン実行関数"""
    
    print("🔍 バイナリデータ構造詳細解析")
    print("=" * 60)
    
    # テストデータ生成
    test_data, row_positions, terminator_pos = create_detailed_test_data()
    
    print(f"📝 テストデータ:")
    print(f"  総サイズ: {len(test_data)}B")
    print(f"  ヘッダ: 19B")
    print(f"  データ部: {len(test_data) - 19}B")
    print(f"  行数: {len(row_positions)}")
    print(f"  終端マーカー位置: {terminator_pos} (0x{terminator_pos:04X})")
    
    # 先頭部分の16進ダンプ
    print(f"\n📋 先頭部分（ヘッダ + 最初の行）:")
    hex_dump(test_data, 0, 100)
    
    # 行構造分析
    analyze_row_structure(test_data, row_positions)
    
    # 問題領域分析
    analyze_problem_area(test_data, row_positions)
    
    # 終端部分の16進ダンプ
    print(f"\n📋 終端部分:")
    hex_dump(test_data, len(test_data) - 100, 100)

if __name__ == "__main__":
    main()