#!/usr/bin/env python3
"""
c_custkey=0の行のバイナリデータをダンプして分析
"""

import cudf
import numpy as np
from pathlib import Path
import json
import sys
import struct

def hexdump(data, offset=0, length=None, highlight_ranges=None):
    """バイナリデータを16進数ダンプ形式で表示"""
    if length:
        data = data[:length]
    
    lines = []
    for i in range(0, len(data), 16):
        hex_bytes = []
        ascii_chars = []
        
        for j in range(16):
            if i + j < len(data):
                byte = data[i + j]
                # ハイライト範囲内かチェック
                is_highlight = False
                color_code = None
                if highlight_ranges:
                    for (start, end, color) in highlight_ranges:
                        if offset + i + j >= start and offset + i + j < end:
                            is_highlight = True
                            color_code = color
                            break
                
                if is_highlight and color_code:
                    hex_bytes.append(f"{color_code}{byte:02X}\033[0m")
                else:
                    hex_bytes.append(f"{byte:02X}")
                
                # ASCII表示
                if 32 <= byte <= 126:
                    if is_highlight and color_code:
                        ascii_chars.append(f"{color_code}{chr(byte)}\033[0m")
                    else:
                        ascii_chars.append(chr(byte))
                else:
                    if is_highlight and color_code:
                        ascii_chars.append(f"{color_code}.\033[0m")
                    else:
                        ascii_chars.append('.')
            else:
                hex_bytes.append('  ')
                ascii_chars.append(' ')
        
        hex_str = ' '.join(hex_bytes[:8]) + '  ' + ' '.join(hex_bytes[8:])
        ascii_str = ''.join(ascii_chars)
        lines.append(f"{offset + i:08X}  {hex_str}  |{ascii_str}|")
    
    return '\n'.join(lines)

def analyze_field_structure(data, row_start, expected_cols=8):
    """行の構造を解析"""
    if row_start + 2 > len(data):
        return None
    
    # フィールド数を読み取り
    num_fields = struct.unpack('>H', data[row_start:row_start+2])[0]
    print(f"\nフィールド数: {num_fields} (期待値: {expected_cols})")
    
    if num_fields != expected_cols:
        print(f"⚠️  フィールド数が期待値と異なります！")
        return None
    
    pos = row_start + 2
    fields = []
    
    # 各フィールドを解析
    for i in range(num_fields):
        if pos + 4 > len(data):
            print(f"エラー: フィールド{i}の長さ読み取り位置がデータ範囲外")
            break
        
        # フィールド長（ビッグエンディアン）
        field_len = struct.unpack('>i', data[pos:pos+4])[0]
        
        field_info = {
            'index': i,
            'len_pos': pos,
            'len': field_len,
            'data_pos': pos + 4,
        }
        
        if field_len == -1:  # NULL
            field_info['data'] = None
            field_info['type'] = 'NULL'
            pos += 4
        else:
            if pos + 4 + field_len > len(data):
                print(f"エラー: フィールド{i}のデータがデータ範囲外")
                field_info['data'] = None
                field_info['type'] = 'ERROR'
            else:
                field_info['data'] = data[pos+4:pos+4+field_len]
                field_info['type'] = 'DATA'
                pos += 4 + field_len
        
        fields.append(field_info)
    
    return fields

def main():
    """メイン処理"""
    # Parquetファイルから問題のある行を探す
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    if not parquet_files:
        print("Parquetファイルが見つかりません")
        return
    
    # customerテーブルのメタデータを読み込み
    metadata_file = Path("customer_metadata.json")
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            print(f"メタデータ読み込み完了")
            print(f"列数: {len(metadata['columns'])}")
            for i, col in enumerate(metadata['columns']):
                print(f"  {i}: {col['name']} (固定長: {col.get('fixed_field_length', -1)})")
    else:
        print("メタデータファイルがありません")
        metadata = None
    
    # 各Parquetファイルを処理
    found_issues = []
    
    for pf in parquet_files:
        print(f"\n処理中: {pf.name}")
        df = cudf.read_parquet(pf)
        
        # c_custkey=0の行を探す
        if 'c_custkey' in df.columns:
            if hasattr(df['c_custkey'].dtype, 'precision'):
                zero_rows = df[df['c_custkey'].astype('int64') == 0]
            else:
                zero_rows = df[df['c_custkey'] == 0]
            
            if len(zero_rows) > 0:
                print(f"  c_custkey=0が{len(zero_rows)}件見つかりました")
                
                # バイナリファイルを探す
                # 優先順位: test_data_large.bin, test_data.bin, その他
                bin_file = None
                for candidate in [Path("test_data_large.bin"), Path("test_data.bin"), Path(f"output/chunk_{pf.name.split('_')[1]}.bin")]:
                    if candidate.exists():
                        bin_file = candidate
                        break
                
                if not bin_file:
                    print(f"  バイナリファイルが見つかりません")
                    continue
                
                # バイナリデータを読み込み
                with open(bin_file, 'rb') as f:
                    binary_data = f.read()
                
                print(f"  バイナリファイルサイズ: {len(binary_data):,} bytes")
                
                # 最初の数行を詳細分析
                for idx in range(min(3, len(zero_rows))):
                    print(f"\n{'='*80}")
                    print(f"問題のある行 #{idx+1}:")
                    print(f"{'='*80}")
                    
                    # 行の情報を取得
                    row_info = {}
                    for col in df.columns:
                        try:
                            value = zero_rows[col].iloc[idx]
                            row_info[col] = value
                        except:
                            row_info[col] = "エラー"
                    
                    # デバッグ情報の表示
                    if '_row_position' in zero_rows.columns:
                        row_pos = int(zero_rows['_row_position'].iloc[idx])
                        print(f"\n行位置: {row_pos:,} (0x{row_pos:X})")
                        
                        if '_thread_id' in zero_rows.columns:
                            thread_id = int(zero_rows['_thread_id'].iloc[idx])
                            print(f"Thread ID: {thread_id}")
                        
                        if '_thread_start_pos' in zero_rows.columns:
                            thread_start = int(zero_rows['_thread_start_pos'].iloc[idx])
                            print(f"Thread開始位置: {thread_start:,}")
                            print(f"行位置からの距離: {row_pos - thread_start:,} bytes")
                        
                        if '_thread_end_pos' in zero_rows.columns:
                            thread_end = int(zero_rows['_thread_end_pos'].iloc[idx])
                            print(f"Thread終了位置: {thread_end:,}")
                        
                        # 各フィールドの値を表示
                        print(f"\nフィールド値:")
                        field_names = ['c_custkey', 'c_name', 'c_address', 'c_city', 
                                     'c_nation', 'c_region', 'c_phone', 'c_mktsegment']
                        for fn in field_names:
                            if fn in row_info:
                                value = row_info[fn]
                                if isinstance(value, (bytes, str)) and fn != 'c_custkey':
                                    print(f"  {fn}: '{value}' (長さ: {len(value)})")
                                else:
                                    print(f"  {fn}: {value}")
                        
                        # バイナリデータの範囲チェック
                        if row_pos >= len(binary_data):
                            print(f"\n⚠️  行位置がバイナリデータ範囲外です！")
                            continue
                        
                        # バイナリデータをダンプ
                        print(f"\n--- バイナリダンプ (行位置の前後) ---")
                        
                        # 表示範囲
                        before = 200
                        after = 400
                        start = max(0, row_pos - before)
                        end = min(len(binary_data), row_pos + after)
                        
                        # フィールド構造を解析
                        print(f"\n行構造の解析:")
                        fields = analyze_field_structure(binary_data, row_pos)
                        
                        # ハイライト範囲を設定
                        highlight_ranges = []
                        colors = [
                            '\033[91m',  # 赤
                            '\033[92m',  # 緑
                            '\033[93m',  # 黄
                            '\033[94m',  # 青
                            '\033[95m',  # マゼンタ
                            '\033[96m',  # シアン
                            '\033[97m',  # 白
                            '\033[90m',  # 灰
                        ]
                        
                        if fields:
                            print(f"\nフィールド詳細:")
                            for i, field in enumerate(fields):
                                color = colors[i % len(colors)]
                                print(f"  フィールド{i}: 長さ={field['len']}, 位置={field['data_pos']}")
                                
                                # データ内容を表示
                                if field['data'] is not None:
                                    if i == 0:  # c_custkey (int64)
                                        if len(field['data']) == 8:
                                            val = struct.unpack('>q', field['data'])[0]
                                            print(f"    値(int64): {val}")
                                    elif 1 <= i <= 7:  # 文字列フィールド
                                        try:
                                            text = field['data'].decode('utf-8', errors='replace')
                                            print(f"    値(文字列): '{text}'")
                                        except:
                                            print(f"    値(バイナリ): {field['data'].hex()}")
                                
                                # ハイライト範囲に追加
                                if field['type'] == 'DATA':
                                    highlight_ranges.append((field['data_pos'], field['data_pos'] + field['len'], color))
                        
                        # 行ヘッダもハイライト
                        highlight_ranges.append((row_pos, row_pos + 2, '\033[91m'))  # 赤で行ヘッダ
                        
                        # ダンプ表示
                        dump_data = binary_data[start:end]
                        print(f"\nバイナリダンプ（{start:,} - {end:,}）:")
                        print(f"行位置マーカー: \033[91m赤=行ヘッダ\033[0m, 各色=フィールドデータ")
                        print(hexdump(dump_data, offset=start, highlight_ranges=highlight_ranges))
                        
                        # 前の行も確認
                        print(f"\n--- 前の行の構造を探索 ---")
                        search_start = max(0, row_pos - 500)
                        found_prev = False
                        
                        for offset in range(row_pos - 2, search_start, -1):
                            if offset + 2 <= len(binary_data):
                                test_fields = struct.unpack('>H', binary_data[offset:offset+2])[0]
                                if test_fields == 8:  # customerテーブルは8フィールド
                                    # 簡易的な検証
                                    test_structure = analyze_field_structure(binary_data, offset)
                                    if test_structure and len(test_structure) == 8:
                                        # 最初のフィールドが8バイト（int64）かチェック
                                        if test_structure[0]['len'] == 8:
                                            print(f"前の行候補を発見: offset={offset:,}")
                                            found_prev = True
                                            break
                        
                        if not found_prev:
                            print("前の行が見つかりませんでした")

if __name__ == "__main__":
    main()