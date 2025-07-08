#!/usr/bin/env python3
"""
PostgreSQLのページ番号とParquetのrow_numberから該当位置のバイト値を出力
"""

import cudf
import struct
from pathlib import Path
import sys

# PostgreSQLの定数
POSTGRES_PAGE_SIZE = 8192  # 8KB
POSTGRES_HEADER_SIZE = 24  # ページヘッダサイズ

def calculate_byte_position_from_page(page_number, offset_in_page=0):
    """ページ番号からバイト位置を計算"""
    return page_number * POSTGRES_PAGE_SIZE + offset_in_page

def find_worker_and_chunk(page_number):
    """ページ番号から担当ワーカーとチャンクを特定"""
    # チャンク1のワーカー配分（ログから）
    chunk1_workers = [
        (0, 0, 13023),
        (1, 13023, 26046),
        (2, 26046, 39069),
        (3, 39069, 52092),
        (4, 52092, 65115),
        (5, 65115, 78138),
        (6, 78138, 91161),
        (7, 91161, 104186),
    ]
    
    # チャンク2のワーカー配分（ログから）
    chunk2_workers = [
        (0, 104186, 117209),
        (1, 117209, 130232),
        (2, 130232, 143255),
        (3, 143255, 156278),
        (4, 156278, 169301),
        (5, 169301, 182324),
        (6, 182324, 195347),
        (7, 195347, 208372),
    ]
    
    # どのチャンク・ワーカーか判定
    if page_number < 104186:
        chunk = 1
        for worker_id, start, end in chunk1_workers:
            if start <= page_number < end:
                return chunk, worker_id, start, end
    else:
        chunk = 2
        for worker_id, start, end in chunk2_workers:
            if start <= page_number < end:
                return chunk, worker_id, start, end
    
    return None, None, None, None

def show_bytes_around_position(binary_data, position, before=100, after=100, highlight_pos=None):
    """指定位置周辺のバイトを表示"""
    start = max(0, position - before)
    end = min(len(binary_data), position + after)
    
    print(f"\nバイナリダンプ (0x{start:08X} - 0x{end:08X}):")
    print("-" * 80)
    
    for i in range(start, end, 16):
        if i >= len(binary_data):
            break
            
        # 16進表示
        hex_part = ""
        ascii_part = ""
        
        for j in range(16):
            if i + j < len(binary_data):
                byte = binary_data[i + j]
                
                # ハイライト位置かチェック
                if highlight_pos and i + j == highlight_pos:
                    hex_part += f"\033[91m{byte:02X}\033[0m "
                elif highlight_pos and abs(i + j - highlight_pos) < 2:
                    hex_part += f"\033[93m{byte:02X}\033[0m "
                else:
                    hex_part += f"{byte:02X} "
                
                # ASCII表示
                if 32 <= byte <= 126:
                    ascii_part += chr(byte)
                else:
                    ascii_part += "."
            else:
                hex_part += "   "
                ascii_part += " "
        
        # 現在位置マーカー
        marker = ""
        if highlight_pos and i <= highlight_pos < i + 16:
            marker = " <-- ここ"
                
        print(f"0x{i:08X}: {hex_part} |{ascii_part}|{marker}")

def analyze_thread_boundary(binary_data, thread_info):
    """スレッド境界の詳細分析"""
    thread_id = thread_info['thread_id']
    thread_start = thread_info['thread_start']
    thread_end = thread_info['thread_end']
    row_position = thread_info['row_position']
    
    print(f"\n{'='*80}")
    print(f"スレッド境界分析")
    print(f"{'='*80}")
    print(f"Thread ID: {thread_id}")
    print(f"Thread範囲: 0x{thread_start:08X} - 0x{thread_end:08X} ({thread_end - thread_start:,} bytes)")
    print(f"Row position: 0x{row_position:08X}")
    print(f"Thread開始からの距離: {row_position - thread_start} bytes")
    
    # スレッド開始位置での00 08検索
    print(f"\nスレッド開始位置からの00 08検索:")
    found_patterns = []
    for i in range(min(50, thread_end - thread_start)):
        if thread_start + i + 2 <= len(binary_data):
            if binary_data[thread_start + i] == 0x00 and binary_data[thread_start + i + 1] == 0x08:
                found_patterns.append(i)
                print(f"  +{i:3d}: 00 08 発見 (0x{thread_start + i:08X})")
    
    if not found_patterns:
        print("  00 08パターンが見つかりません")
    
    # row_position位置の値
    if row_position + 2 <= len(binary_data):
        val = struct.unpack('>H', binary_data[row_position:row_position+2])[0]
        print(f"\nrow_position位置の値: 0x{val:04X} ({val})")
    
    return found_patterns

def main():
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python debug_by_page_and_row.py <row_number>")
        print("  python debug_by_page_and_row.py <page_number> <offset>")
        print("\n例:")
        print("  python debug_by_page_and_row.py 1234567  # Parquetのrow_number")
        print("  python debug_by_page_and_row.py 13023 0  # ページ番号とオフセット")
        return
    
    # バイナリファイルを読み込み
    bin_file = Path("test_data_large.bin") if Path("test_data_large.bin").exists() else Path("test_data.bin")
    if not bin_file.exists():
        print(f"エラー: {bin_file} が見つかりません")
        return
        
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    print(f"バイナリファイル: {bin_file} ({len(binary_data):,} bytes)")
    
    # Parquetファイルから問題のある行を探す
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    if len(sys.argv) == 2:
        # row_numberモード
        target_row = int(sys.argv[1])
        print(f"\nrow_number {target_row} を検索中...")
        
        # 各Parquetファイルを確認
        for pf in parquet_files:
            df = cudf.read_parquet(pf)
            
            # row_numberでフィルタ（インデックスまたは明示的な列）
            if target_row < len(df):
                print(f"\n{pf.name} の row {target_row} を発見")
                
                # デバッグ情報を取得
                row_data = {}
                for col in df.columns:
                    try:
                        row_data[col] = df[col].iloc[target_row]
                    except:
                        row_data[col] = "エラー"
                
                # スレッド情報
                thread_info = {
                    'thread_id': int(row_data.get('_thread_id', -1)),
                    'thread_start': int(row_data.get('_thread_start_pos', -1)),
                    'thread_end': int(row_data.get('_thread_end_pos', -1)),
                    'row_position': int(row_data.get('_row_position', -1))
                }
                
                # c_custkeyの値を確認
                c_custkey = row_data.get('c_custkey', None)
                if c_custkey is not None:
                    if hasattr(c_custkey, 'dtype') and 'decimal' in str(c_custkey.dtype):
                        c_custkey_val = int(c_custkey)
                    else:
                        c_custkey_val = int(c_custkey)
                    
                    print(f"\nc_custkey: {c_custkey_val}")
                    if c_custkey_val == 0:
                        print("⚠️  c_custkey = 0 の問題のある行です！")
                
                # その他のフィールド値
                print(f"\nフィールド値:")
                for field in ['c_custkey', 'c_name', 'c_address', 'c_city', 
                            'c_nation', 'c_region', 'c_phone', 'c_mktsegment']:
                    if field in row_data:
                        print(f"  {field}: {row_data[field]}")
                
                # スレッド境界分析
                if thread_info['row_position'] > 0:
                    patterns = analyze_thread_boundary(binary_data, thread_info)
                    
                    # バイト表示
                    show_bytes_around_position(
                        binary_data, 
                        thread_info['row_position'],
                        before=50,
                        after=100,
                        highlight_pos=thread_info['row_position']
                    )
                
                break
    
    else:
        # ページ番号モード
        page_number = int(sys.argv[1])
        offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        
        chunk, worker, start_page, end_page = find_worker_and_chunk(page_number)
        
        if chunk:
            print(f"\nページ {page_number}:")
            print(f"  チャンク: {chunk}")
            print(f"  ワーカー: {worker}")
            print(f"  ページ範囲: {start_page} - {end_page}")
            
            byte_position = calculate_byte_position_from_page(page_number, offset)
            print(f"  バイト位置: 0x{byte_position:08X} ({byte_position:,})")
            
            if byte_position < len(binary_data):
                show_bytes_around_position(
                    binary_data,
                    byte_position,
                    before=100,
                    after=200,
                    highlight_pos=byte_position
                )
            else:
                print(f"エラー: バイト位置がファイルサイズを超えています")

if __name__ == "__main__":
    main()