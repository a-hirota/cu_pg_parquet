#!/usr/bin/env python3
"""
実際の行ヘッダ位置を分析
"""

import cudf
import struct
from pathlib import Path

def find_row_header(binary_data, start_pos, search_range=1000):
    """指定位置周辺で00 08の行ヘッダを探す"""
    headers = []
    
    # 前方向に探索
    for offset in range(max(0, start_pos - search_range), start_pos):
        if offset + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[offset:offset+2])[0]
            if val == 8:
                headers.append(('before', offset, start_pos - offset))
    
    # 後方向に探索
    for offset in range(start_pos, min(len(binary_data) - 1, start_pos + search_range)):
        if offset + 2 <= len(binary_data):
            val = struct.unpack('>H', binary_data[offset:offset+2])[0]
            if val == 8:
                headers.append(('after', offset, offset - start_pos))
    
    return headers

def main():
    # Parquetファイルを読み込み
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    # バイナリファイルを探す
    bin_file = None
    for candidate in [Path("test_data_large.bin"), Path("test_data.bin")]:
        if candidate.exists():
            bin_file = candidate
            break
    
    if not bin_file:
        print("バイナリファイルが見つかりません")
        return
    
    # バイナリデータを読み込み
    with open(bin_file, 'rb') as f:
        binary_data = f.read()
    
    for pf in parquet_files:
        df = cudf.read_parquet(pf)
        
        # c_custkey=0の行を探す
        if 'c_custkey' in df.columns:
            if hasattr(df['c_custkey'].dtype, 'precision'):
                zero_rows = df[df['c_custkey'].astype('int64') == 0]
            else:
                zero_rows = df[df['c_custkey'] == 0]
            
            if len(zero_rows) > 0:
                print(f"\n=== {pf.name}: c_custkey=0が{len(zero_rows)}件 ===")
                
                for idx in range(min(3, len(zero_rows))):
                    if '_row_position' in zero_rows.columns:
                        row_pos = int(zero_rows['_row_position'].iloc[idx])
                        thread_id = int(zero_rows['_thread_id'].iloc[idx]) if '_thread_id' in zero_rows.columns else -1
                        thread_start = int(zero_rows['_thread_start_pos'].iloc[idx]) if '_thread_start_pos' in zero_rows.columns else -1
                        
                        print(f"\n問題のある行 #{idx+1}:")
                        print(f"  row_position: {row_pos:,} (0x{row_pos:X})")
                        print(f"  thread_id: {thread_id}")
                        print(f"  thread_start: {thread_start:,}")
                        print(f"  距離: {row_pos - thread_start} bytes")
                        
                        # row_positionの位置のデータを確認
                        if row_pos < len(binary_data) - 2:
                            val_at_pos = struct.unpack('>H', binary_data[row_pos:row_pos+2])[0]
                            print(f"  row_position位置の値: 0x{val_at_pos:04X} ({val_at_pos})")
                            
                            # 前後20バイトを表示
                            print(f"\n  row_position周辺のデータ:")
                            start = max(0, row_pos - 20)
                            end = min(len(binary_data), row_pos + 20)
                            
                            for i in range(start, end, 2):
                                if i + 2 <= len(binary_data):
                                    val = struct.unpack('>H', binary_data[i:i+2])[0]
                                    marker = " <-- row_position" if i == row_pos else ""
                                    print(f"    0x{i:08X}: {binary_data[i]:02X} {binary_data[i+1]:02X} (値: {val}){marker}")
                        
                        # 00 08パターンを探す
                        print(f"\n  近くの00 08パターン:")
                        headers = find_row_header(binary_data, row_pos, 200)
                        
                        for direction, pos, distance in headers[:5]:
                            print(f"    {direction}: 0x{pos:08X} (距離: {distance} bytes)")
                            
                            # その位置から行構造を確認
                            if pos + 100 < len(binary_data):
                                # フィールド数
                                num_fields = struct.unpack('>H', binary_data[pos:pos+2])[0]
                                # 最初のフィールド長
                                field1_len = struct.unpack('>i', binary_data[pos+2:pos+6])[0]
                                # 2番目のフィールド長
                                if field1_len > 0 and pos + 6 + field1_len + 4 < len(binary_data):
                                    field2_pos = pos + 6 + field1_len
                                    field2_len = struct.unpack('>i', binary_data[field2_pos:field2_pos+4])[0]
                                    print(f"      フィールド1長: {field1_len}, フィールド2長: {field2_len}")
                                    
                                    # 最初のフィールドがint64（8バイト）なら正しい可能性が高い
                                    if field1_len == 8:
                                        print(f"      → 正しい行ヘッダの可能性大")

if __name__ == "__main__":
    main()