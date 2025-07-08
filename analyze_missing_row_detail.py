#!/usr/bin/env python3
"""
欠落行の詳細分析 - バイナリデータとの照合
"""

import psycopg
import os
import struct

def get_row_binary_data(c_custkey):
    """PostgreSQLから特定の行のバイナリデータを取得"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # バイナリ形式で1行取得
            import io
            output = io.BytesIO()
            with cur.copy(f"""
                COPY (SELECT * FROM customer WHERE c_custkey = {c_custkey})
                TO STDOUT WITH (FORMAT BINARY)
            """) as copy:
                for data in copy:
                    output.write(data)
            
            binary_data = output.getvalue()
            
            # また、テキスト形式でも取得して比較
            cur.execute(f"""
                SELECT c_custkey, c_name, c_address, c_city, 
                       c_nation, c_region, c_phone, c_mktsegment,
                       pg_typeof(c_custkey), octet_length(c_name::text)
                FROM customer 
                WHERE c_custkey = {c_custkey}
            """)
            row_text = cur.fetchone()
    
    return binary_data, row_text

def analyze_binary_format(binary_data):
    """バイナリデータを解析"""
    print("\nバイナリデータ解析:")
    print(f"総サイズ: {len(binary_data)} bytes")
    
    # ヘキサダンプ
    print("\nヘキサダンプ:")
    for i in range(0, len(binary_data), 16):
        hex_part = ' '.join(f'{b:02X}' for b in binary_data[i:i+16])
        ascii_part = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in binary_data[i:i+16])
        print(f"{i:04X}: {hex_part:<48} |{ascii_part}|")
    
    # COPY BINARYヘッダをスキップ
    pos = 19  # 標準ヘッダサイズ
    
    if pos + 2 <= len(binary_data):
        num_fields = struct.unpack('>H', binary_data[pos:pos+2])[0]
        print(f"\nフィールド数: {num_fields}")
        pos += 2
        
        # 各フィールドを解析
        for i in range(num_fields):
            if pos + 4 > len(binary_data):
                break
                
            field_len = struct.unpack('>i', binary_data[pos:pos+4])[0]
            print(f"\nフィールド{i}: 長さ={field_len}")
            
            if field_len > 0 and pos + 4 + field_len <= len(binary_data):
                field_data = binary_data[pos+4:pos+4+field_len]
                
                # 最初のフィールド（c_custkey）は数値
                if i == 0 and field_len == 8:
                    value = struct.unpack('>q', field_data)[0]
                    print(f"  値(int64): {value}")
                # その他は文字列として表示
                elif 1 <= i <= 7:
                    try:
                        text = field_data.decode('utf-8')
                        print(f"  値: '{text}'")
                    except:
                        print(f"  値(hex): {field_data.hex()}")
            
            pos += 4 + (field_len if field_len > 0 else 0)
    
    # 行末のtrailerを確認
    if pos < len(binary_data):
        print(f"\n行末データ: {binary_data[pos:].hex()}")

def main():
    # 最初の欠落キーを詳しく調査
    missing_key = 476029
    
    print(f"欠落キー {missing_key} の詳細分析")
    print("="*80)
    
    # PostgreSQLから実際のデータを取得
    binary_data, row_text = get_row_binary_data(missing_key)
    
    print("\nPostgreSQLのデータ:")
    if row_text:
        print(f"  c_custkey: {row_text[0]}")
        print(f"  c_name: '{row_text[1]}'")
        print(f"  c_address: '{row_text[2]}'")
        print(f"  c_city: '{row_text[3]}'")
        print(f"  c_nation: '{row_text[4]}'")
        print(f"  c_region: '{row_text[5]}'")
        print(f"  c_phone: '{row_text[6]}'")
        print(f"  c_mktsegment: '{row_text[7]}'")
        print(f"  c_custkey型: {row_text[8]}")
        print(f"  c_name長: {row_text[9]} bytes")
    
    # バイナリフォーマットを解析
    analyze_binary_format(binary_data)
    
    # 行のサイズを計算
    # ヘッダ(19) + 行ヘッダ(2) + フィールド長(4*8) + データ + trailer(2)
    estimated_size = 19 + 2 + 32  # 基本構造
    if row_text:
        # c_custkey(8) + 各文字列フィールドの実際の長さ
        estimated_size += 8  # c_custkey
        for i in range(1, 8):
            if row_text[i]:
                estimated_size += len(row_text[i].encode('utf-8'))
    estimated_size += 2  # trailer
    
    print(f"\n推定行サイズ: {estimated_size} bytes")
    print(f"実際のサイズ: {len(binary_data)} bytes")
    
    # スレッド境界での位置を推定
    print("\n\nスレッド境界での推定位置:")
    print("前スレッド 1048575 の終了位置: 0x0BFFFFAE")
    print("後スレッド 3201426 の開始位置: 0x24A32E48")
    print(f"ギャップ: 0x{0x24A32E48 - 0x0BFFFFAE:X} ({0x24A32E48 - 0x0BFFFFAE:,} bytes)")
    
    # この行が収まるかチェック
    gap_size = 0x24A32E48 - 0x0BFFFFAE
    if len(binary_data) - 19 < gap_size:  # COPY headerを除く
        print(f"\n→ 行データ({len(binary_data)-19} bytes)はギャップ({gap_size} bytes)に収まります")
        print("  しかし、どちらのスレッドも処理していません！")

if __name__ == "__main__":
    main()