#!/usr/bin/env python3
"""
バイナリファイルの実際のテーブルを確認
"""

def main():
    with open('test_data_large.bin', 'rb') as f:
        data = f.read(400)
    
    # PGCOPYヘッダ
    print(f"ヘッダ: {data[:11]}")  # PGCOPY\n\xff\r\n\x00
    
    # 最初の行を解析
    pos = 19  # ヘッダの後
    num_fields = (data[pos] << 8) | data[pos+1]
    print(f"\nフィールド数: {num_fields}")
    
    if num_fields == 8:
        print("→ customerテーブル（8フィールド）の可能性")
    elif num_fields == 17:
        print("→ lineorderテーブル（17フィールド）の可能性")
    
    # フィールドを解析
    pos += 2
    fields = []
    for i in range(min(num_fields, 10)):
        if pos + 4 > len(data):
            break
        field_len = int.from_bytes(data[pos:pos+4], 'big', signed=True)
        fields.append(field_len)
        pos += 4
        if field_len > 0:
            pos += field_len
    
    print(f"\nフィールド長: {fields}")
    
    # 特徴的なフィールドを確認
    pos = 21  # 最初のフィールドから
    print("\n最初の数フィールドの内容:")
    for i in range(min(8, num_fields)):
        if pos + 4 > len(data):
            break
        field_len = int.from_bytes(data[pos:pos+4], 'big', signed=True)
        pos += 4
        
        if field_len > 0 and pos + field_len <= len(data):
            field_data = data[pos:pos+field_len]
            
            if i == 0 and field_len == 8:  # 最初のフィールドがint64
                val = int.from_bytes(field_data, 'big')
                print(f"  フィールド0 (int64): {val}")
                if val > 100000000:
                    print("    → lo_orderkeyの可能性（大きな値）")
                elif val < 10000000:
                    print("    → c_custkeyの可能性（小さな値）")
            elif i == 6:  # 7番目のフィールド
                try:
                    text = field_data.decode('utf-8', errors='replace')
                    print(f"  フィールド6: \"{text}\"")
                    if "HIGH" in text or "LOW" in text or "URGENT" in text:
                        print("    → lo_orderpriorityの可能性")
                except:
                    pass
            
            pos += field_len

if __name__ == "__main__":
    main()