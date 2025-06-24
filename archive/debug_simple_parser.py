"""
シンプルなCPU版バイナリデータ解析用デバッグツール
"""
import sys
import numpy as np

def debug_parse_binary(data_bytes, header_size=11, row_start=None):
    """
    バイナリデータを単純な方法で解析
    
    Args:
        data_bytes: バイナリデータ
        header_size: ヘッダーサイズ
        row_start: 行開始位置（Noneの場合はヘッダー後から）
    
    Returns:
        (offsets, lengths): 各フィールドのオフセットと長さ
    """
    if row_start is None:
        pos = header_size
    else:
        pos = row_start
    
    # データサイズチェック
    if pos + 2 > len(data_bytes):
        print("データ不足: フィールド数を読み取れません")
        return [], []
    
    # フィールド数
    field_count = (data_bytes[pos] << 8) | data_bytes[pos + 1]
    print(f"フィールド数: {field_count} (pos={pos}, bytes={data_bytes[pos]:02x} {data_bytes[pos+1]:02x})")
    
    # 終端マーカーチェック
    if field_count == 0xFFFF:
        print(f"終端マーカー検出 (pos={pos})")
        return [], []
    
    pos += 2  # フィールド数をスキップ
    
    # 各フィールドのパース
    offsets = []
    lengths = []
    
    for i in range(field_count):
        if pos + 4 > len(data_bytes):
            print(f"データ不足: フィールド{i}の長さを読み取れません")
            break
        
        # フィールド長を読み取り (ビッグエンディアン)
        field_len = ((data_bytes[pos] << 24) | 
                     (data_bytes[pos+1] << 16) | 
                     (data_bytes[pos+2] << 8) | 
                     data_bytes[pos+3])
        
        # バイトをダンプして確認
        hex_values = " ".join(f"{b:02x}" for b in data_bytes[pos:pos+4])
        print(f"フィールド{i} 長さ: {field_len} (pos={pos}, bytes={hex_values})")
        
        pos += 4  # 長さフィールドをスキップ
        
        if field_len == -1:  # NULL
            offsets.append(0)
            lengths.append(-1)
            print(f"  NULL値")
        else:
            offsets.append(pos)
            lengths.append(field_len)
            
            # フィールドの内容（最大16バイト）
            if field_len > 0:
                max_display = min(field_len, 16)
                hex_values = " ".join(f"{b:02x}" for b in data_bytes[pos:pos+max_display])
                ascii_values = "".join(chr(b) if 32 <= b <= 126 else "." for b in data_bytes[pos:pos+max_display])
                print(f"  データ: {hex_values} | {ascii_values}" + (" ..." if field_len > 16 else ""))
            
            pos += field_len  # データをスキップ
    
    return offsets, lengths

if __name__ == "__main__":
    # コマンドラインからバイナリファイルを指定
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            data = f.read()
        
        header_size = 11  # デフォルト
        if len(sys.argv) > 2:
            header_size = int(sys.argv[2])
        
        debug_parse_binary(np.frombuffer(data, dtype=np.uint8), header_size)
