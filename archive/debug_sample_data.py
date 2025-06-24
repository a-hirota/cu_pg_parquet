"""
テストデータ生成とデバッグ用ツール
"""
import numpy as np
from debug_simple_parser import debug_parse_binary

def create_test_binary_data(ncols=3, null_positions=None):
    """
    テスト用 COPY BINARY データを作成（test_single_row_pg_parser.py とほぼ同じロジック）
    
    Args:
        ncols: 列数
        null_positions: NULL値にする列インデックスのリスト
        
    Returns:
        バイナリデータのNumPy配列
    """
    if null_positions is None:
        null_positions = []
    
    # COPY バイナリフォーマットヘッダー: PGCOPY\n\377\r\n\0 (11バイト)
    header = np.array([80, 71, 67, 79, 80, 89, 10, 255, 13, 10, 0], dtype=np.uint8)
    
    # データ部分の構築
    data_parts = []
    
    # 行数 (1行)
    data_parts.append(np.array([0, ncols], dtype=np.uint8))  # フィールド数 (ncols)
    
    # 各フィールドのデータ
    for i in range(ncols):
        if i in null_positions:
            # NULL値 (-1 = 0xFFFFFFFF)
            data_parts.append(np.array([255, 255, 255, 255], dtype=np.uint8))
        else:
            # 適当なデータ (長さ4バイト + データ)
            field_len = 4  # 固定長4バイト
            # 長さフィールド (ビッグエンディアン)
            data_parts.append(np.array([0, 0, 0, field_len], dtype=np.uint8))
            
            # フィールドの内容 (i + 1の値を繰り返し)
            field_data = np.full(field_len, i + 1, dtype=np.uint8)
            data_parts.append(field_data)
    
    # 終端マーカー (0xFFFF)
    data_parts.append(np.array([255, 255], dtype=np.uint8))
    
    # 全てを結合
    return np.concatenate([header] + data_parts)

def dump_binary_data(raw_data, max_bytes=256):
    """バイナリデータを16進数でダンプ"""
    print("バイナリデータ構造:")
    for i in range(0, min(len(raw_data), max_bytes), 16):
        hex_values = " ".join(f"{b:02x}" for b in raw_data[i:i+16])
        ascii_values = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw_data[i:i+16])
        print(f"{i:04x}: {hex_values.ljust(48)} | {ascii_values}")

if __name__ == "__main__":
    # テストデータ生成（3列、うち2列目がNULL）
    sample_data = create_test_binary_data(ncols=3, null_positions=[1])
    
    # テストデータのダンプ
    dump_binary_data(sample_data)
    
    # バイナリファイル保存（検証用）
    with open("test_sample.bin", "wb") as f:
        f.write(sample_data.tobytes())
    
    print("\nCPUパーサーでのデバッグ解析:")
    # CPUパーサーで解析
    offsets, lengths = debug_parse_binary(sample_data)
    
    print("\n解析結果:")
    print(f"オフセット: {offsets}")
    print(f"長さ: {lengths}")
