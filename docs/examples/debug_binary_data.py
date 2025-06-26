#!/usr/bin/env python
"""
PostgreSQL COPY BINARY データの解析・デバッグ用ユーティリティ

使い方:
    python debug_binary_data.py binary_file.bin [rows] [max_bytes]

引数:
    binary_file.bin - 解析するバイナリファイル
    rows - 表示する行数 (デフォルト: 3)
    max_bytes - 各フィールドの表示最大バイト数 (デフォルト: 16)
"""

import sys
import os
import numpy as np
from typing import List, Tuple


def decode_int32_be(data: np.ndarray, pos: int) -> int:
    """ビッグエンディアンの4バイト整数をデコード"""
    if pos + 4 > len(data):
        return 0
    return ((data[pos] << 24) | (data[pos + 1] << 16) |
            (data[pos + 2] << 8) | data[pos + 3])


def bytes_to_hex(data: np.ndarray, start: int, length: int, max_bytes: int = 16) -> str:
    """バイトデータを16進数文字列に変換（最大max_bytes）"""
    if length == -1:
        return "NULL (-1)"
    
    end = min(start + length, len(data))
    show_len = min(length, max_bytes)
    bytes_data = data[start:start + show_len]
    
    hex_str = " ".join([f"{b:02x}" for b in bytes_data])
    ascii_str = "".join([chr(b) if 32 <= b <= 126 else "." for b in bytes_data])
    
    if length > max_bytes:
        return f"{hex_str} ... ({length} bytes) [{ascii_str}...]"
    return f"{hex_str} ({length} bytes) [{ascii_str}]"


def detect_pg_header_size(data: np.ndarray) -> int:
    """PostgreSQLバイナリフォーマットのヘッダーサイズを検出"""
    # 基本ヘッダー長 (PG + flag + extension marker)
    header_size = 11
    
    # 最小サイズチェック
    if len(data) < header_size:
        return header_size
    
    # PGCOPYヘッダーチェック (PGCOPY\n\377\r\n\0)
    if data[0] != 80 or data[1] != 71:  # 'P', 'G'
        return header_size  # ヘッダーが見つからない場合は標準サイズを返す
    
    # 拡張フラグのチェック
    if len(data) >= header_size + 8:
        # 拡張データ長の取得 (ビッグエンディアン)
        ext_len = decode_int32_be(data, header_size)
        
        if ext_len > 0 and len(data) >= header_size + 8 + ext_len:
            # ヘッダーサイズを調整 (拡張データ長 + int32サイズ x 2)
            header_size += 8 + ext_len
    
    return header_size


def dump_pg_header(data: np.ndarray) -> int:
    """ヘッダー部分をダンプして解析し、ヘッダーサイズを返す"""
    if len(data) < 11:
        print("データが小さすぎます (11バイト未満)")
        return 0
    
    # 基本ヘッダー (PGCOPY\n\377\r\n\0)
    header_str = "".join([chr(b) if 32 <= b <= 126 else f"\\x{b:02x}" for b in data[:11]])
    print(f"ヘッダー: {header_str}")
    print(f"ヘッダーHEX: {' '.join([f'{b:02x}' for b in data[:11]])}")
    
    # ヘッダーサイズ検出
    header_size = detect_pg_header_size(data)
    print(f"検出されたヘッダーサイズ: {header_size} バイト")
    
    # 拡張情報があれば表示
    if header_size > 11:
        ext_flags = decode_int32_be(data, 11)
        ext_len = decode_int32_be(data, 15)
        print(f"拡張フラグ: 0x{ext_flags:08x}, 拡張長: {ext_len} バイト")
        
        if ext_len > 0:
            ext_data = data[19:19+ext_len]
            print(f"拡張データ: {bytes_to_hex(data, 19, ext_len)}")
    
    return header_size


def parse_rows(data: np.ndarray, header_size: int, max_rows: int = 3, max_bytes: int = 16) -> None:
    """指定された行数分のデータを解析して表示"""
    pos = header_size
    row_idx = 0
    
    while pos < len(data) - 2 and row_idx < max_rows:
        # タプルのフィールド数
        num_fields = (data[pos] << 8) | data[pos + 1]
        
        # 終端マーカーチェック
        if num_fields == 0xFFFF:
            print(f"\n終端マーカー (0xFFFF) を検出: pos={pos}")
            break
            
        print(f"\n行 {row_idx + 1}: フィールド数={num_fields}, 開始位置={pos}")
        pos += 2  # フィールド数フィールドをスキップ
        
        # 各フィールドを解析
        for field_idx in range(num_fields):
            if pos + 4 > len(data):
                print(f"  警告: データ終端に達しました (pos={pos})")
                break
                
            # フィールド長を取得
            field_len = decode_int32_be(data, pos)
            pos += 4
            
            # フィールド内容を表示
            if field_len == -1:
                print(f"  フィールド {field_idx + 1}: NULL")
            else:
                field_data = bytes_to_hex(data, pos, field_len, max_bytes)
                print(f"  フィールド {field_idx + 1}: {field_data}, 開始位置={pos}")
                pos += field_len
                
        row_idx += 1
    
    # 全体の行数を推定（厳密ではなく概算）
    estimated_rows = estimate_total_rows(data, header_size)
    print(f"\n推定総行数: 約 {estimated_rows} 行")


def estimate_total_rows(data: np.ndarray, header_size: int) -> int:
    """データ全体の行数を概算（最初の数行から平均行サイズを計算）"""
    pos = header_size
    rows = 0
    total_size = 0
    
    # 最大10行までサンプリング
    while pos < len(data) - 2 and rows < 10:
        row_start = pos
        
        # タプルのフィールド数
        num_fields = (data[pos] << 8) | data[pos + 1]
        
        # 終端マーカーチェック
        if num_fields == 0xFFFF:
            break
            
        pos += 2  # フィールド数フィールドをスキップ
        
        # 各フィールドをスキップ
        valid_row = True
        for _ in range(num_fields):
            if pos + 4 > len(data):
                valid_row = False
                break
                
            field_len = decode_int32_be(data, pos)
            pos += 4
            
            if field_len != -1:
                if pos + field_len > len(data):
                    valid_row = False
                    break
                pos += field_len
        
        if valid_row:
            rows += 1
            total_size += (pos - row_start)
        else:
            break
    
    # 平均行サイズを計算
    if rows > 0:
        avg_row_size = total_size / rows
        remaining_size = len(data) - header_size
        estimated_total = int(remaining_size / avg_row_size)
        return estimated_total
    
    return 0


def main():
    """メイン関数: コマンドライン引数を解析して実行"""
    if len(sys.argv) < 2:
        print(f"使い方: {sys.argv[0]} binary_file.bin [rows] [max_bytes]")
        return 1
        
    filename = sys.argv[1]
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_bytes = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    
    if not os.path.exists(filename):
        print(f"エラー: ファイル '{filename}' が見つかりません")
        return 1
        
    try:
        # バイナリファイルを読み込み
        data = np.fromfile(filename, dtype=np.uint8)
        print(f"ファイル '{filename}' を読み込みました ({len(data)} バイト)")
        
        # ヘッダー解析
        header_size = dump_pg_header(data)
        if header_size == 0:
            print("ヘッダー解析に失敗しました")
            return 1
            
        # 行データの解析
        parse_rows(data, header_size, max_rows, max_bytes)
        
        return 0
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
