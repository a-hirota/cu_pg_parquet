"""
PostgreSQL COPY BINARY パーサーの単一行・複数行テスト

1行・2行のサンプルデータを使用して、パーサーカーネルが以下を正しく処理できるか検証:
- 正しいヘッダーサイズ検出
- NULL値の処理
- フィールド長とオフセットの正確な計算
- 複数行の処理 (row_start_positions を使用)
"""

import os
import numpy as np
import pytest
from numba import cuda

from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size


# 1行だけの最小限の COPY BINARY データを作成
def create_minimal_binary_data(ncols=3, null_positions=None):
    """最小限の COPY BINARY データを作成 (ヘッダー + 1行 + 終端マーカー)
    
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

# CPUで複数行の開始位置を計算するヘルパー関数 (修正版)
def calculate_row_starts_cpu(raw_data, header_size, num_rows):
    """
    CPU上でCOPY BINARYデータの各行の開始位置を計算します。
    ヘッダー後の潜在的なパディング/フラグをスキップし、
    各行のフィールドを正しく読み進めて次の行の開始位置を特定します。
    """
    print(f"[calculate_row_starts_cpu] Input: header_size={header_size}, num_rows={num_rows}, data_len={len(raw_data)}") # DEBUG
    row_starts = np.full(num_rows, -1, dtype=np.int32) # Initialize with -1 (invalid)
    pos = header_size
    array_size = len(raw_data)
    current_row_index = 0

    while current_row_index < num_rows and pos < array_size:
        print(f"[calculate_row_starts_cpu] Trying Row {current_row_index}: Current pos={pos}") # DEBUG

        # --- Find the actual start of the row data ---
        # Peek ahead to find the first valid num_fields
        # A valid num_fields should be > 0 and likely < some reasonable max (e.g., 1000)
        # This helps skip potential zero bytes or other non-field-count data after the header
        # or between rows.
        found_start = False
        original_search_pos = pos
        while pos + 2 <= array_size:
            potential_num_fields = (raw_data[pos] << 8) | raw_data[pos + 1]

            if potential_num_fields == 0xFFFF: # EOF marker
                 print(f"[calculate_row_starts_cpu] Found EOF marker while searching for row start at pos={pos}") # DEBUG
                 pos = array_size # Stop processing
                 break

            # Check for a reasonable number of fields
            if potential_num_fields > 0 and potential_num_fields < 1000: # Adjust 1000 if needed
                 print(f"[calculate_row_starts_cpu] Found potential row start with {potential_num_fields} fields at pos={pos}") # DEBUG
                 found_start = True
                 break # Found the likely start

            # If not a valid field count or EOF, advance by one byte and retry
            pos += 1
            print(f"[calculate_row_starts_cpu] Skipping byte, new search pos={pos}") # DEBUG

        if not found_start or pos >= array_size:
             print(f"[calculate_row_starts_cpu] Could not find valid row start after pos={original_search_pos}. Stopping.") # DEBUG
             break # Stop if no valid start found or reached end

        # --- Process the row starting at the found 'pos' ---
        row_starts[current_row_index] = pos
        print(f"[calculate_row_starts_cpu] Row {current_row_index}: Stored start_pos={pos}") # DEBUG

        # Read the confirmed number of fields
        if pos + 2 > array_size: # Should not happen if found_start is True, but safety check
            print(f"[calculate_row_starts_cpu] Row {current_row_index}: Cannot read num_fields at confirmed start pos={pos}") # DEBUG
            row_starts[current_row_index] = -1 # Mark as invalid
            break

        num_fields = (raw_data[pos] << 8) | raw_data[pos + 1]
        print(f"[calculate_row_starts_cpu] Row {current_row_index}: Confirmed num_fields={num_fields} at pos={pos}") # DEBUG
        pos += 2 # Advance past num_fields

        # --- Inner loop to skip fields and find next row's start ---
        print(f"[calculate_row_starts_cpu] Row {current_row_index}: Starting inner loop for {num_fields} fields, pos={pos}") # DEBUG
        inner_loop_broken = False
        for field_idx in range(num_fields):
            # Check if we can read field length
            if pos + 4 > array_size:
                print(f"[calculate_row_starts_cpu] Row {current_row_index}, Field {field_idx}: Cannot read field_len at pos={pos}") # DEBUG
                inner_loop_broken = True
                pos = array_size # Ensure outer loop terminates
                break # Break inner loop

            # Read field length
            field_len = int.from_bytes(raw_data[pos:pos+4], 'big', signed=True)
            # print(f"[calculate_row_starts_cpu] Row {current_row_index}, Field {field_idx}: Read field_len={field_len} at pos={pos}") # DEBUG (Optional)
            pos += 4 # Advance past field_len

            # If field is not NULL, advance past data
            if field_len >= 0:
                # Check if we can read field data
                if pos + field_len > array_size:
                    print(f"[calculate_row_starts_cpu] Row {current_row_index}, Field {field_idx}: Cannot read field data (len={field_len}) at pos={pos}") # DEBUG
                    inner_loop_broken = True
                    pos = array_size # Ensure outer loop terminates
                    break # Break inner loop
                pos += field_len # Advance past field data
                # print(f"[calculate_row_starts_cpu] Row {current_row_index}, Field {field_idx}: Advanced pos to {pos}") # DEBUG (Optional)
            # else: # NULL case, pos already advanced by 4

        print(f"[calculate_row_starts_cpu] Row {current_row_index}: Finished inner loop, final pos for row (start of next row)={pos}") # DEBUG

        # Move to the next row index
        current_row_index += 1

        # If the inner loop was broken due to data ending, stop processing further rows
        if inner_loop_broken:
             print(f"[calculate_row_starts_cpu] Row {current_row_index-1}: Inner loop broke. Stopping.") # DEBUG
             break

    # Mark any remaining rows (if loop exited early) as invalid
    if current_row_index < num_rows:
        print(f"[calculate_row_starts_cpu] Marking rows from {current_row_index} to {num_rows-1} as invalid.") # DEBUG
        row_starts[current_row_index:] = -1

    print(f"[calculate_row_starts_cpu] Final calculated row_starts: {row_starts}") # DEBUG
    return row_starts


def test_single_row_with_null():
    """NULL値を含む1行データのパースをテスト"""
    # 3列、うち2列目がNULLのテストデータ
    ncols = 3
    raw_host = create_minimal_binary_data(ncols=ncols, null_positions=[1])
    
    # ヘッダーサイズを検出
    header_size = detect_pg_header_size(raw_host)
    assert header_size == 11, f"ヘッダーサイズが想定と異なります: {header_size}"
    
    # デバッグ: バイナリデータを16進数でダンプ
    print("バイナリデータ構造:")
    for i in range(0, min(len(raw_host), 64), 16):
        hex_values = " ".join(f"{b:02x}" for b in raw_host[i:i+16])
        ascii_values = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw_host[i:i+16])
        print(f"{i:04x}: {hex_values} | {ascii_values}")
    
    # 行データの開始位置を計算
    data_start = header_size
    print(f"データ部分開始位置: {data_start}")
    
    # 先頭行のフィールド数を確認
    if data_start + 2 <= len(raw_host):
        num_fields = (raw_host[data_start] << 8) | raw_host[data_start + 1]
        print(f"検出したフィールド数: {num_fields}")
    
    # GPUメモリに転送
    raw_dev = cuda.to_device(raw_host)
    
    # 1行だけパース
    rows = 1
    
    # ヘッダー後の位置から直接開始するよう設定
    row_start_positions_host = calculate_row_starts_cpu(raw_host, header_size, rows)
    row_start_positions = cuda.to_device(row_start_positions_host)
    print(f"CPU calculated row_starts: {row_start_positions_host}") # Debug print
    
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev, rows, ncols,
        header_size=header_size,
        row_start_positions=row_start_positions
    )
    
    
    # 結果をホストメモリに転送
    field_offsets = field_offsets_dev.copy_to_host()
    field_lengths = field_lengths_dev.copy_to_host()
    
    print("\nパース結果:")
    print("Field offsets:", field_offsets)
    print("Field lengths:", field_lengths)
    
    # 検証
    # 1列目: 非NULL
    assert field_lengths[0, 0] == 4, f"1列目の長さが不正: {field_lengths[0, 0]}"
    assert field_offsets[0, 0] > 0, f"1列目のオフセットが不正: {field_offsets[0, 0]}"
    
    # 2列目: NULL
    assert field_lengths[0, 1] == -1, f"2列目の長さが不正: {field_lengths[0, 1]}"
    assert field_offsets[0, 1] == 0, f"2列目のオフセットが不正: {field_offsets[0, 1]}"
    
    # 3列目: 非NULL
    assert field_lengths[0, 2] == 4, f"3列目の長さが不正: {field_lengths[0, 2]}"
    assert field_offsets[0, 2] > 0, f"3列目のオフセットが不正: {field_offsets[0, 2]}"


def test_two_rows():
    """2行データのパースをテスト（2行目もパースできることを確認）"""
    # 基本の1行データを作成
    ncols = 2
    base_data = create_minimal_binary_data(ncols=ncols, null_positions=[])
    
    # 終端マーカーを削除
    base_data = base_data[:-2]
    
    # 2行目を追加 (カラム1=NULL)
    row2_parts = []
    row2_parts.append(np.array([0, ncols], dtype=np.uint8))  # フィールド数
    
    # 1列目: NULL
    row2_parts.append(np.array([255, 255, 255, 255], dtype=np.uint8))
    
    # 2列目: 適当なデータ
    field_len = 4
    row2_parts.append(np.array([0, 0, 0, field_len], dtype=np.uint8))
    row2_parts.append(np.full(field_len, 42, dtype=np.uint8))
    
    # 終端マーカーを追加
    row2_parts.append(np.array([255, 255], dtype=np.uint8))
    
    # 全てを結合
    raw_host = np.concatenate([base_data] + row2_parts)
    
    # ヘッダーサイズを検出
    header_size = detect_pg_header_size(raw_host)
    
    # バイナリデータをダンプ
    print("\nバイナリデータ構造 (2行):")
    for i in range(0, min(len(raw_host), 64), 16):
        hex_values = " ".join(f"{b:02x}" for b in raw_host[i:i+16])
        ascii_values = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw_host[i:i+16])
        print(f"{i:04x}: {hex_values} | {ascii_values}")
    
    # 行データの開始位置を計算
    data_start = header_size
    print(f"データ部分開始位置: {data_start}")
    
    # GPUメモリに転送
    raw_dev = cuda.to_device(raw_host)
    
    # 2行パース
    rows = 2
    # CPUで開始位置を計算して渡す
    row_start_positions_host = calculate_row_starts_cpu(raw_host, header_size, rows)
    row_start_positions = cuda.to_device(row_start_positions_host)
    print(f"CPU calculated row_starts: {row_start_positions_host}") # Debug print

    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev, rows, ncols, header_size=header_size,
        row_start_positions=row_start_positions # Pass calculated positions
    )
    
    # 結果をホストメモリに転送
    field_offsets = field_offsets_dev.copy_to_host()
    field_lengths = field_lengths_dev.copy_to_host()
    
    print("Field offsets (2行):", field_offsets)
    print("Field lengths (2行):", field_lengths)
    
    # 検証 - 1行目
    assert field_lengths[0, 0] == 4, f"1行目1列目の長さが不正: {field_lengths[0, 0]}"
    assert field_lengths[0, 1] == 4, f"1行目2列目の長さが不正: {field_lengths[0, 1]}"
    
    # 検証 - 2行目
    assert field_lengths[1, 0] == -1, f"2行目1列目の長さが不正: {field_lengths[1, 0]}"
    assert field_offsets[1, 0] == 0, f"2行目1列目のオフセットが不正: {field_offsets[1, 0]}"
    
    assert field_lengths[1, 1] == 4, f"2行目2列目の長さが不正: {field_lengths[1, 1]}"
    assert field_offsets[1, 1] > 0, f"2行目2列目のオフセットが不正: {field_offsets[1, 1]}"


if __name__ == "__main__":
    print("単一行テスト実行...")
    test_single_row_with_null()
    print("\n2行テスト実行...")
    test_two_rows()
    print("\n全テスト正常終了")
