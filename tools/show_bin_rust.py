#!/usr/bin/env python3
"""
Rust Binary File Viewer
======================

test_binariesフォルダ内の最新ディレクトリから、指定された位置のバイナリデータを
16進数とASCII表示で確認するツール。

Usage:
    python tools/show_rust_bin.py [--start START] [--end END] [--chunk CHUNK_ID] [--dir DIR]

Examples:
    # 最新のバイナリファイルの先頭512バイトを表示
    python tools/show_rust_bin.py

    # 特定の位置を表示
    python tools/show_rust_bin.py --start 1024 --end 1536

    # chunk_1を指定
    python tools/show_rust_bin.py --chunk 1 --start 0 --end 256
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def find_latest_test_directory(base_path="test_binaries"):
    """test_binaries内の最新のタイムスタンプディレクトリを見つける"""
    if not os.path.exists(base_path):
        print(f"Error: {base_path} directory not found")
        return None

    dirs = []
    for d in os.listdir(base_path):
        full_path = os.path.join(base_path, d)
        if os.path.isdir(full_path):
            try:
                # タイムスタンプ形式: YYYYMMDD_HHMMSS
                datetime.strptime(d, "%Y%m%d_%H%M%S")
                dirs.append(d)
            except ValueError:
                continue

    if not dirs:
        print(f"No timestamp directories found in {base_path}")
        return None

    # 最新のディレクトリを返す
    latest = sorted(dirs)[-1]
    return os.path.join(base_path, latest)


def find_bin_file(directory, chunk_id=None):
    """指定されたディレクトリからbinファイルを見つける"""
    bin_files = [f for f in os.listdir(directory) if f.endswith(".bin")]

    if not bin_files:
        print(f"No .bin files found in {directory}")
        return None

    if chunk_id is not None:
        # chunk_idが指定された場合、該当するファイルを探す
        target = f"_chunk_{chunk_id}.bin"
        for f in bin_files:
            if target in f:
                return os.path.join(directory, f)
        print(f"No file with chunk_id={chunk_id} found")
        return None

    # chunk_idが指定されていない場合、最初のファイルを返す
    bin_files.sort()
    return os.path.join(directory, bin_files[0])


def display_binary_data(file_path, start, end):
    """バイナリデータを16進数とASCII表示"""
    try:
        with open(file_path, "rb") as f:
            # ファイルサイズを確認
            f.seek(0, 2)
            file_size = f.tell()

            # 範囲の調整
            if end > file_size:
                print(f"Warning: end position {end} exceeds file size {file_size}")
                end = file_size

            if start >= file_size:
                print(f"Error: start position {start} exceeds file size {file_size}")
                return

            # 指定位置へ移動してデータ読み取り
            f.seek(start)
            data = f.read(end - start)

            print(f"\nFile: {file_path}")
            print(f"File size: {file_size:,} bytes")
            print(f"Displaying: [{start:,} - {end:,}] ({end-start} bytes)\n")

            # ヘッダー
            print("Offset    00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F  |ASCII Text      |")
            print("-" * 77)

            # 16バイトずつ表示
            for i in range(0, len(data), 16):
                offset = start + i
                chunk = data[i : i + 16]

                # オフセット表示
                hex_part = f"{offset:08X}  "

                # 16進数表示（8バイトごとにスペース）
                for j in range(16):
                    if j < len(chunk):
                        hex_part += f"{chunk[j]:02X} "
                    else:
                        hex_part += "   "
                    if j == 7:
                        hex_part += " "

                # ASCII表示
                ascii_part = "|"
                for j in range(len(chunk)):
                    b = chunk[j]
                    # 印字可能文字の範囲（32-126）
                    if 32 <= b <= 126:
                        ascii_part += chr(b)
                    else:
                        ascii_part += "."
                ascii_part += " " * (16 - len(chunk))
                ascii_part += "|"

                print(hex_part + " " + ascii_part)

            print()

            # PostgreSQL COPY BINARYフォーマットのヘッダー解析（最初の場合）
            if start == 0 and len(data) >= 11:
                print("=== PostgreSQL COPY BINARY Header Analysis ===")
                signature = data[0:11]
                print(f"Signature: {signature.hex()} ({repr(signature)})")
                if signature == b"PGCOPY\\n\\377\\r\\n\\000":
                    print("✓ Valid PostgreSQL COPY BINARY signature detected")
                else:
                    print("✗ Invalid signature (expected: 5047434f50590a377f0d0a00)")

                if len(data) >= 19:
                    flags = int.from_bytes(data[11:15], "big")
                    extension_area_length = int.from_bytes(data[15:19], "big")
                    print(f"Flags field: {flags}")
                    print(f"Header extension area length: {extension_area_length}")

                    if len(data) >= 19 + extension_area_length + 2:
                        col_count = int.from_bytes(
                            data[19 + extension_area_length : 19 + extension_area_length + 2], "big"
                        )
                        print(f"Column count: {col_count}")

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")


def search_binary_pattern(file_path, hex_pattern):
    """バイナリファイル内で16進数パターンを検索"""
    # 16進数文字列をバイト配列に変換
    hex_bytes = hex_pattern.replace(" ", "").strip()
    pattern = bytes.fromhex(hex_bytes)

    # ファイル名のみ表示（フルパスではなく）
    filename = os.path.basename(file_path)
    print(f"\nSearching in: {filename}")
    print(f"Pattern: {hex_pattern} ({len(pattern)} bytes)")

    found_positions = []

    try:
        with open(file_path, "rb") as f:
            # ファイル全体を読み込む（大きなファイルの場合は分割読み込みが必要）
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(0)

            # チャンクごとに読み込んで検索（メモリ効率のため）
            chunk_size = 10 * 1024 * 1024  # 10MB chunks
            overlap = len(pattern) - 1  # パターンが境界をまたぐ場合のオーバーラップ

            position = 0
            while position < file_size:
                f.seek(position)
                chunk = f.read(chunk_size + overlap)

                # チャンク内でパターンを検索
                offset = 0
                while True:
                    index = chunk.find(pattern, offset)
                    if index == -1:
                        break

                    actual_position = position + index
                    found_positions.append(actual_position)
                    offset = index + 1

                position += chunk_size
                if position + overlap >= file_size:
                    break

        # 結果の表示
        if found_positions:
            print(f"Found {len(found_positions)} occurrence(s):\n")
            for i, pos in enumerate(found_positions):
                print(f"  {i+1}. Offset 0x{pos:08X} ({pos:,} bytes)")

                # 前後のコンテキストを表示
                with open(file_path, "rb") as f:
                    context_start = max(0, pos - 16)
                    f.seek(context_start)
                    context_data = f.read(len(pattern) + 32)

                    print(f"     Context:")
                    # 16進数表示
                    for j in range(0, len(context_data), 16):
                        offset = context_start + j
                        chunk_data = context_data[j : j + 16]
                        hex_part = f"     {offset:08X}  "

                        for k in range(16):
                            if k < len(chunk_data):
                                # パターン部分をハイライト
                                if pos <= offset + k < pos + len(pattern):
                                    hex_part += f"\033[1;31m{chunk_data[k]:02X}\033[0m "
                                else:
                                    hex_part += f"{chunk_data[k]:02X} "
                            else:
                                hex_part += "   "
                            if k == 7:
                                hex_part += " "

                        # ASCII表示
                        ascii_part = "|"
                        for k in range(len(chunk_data)):
                            b = chunk_data[k]
                            if 32 <= b <= 126:
                                if pos <= offset + k < pos + len(pattern):
                                    ascii_part += f"\033[1;31m{chr(b)}\033[0m"
                                else:
                                    ascii_part += chr(b)
                            else:
                                ascii_part += "."
                        ascii_part += " " * (16 - len(chunk_data))
                        ascii_part += "|"

                        print(hex_part + " " + ascii_part)
                    print()
        else:
            print("Pattern not found in file.")

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        print(f"Error searching file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="View binary data from PostgreSQL COPY BINARY files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--start", type=int, default=0, help="Start position in bytes (default: 0)")
    parser.add_argument("--end", type=int, default=512, help="End position in bytes (default: 512)")
    parser.add_argument(
        "--chunk", type=int, default=None, help="Chunk ID to display (e.g., 0, 1, 2...)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Specific directory path (default: latest in test_binaries)",
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Specific file path (overrides --dir and --chunk)"
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help='Search for hex pattern (e.g., "50 47 43 4F 50 59 0A FF 0D 0A")',
    )

    args = parser.parse_args()

    # ファイルパスの決定
    if args.file:
        file_path = args.file
        # 検索モードか表示モードか
        if args.search:
            search_binary_pattern(file_path, args.search)
        else:
            # バイナリデータの表示
            display_binary_data(file_path, args.start, args.end)
    else:
        # ディレクトリの決定
        if args.dir:
            directory = args.dir
        else:
            directory = find_latest_test_directory()
            if not directory:
                sys.exit(1)

        print(f"Using directory: {directory}")

        # 検索モードの場合は全ファイルを検索
        if args.search:
            # ディレクトリ内のすべての.binファイルを取得
            bin_files = sorted([f for f in os.listdir(directory) if f.endswith(".bin")])

            if not bin_files:
                print(f"No .bin files found in {directory}")
                sys.exit(1)

            print(f"Found {len(bin_files)} .bin file(s) to search:")
            for f in bin_files:
                print(f"  - {f}")
            print()

            # 全体の統計を収集
            total_matches = 0
            file_results = []

            # 各ファイルで検索
            for bin_file in bin_files:
                file_path = os.path.join(directory, bin_file)
                print(f"\n{'='*70}")
                print(f"Searching: {bin_file}")
                print(f"{'='*70}")

                # 一時的に標準出力をキャプチャして結果を取得
                import io
                from contextlib import redirect_stdout

                f = io.StringIO()
                with redirect_stdout(f):
                    search_binary_pattern(file_path, args.search)
                output = f.getvalue()
                print(output)

                # マッチ数をカウント
                if "Found" in output and "occurrence(s):" in output:
                    import re

                    match = re.search(r"Found (\d+) occurrence\(s\):", output)
                    if match:
                        count = int(match.group(1))
                        total_matches += count
                        file_results.append((bin_file, count))
                else:
                    file_results.append((bin_file, 0))

            # サマリー表示
            print(f"\n{'='*70}")
            print(f"SEARCH SUMMARY")
            print(f"{'='*70}")
            print(f"Pattern: {args.search}")
            print(f"Total matches: {total_matches}")
            print(f"\nPer file:")
            for fname, count in file_results:
                if count > 0:
                    print(f"  {fname}: {count} match(es)")
                else:
                    print(f"  {fname}: no matches")
        else:
            # 表示モードの場合は指定されたチャンクまたは最初のファイルを表示
            file_path = find_bin_file(directory, args.chunk)
            if not file_path:
                sys.exit(1)
            display_binary_data(file_path, args.start, args.end)


if __name__ == "__main__":
    main()
