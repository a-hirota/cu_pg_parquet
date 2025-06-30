#!/usr/bin/env python3
"""
チャンク境界での行処理を詳細に分析

16チャンク時の9700万行欠損問題を調査するため、
各チャンクの境界部分でのデータを詳細に分析する
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size
from src.types import ColumnMeta


def analyze_chunk_boundaries(chunk_files: List[str], columns: List[ColumnMeta]):
    """各チャンクの境界部分を分析"""
    results = []
    
    print("=== チャンク境界分析 ===\n")
    
    for i, chunk_file in enumerate(chunk_files):
        if not os.path.exists(chunk_file):
            print(f"チャンク{i}: ファイルが存在しません - {chunk_file}")
            continue
            
        file_size = os.path.getsize(chunk_file)
        print(f"チャンク{i}: {chunk_file}")
        print(f"  ファイルサイズ: {file_size:,} bytes ({file_size/1024**3:.2f} GB)")
        
        with open(chunk_file, 'rb') as f:
            # ヘッダー部分を読み込み
            header_data = f.read(128)
            header_size = detect_pg_header_size(header_data)
            print(f"  ヘッダーサイズ: {header_size} bytes")
            
            # ファイルの最後の1MBを読み込み
            tail_size = min(1024*1024, file_size)
            f.seek(-tail_size, 2)
            tail_position = f.tell()
            tail_data = f.read()
            
            print(f"  最後の{tail_size:,}バイトを分析 (位置: {tail_position:,})")
            
            # 最後の行を探す（PostgreSQL binary formatの行マーカーを探す）
            # 行の開始は通常、フィールド数(2バイト)から始まる
            last_row_info = find_last_complete_row(tail_data, len(columns))
            
            if last_row_info:
                print(f"  最後の完全な行: 位置={tail_position + last_row_info['offset']:,}, サイズ={last_row_info['size']}バイト")
            else:
                print(f"  ⚠️  最後の完全な行を検出できませんでした")
            
            # 次のチャンクの最初を確認
            if i < len(chunk_files) - 1:
                next_chunk = chunk_files[i + 1]
                if os.path.exists(next_chunk):
                    with open(next_chunk, 'rb') as f2:
                        # 次のチャンクのヘッダーと最初の部分を読み込み
                        next_header = f2.read(128)
                        next_header_size = detect_pg_header_size(next_header)
                        
                        # 最初のデータ部分
                        f2.seek(next_header_size)
                        first_data = f2.read(1024*1024)
                        
                        first_row_info = find_first_row(first_data, len(columns))
                        
                        if first_row_info:
                            print(f"  次のチャンクの最初の行: オフセット={next_header_size + first_row_info['offset']}, サイズ={first_row_info['size']}バイト")
                        
                        # データの連続性をチェック
                        if last_row_info and first_row_info:
                            # 最後の行の終端と次の最初の行の開始の間隔を確認
                            gap = next_header_size + first_row_info['offset']
                            print(f"  チャンク間ギャップ: {gap}バイト")
                            
                            if gap > 100:  # 100バイト以上の隙間がある場合は警告
                                print(f"  ⚠️  チャンク間に大きなギャップを検出！")
            
            results.append({
                'chunk_id': i,
                'file_size': file_size,
                'header_size': header_size,
                'last_row_info': last_row_info,
                'chunk_file': chunk_file
            })
        
        print()
    
    return results


def find_last_complete_row(data: bytes, ncols: int) -> Dict[str, Any]:
    """バイナリデータから最後の完全な行を探す"""
    # PostgreSQL binary formatでは、各行は:
    # - フィールド数 (2バイト, int16)
    # - 各フィールド: 長さ(4バイト, int32) + データ
    
    # 後ろから探索
    pos = len(data) - 1
    
    while pos > ncols * 6:  # 最小限必要なバイト数
        # フィールド数の候補を探す
        if pos >= 2:
            field_count = int.from_bytes(data[pos-1:pos+1], 'big')
            if field_count == ncols:
                # この位置から行を検証
                row_start = pos - 1
                row_size = validate_row_at_position(data, row_start, ncols)
                
                if row_size > 0:
                    return {
                        'offset': row_start,
                        'size': row_size,
                        'field_count': ncols
                    }
        
        pos -= 1
    
    return None


def find_first_row(data: bytes, ncols: int) -> Dict[str, Any]:
    """バイナリデータから最初の行を探す"""
    pos = 0
    
    while pos < len(data) - ncols * 6:
        # フィールド数をチェック
        if pos + 2 <= len(data):
            field_count = int.from_bytes(data[pos:pos+2], 'big')
            if field_count == ncols:
                # この位置から行を検証
                row_size = validate_row_at_position(data, pos, ncols)
                
                if row_size > 0:
                    return {
                        'offset': pos,
                        'size': row_size,
                        'field_count': ncols
                    }
        
        pos += 1
    
    return None


def validate_row_at_position(data: bytes, pos: int, expected_ncols: int) -> int:
    """指定位置から行を検証し、行サイズを返す"""
    if pos + 2 > len(data):
        return 0
    
    field_count = int.from_bytes(data[pos:pos+2], 'big')
    if field_count != expected_ncols:
        return 0
    
    current_pos = pos + 2
    
    for i in range(field_count):
        if current_pos + 4 > len(data):
            return 0
        
        field_length = int.from_bytes(data[current_pos:current_pos+4], 'big', signed=True)
        current_pos += 4
        
        if field_length < -1:  # -1はNULL値
            return 0
        
        if field_length > 0:
            if current_pos + field_length > len(data):
                return 0
            current_pos += field_length
    
    return current_pos - pos


def main():
    """メイン処理"""
    # チャンクファイルを検索
    chunk_files = []
    for i in range(32):  # 最大32チャンクまで検索
        chunk_file = f"/dev/shm/chunk_{i}.bin"
        if os.path.exists(chunk_file):
            chunk_files.append(chunk_file)
    
    if not chunk_files:
        print("チャンクファイルが見つかりません")
        return
    
    print(f"見つかったチャンクファイル: {len(chunk_files)}個\n")
    
    # メタデータファイルから列情報を取得
    meta_file = f"/dev/shm/lineorder_meta_0.json"
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
            ncols = metadata.get('ncols', 17)  # lineorderテーブルは17列
            print(f"列数: {ncols}\n")
    else:
        print("メタデータファイルが見つかりません。デフォルトの17列を使用します。")
        ncols = 17
    
    # ダミーの列情報（実際の分析には影響しない）
    columns = [None] * ncols
    
    # 境界分析を実行
    results = analyze_chunk_boundaries(chunk_files, columns)
    
    # サマリー
    print("\n=== 分析サマリー ===")
    print(f"分析したチャンク数: {len(results)}")
    
    incomplete_chunks = [r for r in results if not r.get('last_row_info')]
    if incomplete_chunks:
        print(f"\n⚠️  不完全な行で終わっているチャンク: {len(incomplete_chunks)}個")
        for chunk in incomplete_chunks:
            print(f"  - チャンク{chunk['chunk_id']}")


if __name__ == "__main__":
    main()