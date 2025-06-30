#!/usr/bin/env python3
"""
チャンクファイル内のデータ重複を検証
"""

import os
import struct
import hashlib

def find_duplicate_patterns(chunk_file, sample_size=1000000):
    """ファイル内の重複パターンを検出"""
    if not os.path.exists(chunk_file):
        print(f"ファイルが存在しません: {chunk_file}")
        return
        
    file_size = os.path.getsize(chunk_file)
    print(f"ファイル: {chunk_file}")
    print(f"サイズ: {file_size:,} bytes")
    
    # ファイルの異なる位置からサンプルを取得
    positions = [
        0,                          # 開始位置
        file_size // 4,             # 1/4地点
        file_size // 2,             # 中間地点
        3 * file_size // 4,         # 3/4地点
        file_size - sample_size     # 終端付近
    ]
    
    samples = {}
    with open(chunk_file, 'rb') as f:
        for i, pos in enumerate(positions):
            if pos < 0 or pos + sample_size > file_size:
                continue
            f.seek(pos)
            data = f.read(sample_size)
            hash_val = hashlib.md5(data).hexdigest()
            samples[f"位置{i} ({pos:,})"] = {
                'hash': hash_val,
                'first_bytes': data[:100].hex()[:50] + '...'
            }
    
    print("\n【サンプルハッシュ値】")
    for name, info in samples.items():
        print(f"{name}: {info['hash']}")
    
    # 重複をチェック
    hash_counts = {}
    for name, info in samples.items():
        h = info['hash']
        if h not in hash_counts:
            hash_counts[h] = []
        hash_counts[h].append(name)
    
    print("\n【重複パターン】")
    duplicates_found = False
    for hash_val, positions in hash_counts.items():
        if len(positions) > 1:
            duplicates_found = True
            print(f"同一データが検出された位置: {', '.join(positions)}")
    
    if not duplicates_found:
        print("重複パターンは検出されませんでした")
    
    # PostgreSQL COPYバイナリデータの連続性をチェック
    print("\n【データ連続性チェック】")
    with open(chunk_file, 'rb') as f:
        # ヘッダーをスキップ
        f.seek(19)  # 基本ヘッダーサイズ
        
        tuple_count = 0
        error_count = 0
        
        while tuple_count < 100:  # 最初の100タプルをチェック
            pos = f.tell()
            tuple_header = f.read(2)
            if len(tuple_header) < 2:
                break
                
            num_fields = struct.unpack('>h', tuple_header)[0]
            
            if num_fields != 17:  # lineorderは17フィールド
                print(f"位置 {pos}: 異常なフィールド数 {num_fields}")
                error_count += 1
                if error_count > 5:
                    break
                # 1バイトずつ進めて再試行
                f.seek(pos + 1)
                continue
            
            # フィールドを読んでスキップ
            try:
                for i in range(num_fields):
                    field_size_data = f.read(4)
                    if len(field_size_data) < 4:
                        raise EOFError
                    field_size = struct.unpack('>i', field_size_data)[0]
                    if field_size < -1 or field_size > 1000000:  # 異常なサイズ
                        raise ValueError(f"異常なフィールドサイズ: {field_size}")
                    if field_size > 0:
                        f.seek(field_size, 1)
                tuple_count += 1
            except Exception as e:
                print(f"位置 {pos}: エラー - {e}")
                error_count += 1
                if error_count > 5:
                    break
                f.seek(pos + 1)
        
        print(f"正常に読み取れたタプル数: {tuple_count}")
        print(f"エラー数: {error_count}")

def main():
    chunk_file = "/dev/shm/chunk_0.bin"
    if os.path.exists(chunk_file):
        find_duplicate_patterns(chunk_file)
    else:
        print("チャンクファイルが存在しません。先にbenchmark_rust_gpu_direct.pyを実行してください。")

if __name__ == "__main__":
    main()