#!/usr/bin/env python3
"""
チャンクファイル内の実際のタプル数をカウント
"""

import struct
import time

def count_tuples_in_chunk(chunk_file):
    """チャンクファイル内のタプル数を正確にカウント"""
    with open(chunk_file, 'rb') as f:
        # ヘッダーをスキップ
        header = f.read(19)
        if not header.startswith(b'PGCOPY\n\377\r\n\0'):
            print("無効なPostgreSQL COPYファイル")
            return 0
        
        tuple_count = 0
        error_count = 0
        start_time = time.time()
        
        while True:
            if tuple_count % 100000 == 0 and tuple_count > 0:
                elapsed = time.time() - start_time
                print(f"進行状況: {tuple_count:,} タプル ({elapsed:.1f}秒)")
            
            # タプルヘッダー読み込み
            tuple_header = f.read(2)
            if len(tuple_header) < 2:
                break
            
            num_fields = struct.unpack('>h', tuple_header)[0]
            
            # 終了マーカー
            if num_fields == -1:
                break
            
            if num_fields != 17:  # lineorderは17フィールド
                print(f"警告: 位置 {f.tell() - 2} で異常なフィールド数: {num_fields}")
                error_count += 1
                if error_count > 10:
                    break
                continue
            
            # フィールドを読んでスキップ
            try:
                for i in range(num_fields):
                    field_size_data = f.read(4)
                    if len(field_size_data) < 4:
                        raise EOFError
                    field_size = struct.unpack('>i', field_size_data)[0]
                    if field_size > 0:
                        f.seek(field_size, 1)
                    elif field_size < -1:
                        raise ValueError(f"無効なフィールドサイズ: {field_size}")
                
                tuple_count += 1
                
            except Exception as e:
                print(f"エラー at {f.tell()}: {e}")
                error_count += 1
                if error_count > 10:
                    break
        
        return tuple_count

def main():
    chunk_file = "/dev/shm/chunk_0.bin"
    print(f"チャンクファイル: {chunk_file}")
    print("タプル数をカウント中...")
    
    actual_tuples = count_tuples_in_chunk(chunk_file)
    
    print(f"\n結果:")
    print(f"実際のタプル数: {actual_tuples:,}")
    print(f"Rust推定: 15,578,741")
    print(f"GPU処理: 9,331,177")
    print(f"差分: {actual_tuples - 9331177:,} タプル")

if __name__ == "__main__":
    main()