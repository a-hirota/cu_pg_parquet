#!/usr/bin/env python3
"""
チャンクファイルのバイナリデータを分析
"""

import os
import struct
import subprocess

def analyze_chunk_file(chunk_file):
    """チャンクファイルのヘッダーを分析"""
    if not os.path.exists(chunk_file):
        print(f"ファイルが存在しません: {chunk_file}")
        return
        
    file_size = os.path.getsize(chunk_file)
    print(f"\nファイル: {chunk_file}")
    print(f"ファイルサイズ: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
    
    # 最初の128バイトを読み込んでヘッダーを確認
    with open(chunk_file, 'rb') as f:
        header = f.read(128)
        
        # PostgreSQL COPYバイナリヘッダーのシグネチャ
        # PGCOPY\n\377\r\n\0
        expected_sig = b'PGCOPY\n\377\r\n\0'
        
        if header.startswith(expected_sig):
            print("✅ 有効なPostgreSQL COPYバイナリヘッダーを検出")
            
            # フラグとヘッダー拡張を読む
            flags = struct.unpack('>I', header[11:15])[0]
            header_ext_len = struct.unpack('>I', header[15:19])[0]
            
            print(f"フラグ: {flags}")
            print(f"ヘッダー拡張長: {header_ext_len}")
            
            # 実際のデータ開始位置
            data_start = 19 + header_ext_len
            print(f"データ開始位置: {data_start}")
            
            # 最初の数タプルをチェック
            f.seek(data_start)
            print("\n最初の5タプルのサイズ:")
            for i in range(5):
                tuple_header = f.read(2)
                if len(tuple_header) < 2:
                    break
                num_fields = struct.unpack('>h', tuple_header)[0]
                print(f"  タプル{i}: {num_fields}フィールド")
                
                # フィールドサイズを読む
                field_sizes = []
                for j in range(num_fields):
                    field_size_data = f.read(4)
                    if len(field_size_data) < 4:
                        break
                    field_size = struct.unpack('>i', field_size_data)[0]
                    field_sizes.append(field_size)
                    # フィールドデータをスキップ
                    if field_size > 0:
                        f.seek(field_size, 1)
                
                if i == 0:
                    print(f"    フィールドサイズ: {field_sizes}")
        else:
            print("❌ 無効なヘッダー")
            print(f"最初の16バイト: {header[:16].hex()}")

def main():
    # チャンク0を作成してみる
    print("チャンク0を再度作成して分析します...")
    
    env = os.environ.copy()
    env['CHUNK_ID'] = '0'
    env['TOTAL_CHUNKS'] = '16'
    env['TABLE_NAME'] = 'lineorder'
    
    rust_binary = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk"
    
    # 実行
    subprocess.run([rust_binary], env=env, capture_output=True)
    
    # チャンクファイルを分析
    chunk_file = "/dev/shm/chunk_0.bin"
    analyze_chunk_file(chunk_file)
    
    # ファイルサイズとPostgreSQLの行数から1行あたりのバイト数を計算
    file_size = os.path.getsize(chunk_file)
    expected_rows = 15578741  # Rustが報告した推定行数
    
    print(f"\n【統計】")
    print(f"ファイルサイズ: {file_size:,} bytes")
    print(f"推定行数: {expected_rows:,}")
    print(f"1行あたり: {file_size / expected_rows:.1f} bytes/row")
    
    # ワーカーの重複を確認
    print("\n【ワーカー重複チェック】")
    print("すべてのワーカーのoffsetが0なので、データが重複している可能性があります")
    print("実際のデータサイズは約1/16になっている可能性が高いです")
    
    # 予想される実際のサイズ
    expected_actual_size = file_size / 16
    print(f"\n予想される実際のユニークデータサイズ: {expected_actual_size / 1024**3:.2f} GB")
    print(f"予想される実際の行数: {expected_rows / 16:,.0f}")

if __name__ == "__main__":
    main()