#!/usr/bin/env python3
"""
行ヘッダ検出パターンのテスト
========================

偶数/奇数行の検出パターンを詳細に調査
"""

import numpy as np
from numba import cuda
import os
import sys

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size


@cuda.jit
def analyze_row_headers(raw_data, header_size, ncols, results):
    """行ヘッダの出現パターンを分析"""
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # 各スレッドが1MBずつ処理
    chunk_size = 1024 * 1024
    start_pos = header_size + tid * chunk_size
    end_pos = min(start_pos + chunk_size, raw_data.size)
    
    if start_pos >= raw_data.size:
        return
    
    # 結果記録用
    # results[tid, 0] = 検出した行数
    # results[tid, 1] = 偶数位置の行数
    # results[tid, 2] = 奇数位置の行数
    # results[tid, 3] = 最初の行位置
    # results[tid, 4] = 最後の行位置
    
    row_count = 0
    even_count = 0
    odd_count = 0
    first_row = -1
    last_row = -1
    
    pos = start_pos
    while pos < end_pos - 1:
        # 行ヘッダチェック（2バイト）
        if pos + 1 < raw_data.size:
            num_fields = (raw_data[pos] << 8) | raw_data[pos + 1]
            
            if num_fields == ncols:
                # 行ヘッダ発見
                row_count += 1
                
                if first_row == -1:
                    first_row = pos
                last_row = pos
                
                # 位置の偶奇判定
                if pos % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
                
                # 次の行へ（簡易的に230バイト進む）
                pos += 230
            else:
                pos += 1
        else:
            break
    
    # 結果記録
    results[tid, 0] = row_count
    results[tid, 1] = even_count
    results[tid, 2] = odd_count
    results[tid, 3] = first_row
    results[tid, 4] = last_row


def create_test_binary_data(rows=1000, ncols=17):
    """テスト用のPostgreSQLバイナリデータを作成"""
    # ヘッダー
    header = b"PGCOPY\n\377\r\n\0"  # 11 bytes
    header += b"\x00\x00\x00\x00"   # flags (4 bytes)
    header += b"\x00\x00\x00\x00"   # header extension (4 bytes)
    
    data = bytearray(header)
    
    # 各行のデータ
    for i in range(rows):
        # 行ヘッダ（フィールド数）
        data.extend(ncols.to_bytes(2, 'big'))
        
        # 各フィールド
        for j in range(ncols):
            # フィールド長（4バイト）
            field_len = 8  # 固定長8バイト
            data.extend(field_len.to_bytes(4, 'big'))
            
            # フィールドデータ
            data.extend(b'\x00' * field_len)
    
    # 終端マーカー
    data.extend(b'\xff\xff')
    
    return np.frombuffer(data, dtype=np.uint8)


def analyze_detection_pattern():
    """行検出パターンを分析"""
    print("=== 行ヘッダ検出パターン分析 ===")
    
    # テストデータ作成
    test_data = create_test_binary_data(rows=1000, ncols=17)
    header_size = detect_pg_header_size(test_data)
    
    print(f"テストデータサイズ: {len(test_data):,} bytes")
    print(f"ヘッダーサイズ: {header_size} bytes")
    
    # GPU転送
    test_data_gpu = cuda.to_device(test_data)
    
    # 結果配列
    num_threads = 32
    results = np.zeros((num_threads, 5), dtype=np.int32)
    results_gpu = cuda.to_device(results)
    
    # GPU実行
    analyze_row_headers[32, 1](test_data_gpu, header_size, 17, results_gpu)
    cuda.synchronize()
    
    # 結果取得
    results = results_gpu.copy_to_host()
    
    # 結果分析
    print("\n=== スレッドごとの検出結果 ===")
    total_rows = 0
    total_even = 0
    total_odd = 0
    
    for tid in range(num_threads):
        if results[tid, 0] > 0:  # 行を検出したスレッド
            print(f"Thread {tid}: 行数={results[tid, 0]}, "
                  f"偶数位置={results[tid, 1]}, 奇数位置={results[tid, 2]}, "
                  f"最初={results[tid, 3]}, 最後={results[tid, 4]}")
            
            total_rows += results[tid, 0]
            total_even += results[tid, 1]
            total_odd += results[tid, 2]
    
    print(f"\n合計: 行数={total_rows}, 偶数位置={total_even}, 奇数位置={total_odd}")
    print(f"偶数/奇数比: {total_even}/{total_odd} = {total_even/total_odd if total_odd > 0 else 'N/A'}")
    
    # 位置の分布を確認
    print("\n=== 行開始位置の分布 ===")
    positions = []
    for tid in range(num_threads):
        if results[tid, 3] >= 0:
            positions.append(results[tid, 3])
    
    if positions:
        positions.sort()
        print(f"検出された行位置（最初の10個）: {positions[:10]}")
        
        # 連続する行の間隔を確認
        intervals = []
        for i in range(1, len(positions)):
            intervals.append(positions[i] - positions[i-1])
        
        if intervals:
            print(f"行間隔の平均: {np.mean(intervals):.1f} bytes")
            print(f"行間隔の範囲: {min(intervals)} - {max(intervals)} bytes")


def test_actual_lineorder_sample():
    """実際のlineorderデータのサンプルで検証"""
    print("\n\n=== 実際のlineorderデータサンプル分析 ===")
    
    # lineorderの最初の1MBを取得して分析
    import subprocess
    
    cmd = """
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && 
    conda activate cudf_dev && 
    python -c "
import psycopg2
import io
conn = psycopg2.connect('dbname=postgres user=postgres host=localhost port=5432')
cur = conn.cursor()
copy_sql = 'COPY (SELECT * FROM lineorder LIMIT 1000) TO STDOUT WITH BINARY'
output = io.BytesIO()
cur.copy_expert(copy_sql, output)
output.seek(0)
data = output.read()
print(f'Data size: {len(data)}')
# 最初の行ヘッダを探す
for i in range(19, min(1000, len(data)-1)):
    if i + 1 < len(data):
        num_fields = (data[i] << 8) | data[i + 1]
        if num_fields == 17:  # lineorderは17列
            print(f'First row header at position {i} ({"even" if i % 2 == 0 else "odd"})')
            # 次の10個の行ヘッダも探す
            pos = i
            count = 0
            while pos < len(data) - 1 and count < 10:
                if (data[pos] << 8) | data[pos + 1] == 17:
                    print(f'  Row {count+1} at {pos} ({"even" if pos % 2 == 0 else "odd"})')
                    count += 1
                    pos += 200  # 概算で次の行へ
                else:
                    pos += 1
            break
cur.close()
conn.close()
"
    """
    
    result = subprocess.run(
        cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")


if __name__ == "__main__":
    # 環境設定
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # パターン分析実行
    analyze_detection_pattern()
    test_actual_lineorder_sample()