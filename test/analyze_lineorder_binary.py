#!/usr/bin/env python3
"""
lineorderのバイナリデータパターン分析
==================================
"""

import psycopg2
import io
import numpy as np

def analyze_lineorder_binary():
    """lineorderのバイナリデータを分析"""
    conn = psycopg2.connect('dbname=postgres user=postgres host=localhost port=5432')
    cur = conn.cursor()
    
    # バイナリデータ取得
    copy_sql = 'COPY (SELECT * FROM lineorder LIMIT 5000) TO STDOUT WITH BINARY'
    output = io.BytesIO()
    cur.copy_expert(copy_sql, output)
    output.seek(0)
    data = output.read()
    
    print(f"=== lineorderバイナリデータ分析 ===")
    print(f"データサイズ: {len(data):,} bytes")
    
    # ヘッダー確認
    header = data[:19]
    print(f"\nヘッダー: {header.hex()}")
    
    # 行ヘッダパターンを探す
    ncols = 17  # lineorderは17列
    positions = []
    
    print(f"\n行ヘッダ（{ncols}列）の検出:")
    
    # 最初の10個の行ヘッダを探す
    pos = 19  # ヘッダー後から開始
    found = 0
    search_limit = min(len(data) - 1, 10000)
    
    while pos < search_limit and found < 20:
        if pos + 1 < len(data):
            num_fields = (data[pos] << 8) | data[pos + 1]
            
            if num_fields == ncols:
                positions.append(pos)
                parity = "偶数" if pos % 2 == 0 else "奇数"
                print(f"  行 {found + 1}: 位置 {pos} ({parity})")
                found += 1
                
                # 次の行を予測（概算）
                pos += 200
            else:
                pos += 1
        else:
            break
    
    # 行間隔の分析
    if len(positions) > 1:
        intervals = []
        for i in range(1, len(positions)):
            intervals.append(positions[i] - positions[i-1])
        
        print(f"\n行間隔:")
        print(f"  平均: {np.mean(intervals):.1f} bytes")
        print(f"  最小: {min(intervals)} bytes")
        print(f"  最大: {max(intervals)} bytes")
        print(f"  間隔リスト（最初の10個）: {intervals[:10]}")
    
    # 偶数/奇数位置の統計
    even_positions = [p for p in positions if p % 2 == 0]
    odd_positions = [p for p in positions if p % 2 == 1]
    
    print(f"\n位置の偶奇統計:")
    print(f"  偶数位置: {len(even_positions)}個 ({len(even_positions)/len(positions)*100:.1f}%)")
    print(f"  奇数位置: {len(odd_positions)}個 ({len(odd_positions)/len(positions)*100:.1f}%)")
    
    # 行データの詳細確認（最初の3行）
    print(f"\n=== 最初の3行の詳細 ===")
    for i in range(min(3, len(positions))):
        row_pos = positions[i]
        print(f"\n行 {i + 1} (位置 {row_pos}):")
        
        # 行ヘッダ周辺のバイト表示
        start = max(0, row_pos - 10)
        end = min(len(data), row_pos + 20)
        
        print(f"  バイトデータ（位置 {start}-{end}）:")
        hex_data = data[start:end].hex()
        # 2バイトごとに区切って表示
        formatted_hex = ' '.join([hex_data[j:j+2] for j in range(0, len(hex_data), 2)])
        print(f"  {formatted_hex}")
        
        # 行ヘッダの位置をマーク
        marker_pos = row_pos - start
        marker = ' ' * (marker_pos * 3) + '^^'
        print(f"  {marker} (行ヘッダ: {ncols}列)")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    analyze_lineorder_binary()