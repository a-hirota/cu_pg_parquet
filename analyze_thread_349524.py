#!/usr/bin/env python3
"""
Thread 349524の問題を詳しく分析
"""

import cudf
from pathlib import Path
import struct

def analyze_thread_349524():
    """Thread 349524がなぜ1行しか処理しないのか分析"""
    
    print("Thread 349524の詳細分析:")
    print("="*80)
    
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        df = cudf.read_parquet(pf)
        thread_data = df[df['_thread_id'] == 349524]
        
        if len(thread_data) > 0:
            print(f"\n{pf.name}:")
            
            # pandasに変換して処理
            thread_pd = thread_data.to_pandas()
            
            for idx, row in thread_pd.iterrows():
                start_pos = int(row['_thread_start_pos'])
                end_pos = int(row['_thread_end_pos'])
                row_pos = int(row['_row_position'])
                custkey = int(row['c_custkey'])
                
                print(f"  検出行:")
                print(f"    c_custkey: {custkey}")
                print(f"    行位置: 0x{row_pos:08X}")
                print(f"    スレッド範囲: 0x{start_pos:08X} - 0x{end_pos:08X} ({end_pos - start_pos} bytes)")
                
                # 行の推定サイズ（PostgreSQL COPY BINARYフォーマット）
                # 行ヘッダ(2) + フィールド数8 * (長さ4 + データ) + 行末
                estimated_row_size = 2 + 8 * 4 + 8 + 7 * 15  # 概算
                print(f"    推定行サイズ: 約{estimated_row_size} bytes")
                
                # 残りスペース
                remaining = end_pos - row_pos
                print(f"    スレッド終了までの残り: {remaining} bytes")
                
                if remaining > estimated_row_size:
                    print(f"    → 理論的にはもう1行処理可能")
                else:
                    print(f"    → スペース不足でもう1行処理不可")
                
                # 次の行の推定位置
                next_row_pos = row_pos + estimated_row_size
                print(f"    次の行の推定位置: 0x{next_row_pos:08X}")
                
                if next_row_pos < end_pos:
                    print(f"    → 次の行はスレッド範囲内")
                else:
                    print(f"    → 次の行はスレッド範囲外")

def check_binary_data_pattern():
    """バイナリデータのパターンを確認"""
    
    print("\n\nバイナリデータパターンの確認:")
    print("="*80)
    
    # Thread 349524の終了位置付近のデータパターンを推測
    thread_end = 0x03FFFFD3  # Thread 349524の終了位置
    next_thread_start = 0x08000093  # Thread 699051の開始位置
    
    gap = next_thread_start - thread_end
    print(f"Thread 349524終了 (0x{thread_end:08X}) → Thread 699051開始 (0x{next_thread_start:08X})")
    print(f"ギャップ: 0x{gap:08X} ({gap:,} bytes = {gap/1024/1024:.1f} MB)")
    
    # このギャップは明らかに異常
    print("\n分析:")
    print("- 通常のthread_strideは192バイト")
    print("- しかし、実際のギャップは67MB以上")
    print("- これは明らかにデータの不連続性を示している")
    
    # 可能性のある原因
    print("\n可能性のある原因:")
    print("1. データが16個のチャンクに分割されている")
    print("2. Thread 349524はチャンク境界に位置している")
    print("3. 次のスレッド（699051）は別のチャンクのデータを処理している")

def analyze_chunk_boundaries():
    """チャンク境界を分析"""
    
    print("\n\nチャンク境界の分析:")
    print("="*80)
    
    # 各チャンクのデータ範囲を推定
    total_size = 97710505856  # 91GB
    num_chunks = 16
    chunk_size = total_size // num_chunks
    
    print(f"総データサイズ: {total_size:,} bytes")
    print(f"チャンク数: {num_chunks}")
    print(f"チャンクサイズ: {chunk_size:,} bytes ({chunk_size/1024/1024/1024:.1f} GB)")
    
    # Thread 349524の位置から、どのチャンクか推定
    thread_pos = 0x03FFFFD3
    chunk_id = thread_pos // chunk_size
    offset_in_chunk = thread_pos % chunk_size
    
    print(f"\nThread 349524の位置 (0x{thread_pos:08X}):")
    print(f"  推定チャンク: {chunk_id}")
    print(f"  チャンク内オフセット: 0x{offset_in_chunk:08X} ({offset_in_chunk/1024/1024:.1f} MB)")
    
    # Thread 699051の位置
    thread_pos2 = 0x08000093
    chunk_id2 = thread_pos2 // chunk_size
    offset_in_chunk2 = thread_pos2 % chunk_size
    
    print(f"\nThread 699051の位置 (0x{thread_pos2:08X}):")
    print(f"  推定チャンク: {chunk_id2}")
    print(f"  チャンク内オフセット: 0x{offset_in_chunk2:08X} ({offset_in_chunk2/1024/1024:.1f} MB)")

if __name__ == "__main__":
    analyze_thread_349524()
    check_binary_data_pattern()
    analyze_chunk_boundaries()