#!/usr/bin/env python3
"""
実際の問題を正確に分析
"""

import cudf
from pathlib import Path

def analyze_missing_keys():
    """欠落している2つのキーの正確な位置を分析"""
    
    print("欠落キーの正確な分析:")
    print("="*80)
    
    missing_keys = [3483509, 6491094]
    
    for missing_key in missing_keys:
        print(f"\n欠落キー {missing_key}:")
        
        # PostgreSQLでの理論的な位置を推定
        # customerテーブルは1から12,030,000まで連続
        theoretical_position = missing_key - 1  # 0-indexed
        rows_per_chunk = 12030000 // 2  # 2チャンクに分割
        theoretical_chunk = 0 if theoretical_position < rows_per_chunk else 1
        
        print(f"  理論的な位置: {theoretical_position}番目の行")
        print(f"  理論的なチャンク: {theoretical_chunk}")
        
        # 実際のデータでの前後を確認
        for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
            df = cudf.read_parquet(pf)
            
            if hasattr(df['c_custkey'].dtype, 'precision'):
                df['c_custkey_int'] = df['c_custkey'].astype('int64')
            else:
                df['c_custkey_int'] = df['c_custkey']
            
            # 前後のキーを探す
            mask_before = df['c_custkey_int'] == missing_key - 1
            mask_after = df['c_custkey_int'] == missing_key + 1
            
            if mask_before.sum() > 0 and mask_after.sum() > 0:
                print(f"\n  {pf.name}で前後のキーを発見:")
                
                # 前のキー
                before_row = df[mask_before].iloc[0].to_pandas()
                print(f"    前のキー ({missing_key-1}):")
                print(f"      Thread: {int(before_row['_thread_id'].iloc[0])}")
                print(f"      Position: 0x{int(before_row['_row_position'].iloc[0]):08X}")
                print(f"      Thread end: 0x{int(before_row['_thread_end_pos'].iloc[0]):08X}")
                
                # 後のキー
                after_row = df[mask_after].iloc[0].to_pandas()
                print(f"    後のキー ({missing_key+1}):")
                print(f"      Thread: {int(after_row['_thread_id'].iloc[0])}")
                print(f"      Position: 0x{int(after_row['_row_position'].iloc[0]):08X}")
                print(f"      Thread start: 0x{int(after_row['_thread_start_pos'].iloc[0]):08X}")
                
                # ギャップを計算
                before_pos = int(before_row['_row_position'].iloc[0])
                after_pos = int(after_row['_row_position'].iloc[0])
                gap = after_pos - before_pos
                
                print(f"\n    位置のギャップ: 0x{gap:X} ({gap} bytes)")
                print(f"    通常の行サイズ: 約96-150 bytes")
                
                if gap > 200:
                    print(f"    → ギャップが大きすぎる！")

def analyze_thread_349524_detail():
    """Thread 349524の詳細な動作を分析"""
    
    print("\n\nThread 349524の詳細分析:")
    print("="*80)
    
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        df = cudf.read_parquet(pf)
        
        # Thread 349524のデータ
        thread_data = df[df['_thread_id'] == 349524]
        
        if len(thread_data) > 0:
            print(f"\n{pf.name}:")
            
            # Thread 349525（次のスレッド）のデータも取得
            next_thread_data = df[df['_thread_id'] == 349525]
            
            # pandas変換
            thread_pd = thread_data.to_pandas()
            next_pd = next_thread_data.to_pandas() if len(next_thread_data) > 0 else None
            
            # Thread 349524の情報
            for idx, row in thread_pd.iterrows():
                thread_start = int(row['_thread_start_pos'])
                thread_end = int(row['_thread_end_pos'])
                row_pos = int(row['_row_position'])
                custkey = int(row['c_custkey'])
                
                print(f"  Thread 349524:")
                print(f"    c_custkey: {custkey}")
                print(f"    スレッド範囲: 0x{thread_start:08X} - 0x{thread_end:08X}")
                print(f"    行位置: 0x{row_pos:08X}")
                print(f"    行終了推定: 0x{row_pos + 96:08X}")
                
                # 次の行が収まるか確認
                next_row_start = row_pos + 96  # 最小行サイズ
                if next_row_start < thread_end:
                    print(f"    → 次の行開始位置 0x{next_row_start:08X} < スレッド終了 0x{thread_end:08X}")
                    print(f"    → 理論的には次の行を処理可能")
                    print(f"    → しかし実際には処理していない！")
            
            # Thread 349525の情報
            if next_pd is not None and len(next_pd) > 0:
                print(f"\n  Thread 349525（次のスレッド）:")
                first_row = next_pd.iloc[0]
                next_start = int(first_row['_thread_start_pos'])
                next_pos = int(first_row['_row_position'])
                next_custkey = int(first_row['c_custkey'])
                
                print(f"    c_custkey: {next_custkey}")
                print(f"    スレッド開始: 0x{next_start:08X}")
                print(f"    最初の行位置: 0x{next_pos:08X}")
                print(f"    オフセット: {next_pos - next_start} bytes")

def check_binary_format():
    """バイナリフォーマットの問題を確認"""
    
    print("\n\nバイナリフォーマットの分析:")
    print("="*80)
    
    # Thread 349524の終了位置付近の状況
    thread_end = 0x03FFFFD3
    
    print(f"Thread 349524の終了位置: 0x{thread_end:08X}")
    print(f"  = {thread_end:,} bytes")
    print(f"  = {thread_end / 1024 / 1024:.6f} MB")
    print(f"  = ほぼ64MB (0x04000000 = {0x04000000 / 1024 / 1024} MB)")
    
    print("\n仮説:")
    print("1. バイナリデータに64MBごとの特殊なマーカーやパディングがある")
    print("2. Thread 349524は64MB境界直前で停止するように設計されている")
    print("3. 欠落行は境界をまたぐ位置にある")

if __name__ == "__main__":
    analyze_missing_keys()
    analyze_thread_349524_detail()
    check_binary_format()