#!/usr/bin/env python3
"""
スレッド境界の正確な分析
"""

import cudf
from pathlib import Path

def analyze_exact_boundaries():
    """各欠落キー周辺のスレッド境界を正確に分析"""
    
    # 欠落キーのリスト
    missing_keys = [476029, 1227856, 1979731, 2731633, 3483451, 4235332, 
                   4987296, 6491094, 7243028, 7994887, 9498603, 10250384, 
                   11002286, 11754161]
    
    # 各Parquetファイルを調査
    parquet_files = sorted(Path("output").glob("customer_chunk_*_queue.parquet"))
    
    for missing_key in missing_keys[:5]:  # 最初の5つを詳しく分析
        print(f"\n{'='*80}")
        print(f"欠落キー: {missing_key}")
        print(f"{'='*80}")
        
        for pf in parquet_files:
            df = cudf.read_parquet(pf)
            
            # c_custkeyを整数に変換
            if hasattr(df['c_custkey'].dtype, 'precision'):
                df['c_custkey_int'] = df['c_custkey'].astype('int64')
            else:
                df['c_custkey_int'] = df['c_custkey']
            
            # 欠落キーの前後10行を取得
            mask_near = ((df['c_custkey_int'] >= missing_key - 10) & 
                        (df['c_custkey_int'] <= missing_key + 10))
            
            if mask_near.sum() > 0:
                near_df = df[mask_near].sort_values('c_custkey_int')
                print(f"\n{pf.name} での周辺データ:")
                
                # pandasに変換して表示
                near_pd = near_df[['c_custkey_int', '_thread_id', '_row_position', 
                                 '_thread_start_pos', '_thread_end_pos']].to_pandas()
                
                for idx, row in near_pd.iterrows():
                    custkey = int(row['c_custkey_int'])
                    thread_id = int(row['_thread_id'])
                    row_pos = int(row['_row_position'])
                    thread_start = int(row['_thread_start_pos'])
                    thread_end = int(row['_thread_end_pos'])
                    
                    marker = " ← 欠落" if custkey == missing_key else ""
                    if custkey == missing_key - 1:
                        marker = " ← 直前"
                    elif custkey == missing_key + 1:
                        marker = " ← 直後"
                    
                    print(f"  {custkey:8d}: Thread {thread_id:7d}, "
                          f"Pos 0x{row_pos:08X}, "
                          f"Thread範囲 0x{thread_start:08X}-0x{thread_end:08X}{marker}")
                
                # 欠落キーが存在しない場合の詳細
                if missing_key not in near_pd['c_custkey_int'].values:
                    # 直前と直後の行を特定
                    before_rows = near_pd[near_pd['c_custkey_int'] < missing_key]
                    after_rows = near_pd[near_pd['c_custkey_int'] > missing_key]
                    
                    if len(before_rows) > 0 and len(after_rows) > 0:
                        last_before = before_rows.iloc[-1]
                        first_after = after_rows.iloc[0]
                        
                        thread_before = int(last_before['_thread_id'])
                        thread_after = int(first_after['_thread_id'])
                        
                        pos_before = int(last_before['_row_position'])
                        pos_after = int(first_after['_row_position'])
                        
                        thread_end_before = int(last_before['_thread_end_pos'])
                        thread_start_after = int(first_after['_thread_start_pos'])
                        
                        print(f"\n  ギャップ分析:")
                        print(f"    前の行の位置: 0x{pos_before:08X}")
                        print(f"    前スレッドの終了: 0x{thread_end_before:08X}")
                        print(f"    前の行から終了までの距離: {thread_end_before - pos_before} bytes")
                        
                        print(f"    後の行の位置: 0x{pos_after:08X}")
                        print(f"    後スレッドの開始: 0x{thread_start_after:08X}")
                        print(f"    開始から後の行までの距離: {pos_after - thread_start_after} bytes")
                        
                        print(f"    スレッド間ギャップ: {thread_start_after - thread_end_before} bytes")
                        
                        # ギャップが妥当かチェック
                        gap = thread_start_after - thread_end_before
                        if gap < 0:
                            print(f"    → スレッドがオーバーラップしています！")
                        elif gap == 0:
                            print(f"    → スレッドが連続しています")
                        elif gap < 200:
                            print(f"    → 小さなギャップ（通常の行サイズ内）")
                        else:
                            print(f"    → 大きなギャップ！データが失われている可能性")

if __name__ == "__main__":
    analyze_exact_boundaries()