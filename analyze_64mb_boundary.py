#!/usr/bin/env python3
"""
64MB境界の詳細分析 - なぜ2行だけ欠落するのか
"""

import cudf
from pathlib import Path

def analyze_boundary_threads():
    """64MB境界付近のスレッドを詳細分析"""
    
    print("64MB境界付近のスレッド分析:")
    print("="*80)
    
    # 境界値
    boundary_64mb = 0x04000000
    
    # 境界付近のスレッドを特定
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        print(f"\n{pf.name}:")
        df = cudf.read_parquet(pf)
        
        # 境界前後1MBの範囲
        before_boundary = boundary_64mb - 0x100000  # 64MB - 1MB
        after_boundary = boundary_64mb + 0x100000   # 64MB + 1MB
        
        mask = ((df['_row_position'] >= before_boundary) & 
                (df['_row_position'] <= after_boundary))
        
        boundary_data = df[mask]
        
        if len(boundary_data) > 0:
            # スレッドごとにグループ化
            thread_groups = boundary_data.groupby('_thread_id').agg({
                '_row_position': ['min', 'max', 'count'],
                'c_custkey': ['min', 'max']
            }).to_pandas()
            
            print(f"  境界付近のスレッド数: {len(thread_groups)}")
            
            # 境界をまたぐスレッドを探す
            for thread_id, row in thread_groups.iterrows():
                min_pos = int(row[('_row_position', 'min')])
                max_pos = int(row[('_row_position', 'max')])
                count = int(row[('_row_position', 'count')])
                
                if min_pos < boundary_64mb and max_pos >= boundary_64mb:
                    print(f"\n  Thread {thread_id} が境界をまたぐ:")
                    print(f"    行数: {count}")
                    print(f"    位置範囲: 0x{min_pos:08X} - 0x{max_pos:08X}")
                    print(f"    c_custkey範囲: {int(row[('c_custkey', 'min')])} - {int(row[('c_custkey', 'max')])}")
                    
                    # このスレッドの詳細データ
                    thread_detail = boundary_data[boundary_data['_thread_id'] == thread_id].sort_values('_row_position')
                    thread_pd = thread_detail.to_pandas()
                    
                    # 境界前後の行を表示
                    before_64mb = thread_pd[thread_pd['_row_position'] < boundary_64mb]
                    after_64mb = thread_pd[thread_pd['_row_position'] >= boundary_64mb]
                    
                    if len(before_64mb) > 0:
                        last_before = before_64mb.iloc[-1]
                        print(f"\n    64MB境界前の最後の行:")
                        print(f"      c_custkey: {int(last_before['c_custkey'])}")
                        print(f"      Position: 0x{int(last_before['_row_position']):08X}")
                    
                    if len(after_64mb) > 0:
                        first_after = after_64mb.iloc[0]
                        print(f"\n    64MB境界後の最初の行:")
                        print(f"      c_custkey: {int(first_after['c_custkey'])}")
                        print(f"      Position: 0x{int(first_after['_row_position']):08X}")
                        
                        # ギャップを計算
                        if len(before_64mb) > 0:
                            gap = int(first_after['_row_position']) - int(last_before['_row_position'])
                            print(f"\n    境界でのギャップ: {gap} bytes")
                            print(f"    通常の行間隔: 約100-150 bytes")
                            
                            if gap > 200:
                                print(f"    → ギャップが異常に大きい！")

def analyze_thread_349525():
    """Thread 349525の177バイトスキップを分析"""
    
    print("\n\nThread 349525の異常なスキップ分析:")
    print("="*80)
    
    for pf in sorted(Path("output").glob("customer_chunk_*_queue.parquet")):
        df = cudf.read_parquet(pf)
        
        thread_data = df[df['_thread_id'] == 349525]
        
        if len(thread_data) > 0:
            print(f"\n{pf.name}:")
            
            thread_pd = thread_data.sort_values('_row_position').to_pandas()
            first_row = thread_pd.iloc[0]
            
            thread_start = int(first_row['_thread_start_pos'])
            first_row_pos = int(first_row['_row_position'])
            skip = first_row_pos - thread_start
            
            print(f"  Thread開始: 0x{thread_start:08X}")
            print(f"  最初の行: 0x{first_row_pos:08X}")
            print(f"  スキップ: {skip} bytes")
            
            print(f"\n  スキップの内容:")
            print(f"    0x{thread_start:08X} - 0x{first_row_pos:08X}")
            print(f"    = 0x03FFFFD3 - 0x04000084")
            
            # 特殊な値をチェック
            if thread_start == 0x03FFFFD3:
                print(f"    → Thread 349524の終了位置と一致")
            
            missing_range_start = thread_start
            missing_range_end = first_row_pos
            
            print(f"\n  この範囲に欠落行が存在する可能性:")
            print(f"    範囲: 0x{missing_range_start:08X} - 0x{missing_range_end:08X}")
            print(f"    サイズ: {missing_range_end - missing_range_start} bytes")
            print(f"    推定行数: {(missing_range_end - missing_range_start) // 100} 行（1行100バイトと仮定）")

def summarize_findings():
    """発見事項のまとめ"""
    
    print("\n\n発見事項のまとめ:")
    print("="*80)
    
    print("\n1. 64MB境界での異常:")
    print("   - Thread 349524は64MB境界直前（0x03FFFFD3）で停止")
    print("   - Thread 349525は177バイトスキップして開始")
    print("   - このスキップ範囲（0x03FFFFD3 - 0x04000084）に欠落行が存在")
    
    print("\n2. なぜ2行だけの欠落か:")
    print("   - Thread 349524が処理すべきだった2行目")
    print("   - Thread 699051の前にあるべき行")
    print("   - 両方とも64MB境界付近の特殊な位置にある")
    
    print("\n3. 考えられる原因:")
    print("   - PostgreSQL COPY BINARYが64MB境界で特殊な処理をしている")
    print("   - データ転送時に64MB境界でアライメントやパディングが発生")
    print("   - CUDAカーネルが境界付近の行を正しく検出できない")

if __name__ == "__main__":
    analyze_boundary_threads()
    analyze_thread_349525()
    summarize_findings()