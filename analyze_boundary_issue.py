#!/usr/bin/env python3
"""
スレッド境界での問題を分析
"""

import cudf
from pathlib import Path

def analyze_boundary_issue():
    """スレッド境界付近の行を分析"""
    parquet_files = sorted(Path("output").glob("chunk_*_queue.parquet"))
    
    boundary_issues = []
    
    for pf in parquet_files:
        df = cudf.read_parquet(pf)
        
        if '_thread_id' in df.columns and '_row_position' in df.columns:
            # スレッドの開始位置付近の行を探す
            for i in range(len(df)):
                try:
                    thread_id = df['_thread_id'].iloc[i]
                    row_pos = df['_row_position'].iloc[i]
                    thread_start = df['_thread_start_pos'].iloc[i]
                    thread_end = df['_thread_end_pos'].iloc[i]
                    
                    # スレッド開始位置から20バイト以内の行
                    distance_from_start = row_pos - thread_start
                    if 0 <= distance_from_start <= 20:
                        key = df['c_custkey'].iloc[i]
                        name = df['c_name'].iloc[i] if 'c_name' in df.columns else 'N/A'
                        
                        # 問題のある行（c_custkey=0やデータがずれている行）
                        if key == 0 or (isinstance(name, str) and not name.startswith('Customer#')):
                            boundary_issues.append({
                                'file': pf.name,
                                'row_idx': i,
                                'thread_id': thread_id,
                                'row_pos': row_pos,
                                'thread_start': thread_start,
                                'thread_end': thread_end,
                                'distance': distance_from_start,
                                'c_custkey': key,
                                'c_name': name
                            })
                except Exception as e:
                    pass
    
    print(f"=== スレッド境界での問題 ===")
    print(f"見つかった問題: {len(boundary_issues)}件\n")
    
    for issue in boundary_issues[:10]:  # 最初の10件
        print(f"ファイル: {issue['file']}")
        print(f"  thread_id: {issue['thread_id']}")
        print(f"  row_position: {issue['row_pos']:,}")
        print(f"  thread範囲: [{issue['thread_start']:,} - {issue['thread_end']:,}]")
        print(f"  開始位置からの距離: {issue['distance']}バイト")
        print(f"  c_custkey: {issue['c_custkey']}")
        print(f"  c_name: '{issue['c_name']}'")
        print()
    
    # 共通パターンを分析
    distances = [issue['distance'] for issue in boundary_issues]
    if distances:
        print(f"開始位置からの距離の分布:")
        for d in sorted(set(distances)):
            count = distances.count(d)
            print(f"  {d}バイト: {count}件")

if __name__ == "__main__":
    analyze_boundary_issue()