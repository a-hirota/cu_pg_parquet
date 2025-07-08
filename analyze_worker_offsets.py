#!/usr/bin/env python3
"""
ワーカーの実際の書き込みオフセットを分析
"""

def analyze_worker_behavior():
    """16並列ワーカーの実際の動作を分析"""
    
    print("16並列ワーカーの書き込み動作分析:")
    print("="*80)
    
    # 実際のデータ
    total_size = 848_104_872  # 約848MB
    num_workers = 16
    buffer_size = 64 * 1024 * 1024  # 64MB
    
    # PostgreSQLのページベースで分割
    # 各ワーカーは異なるページ範囲を処理
    
    print(f"\n総データサイズ: {total_size:,} bytes ({total_size/1024/1024:.1f}MB)")
    print(f"ワーカー数: {num_workers}")
    print(f"バッファサイズ: {buffer_size:,} bytes ({buffer_size/1024/1024}MB)")
    
    # 重要な発見：
    # worker_offset = Arc::new(AtomicU64::new(0)); は共有されている！
    # つまり、各ワーカーは順番にオフセットを取得して書き込む
    
    print("\n重要な発見:")
    print("- worker_offsetはすべてのワーカー間で共有されている（AtomicU64）")
    print("- 各ワーカーはfetch_addで現在のオフセットを取得し、自分のバッファサイズ分を予約")
    print("- つまり、ワーカーの書き込み順序は非決定的")
    
    print("\n考えられるシナリオ:")
    print("1. ワーカーAが0-53MBを処理、バッファに蓄積")
    print("2. ワーカーBが0-53MBを処理、バッファに蓄積")
    print("3. ワーカーAがフラッシュ → オフセット0-53MBに書き込み")
    print("4. ワーカーBがフラッシュ → オフセット53-106MBに書き込み")
    print("5. この時、53MB付近で行が分断される可能性")
    
    # 64MB境界に到達するシナリオ
    print("\n64MB境界に到達するシナリオ:")
    cumulative_offset = 0
    for i in range(num_workers):
        worker_data = total_size // num_workers  # 約53MB
        next_offset = cumulative_offset + worker_data
        
        print(f"\nワーカー{i:2d}:")
        print(f"  書き込み範囲: {cumulative_offset:,} - {next_offset:,} ({worker_data/1024/1024:.1f}MB)")
        print(f"  開始: 0x{cumulative_offset:08X}")
        print(f"  終了: 0x{next_offset:08X}")
        
        # 64MB境界をまたぐかチェック
        if cumulative_offset < 0x04000000 <= next_offset:
            print(f"  → 64MB境界（0x04000000）をまたぐ！")
            
        if cumulative_offset < 0x08000000 <= next_offset:
            print(f"  → 128MB境界（0x08000000）をまたぐ！")
        
        cumulative_offset = next_offset
        
        if cumulative_offset > 0x08000000:
            break

def explain_actual_behavior():
    """実際の動作の説明"""
    
    print("\n\n実際の動作メカニズム:")
    print("="*80)
    
    print("""
1. 各ワーカーは独立してPostgreSQLからデータを読み込む
2. 各ワーカーは自分のローカルバッファ（64MB）に蓄積
3. バッファが満杯になるか、データ終了時にフラッシュ
4. フラッシュ時：
   - worker_offset.fetch_add(buffer_size) で書き込み位置を取得
   - 他のワーカーと競合しないように原子的に位置を確保
   - ファイルの確保した位置に書き込み

問題の原因：
- 一部のワーカーが64MBちょうどのデータを書き込む
- 次のワーカーが64MB境界から書き込みを開始
- 境界をまたぐ行が正しく処理されない
""")

if __name__ == "__main__":
    analyze_worker_behavior()
    explain_actual_behavior()