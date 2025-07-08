#!/usr/bin/env python3
"""
スレッドIDからブロックID、ワープIDを分析
"""

def analyze_thread_pattern(thread_id):
    """スレッドIDからブロック、ワープ情報を計算"""
    threads_per_block = 256
    threads_per_warp = 32
    
    block_id = thread_id // threads_per_block
    thread_in_block = thread_id % threads_per_block
    warp_in_block = thread_in_block // threads_per_warp
    thread_in_warp = thread_in_block % threads_per_warp
    
    return {
        'thread_id': thread_id,
        'block_id': block_id,
        'thread_in_block': thread_in_block,
        'warp_in_block': warp_in_block,
        'thread_in_warp': thread_in_warp
    }

def main():
    # 問題のあるスレッドIDペア
    thread_pairs = [
        (1048575, 3201426),  # 476029欠落
        (1398101, 3404032),  # 3483451欠落
        (2446677, 4214529),  # 4235332欠落
        (349524, 2796205),   # 4987296欠落
    ]
    
    print("スレッドIDパターン分析:")
    print("="*80)
    
    for before, after in thread_pairs:
        print(f"\nThread {before:,} → {after:,}")
        
        before_info = analyze_thread_pattern(before)
        after_info = analyze_thread_pattern(after)
        
        print(f"\n  Thread {before:,}:")
        print(f"    Block ID: {before_info['block_id']:,}")
        print(f"    Thread in block: {before_info['thread_in_block']}")
        print(f"    Warp in block: {before_info['warp_in_block']}")
        print(f"    Thread in warp: {before_info['thread_in_warp']}")
        print(f"    16進数: 0x{before:06X}")
        
        print(f"\n  Thread {after:,}:")
        print(f"    Block ID: {after_info['block_id']:,}")
        print(f"    Thread in block: {after_info['thread_in_block']}")
        print(f"    Warp in block: {after_info['warp_in_block']}")
        print(f"    Thread in warp: {after_info['thread_in_warp']}")
        print(f"    16進数: 0x{after:06X}")
        
        print(f"\n  ギャップ:")
        block_gap = after_info['block_id'] - before_info['block_id']
        thread_gap = after - before
        print(f"    ブロック差: {block_gap:,}")
        print(f"    スレッド差: {thread_gap:,}")
        
        # 特殊なパターンをチェック
        if before_info['thread_in_block'] == 255:
            print(f"    → Thread {before}はブロックの最後のスレッド")
        
        # ビットパターンをチェック
        if before & 0xFFFFF == 0xFFFFF:
            print(f"    → 20ビット境界 (0xFFFFF)")
        elif before & 0x1FFFFF == 0x1FFFFF:
            print(f"    → 21ビット境界 (0x1FFFFF)")
        
        # スキップされたブロックを分析
        skipped_blocks = list(range(before_info['block_id'] + 1, after_info['block_id']))
        if len(skipped_blocks) > 0:
            print(f"\n  スキップされたブロック: {len(skipped_blocks):,}個")
            print(f"    範囲: Block {skipped_blocks[0]:,} - {skipped_blocks[-1]:,}")
            
            # ブロックIDのパターンを確認
            for i in skipped_blocks[:5]:  # 最初の5個
                print(f"      Block {i:,} (0x{i:06X})")

if __name__ == "__main__":
    main()