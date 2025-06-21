"""
psycopg3チャンクサイズ分析スクリプト
psycopg3のCOPY処理でのチャンクサイズを分析し、ボトルネックを特定

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import time
import psycopg
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from datetime import datetime

TABLE_NAME = "lineorder"
OUTPUT_DIR = "benchmark/psycopg3_analysis"

def analyze_copy_chunks(limit_rows: int = 1000000, detailed_output: bool = True):
    """psycopg3のCOPYチャンクサイズを詳細分析"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print(f"=== psycopg3 COPYチャンクサイズ分析 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"行数制限: {limit_rows:,}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 接続とCOPY実行
    conn = psycopg.connect(dsn)
    chunk_sizes = []
    chunk_times = []
    total_bytes = 0
    chunk_count = 0
    
    try:
        copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        
        print("\nCOPY処理を開始...")
        start_time = time.time()
        
        with conn.cursor() as cur:
            with cur.copy(copy_sql) as copy_obj:
                while True:
                    chunk_start = time.time()
                    chunk = copy_obj.read()
                    
                    if not chunk:
                        break
                    
                    chunk_time = time.time() - chunk_start
                    chunk_size = len(chunk)
                    chunk_count += 1
                    total_bytes += chunk_size
                    
                    chunk_sizes.append(chunk_size)
                    chunk_times.append(chunk_time)
                    
                    # 進捗表示
                    if chunk_count % 100 == 0:
                        avg_size = np.mean(chunk_sizes) / 1024
                        print(f"  {chunk_count}チャンク処理済 "
                              f"(平均{avg_size:.1f}KB, 合計{total_bytes/(1024*1024):.1f}MB)")
        
        total_time = time.time() - start_time
        
    finally:
        conn.close()
    
    # 統計分析
    chunk_sizes_kb = [s / 1024 for s in chunk_sizes]
    
    print(f"\n=== 分析結果 ===")
    print(f"総チャンク数: {chunk_count}")
    print(f"総データサイズ: {total_bytes / (1024*1024):.2f} MB")
    print(f"総処理時間: {total_time:.2f} 秒")
    print(f"スループット: {(total_bytes / (1024*1024)) / total_time:.2f} MB/sec")
    
    print(f"\n--- チャンクサイズ統計 ---")
    print(f"平均: {np.mean(chunk_sizes_kb):.1f} KB")
    print(f"中央値: {np.median(chunk_sizes_kb):.1f} KB")
    print(f"最小: {min(chunk_sizes_kb):.1f} KB")
    print(f"最大: {max(chunk_sizes_kb):.1f} KB")
    print(f"標準偏差: {np.std(chunk_sizes_kb):.1f} KB")
    
    # 詳細分析
    if detailed_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ヒストグラム作成
        plt.figure(figsize=(10, 6))
        plt.hist(chunk_sizes_kb, bins=50, edgecolor='black')
        plt.xlabel('Chunk Size (KB)')
        plt.ylabel('Frequency')
        plt.title(f'psycopg3 COPY Chunk Size Distribution\n{TABLE_NAME} ({limit_rows:,} rows)')
        plt.grid(True, alpha=0.3)
        
        # 8KB位置に縦線を追加（psycopg3のデフォルトバッファサイズ）
        plt.axvline(x=8, color='red', linestyle='--', linewidth=2, label='8KB (default)')
        plt.legend()
        
        hist_path = os.path.join(OUTPUT_DIR, f"chunk_size_histogram_{timestamp}.png")
        plt.savefig(hist_path)
        print(f"\nヒストグラム保存: {hist_path}")
        plt.close()
        
        # 時系列グラフ
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(chunk_sizes_kb[:1000], alpha=0.7)  # 最初の1000チャンク
        plt.ylabel('Chunk Size (KB)')
        plt.title('Chunk Size Over Time (First 1000 chunks)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(np.array(chunk_times[:1000]) * 1000, alpha=0.7, color='orange')
        plt.ylabel('Read Time (ms)')
        plt.xlabel('Chunk Number')
        plt.grid(True, alpha=0.3)
        
        timeline_path = os.path.join(OUTPUT_DIR, f"chunk_timeline_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(timeline_path)
        print(f"時系列グラフ保存: {timeline_path}")
        plt.close()
        
        # JSON詳細データ
        analysis_data = {
            'metadata': {
                'table': TABLE_NAME,
                'limit_rows': limit_rows,
                'timestamp': timestamp,
                'total_chunks': chunk_count,
                'total_bytes': total_bytes,
                'total_time': total_time
            },
            'statistics': {
                'mean_kb': float(np.mean(chunk_sizes_kb)),
                'median_kb': float(np.median(chunk_sizes_kb)),
                'min_kb': float(min(chunk_sizes_kb)),
                'max_kb': float(max(chunk_sizes_kb)),
                'std_kb': float(np.std(chunk_sizes_kb)),
                'throughput_mb_sec': (total_bytes / (1024*1024)) / total_time
            },
            'chunk_details': [
                {
                    'chunk_num': i,
                    'size_bytes': chunk_sizes[i],
                    'time_ms': chunk_times[i] * 1000
                }
                for i in range(min(1000, len(chunk_sizes)))  # 最初の1000チャンクのみ
            ]
        }
        
        json_path = os.path.join(OUTPUT_DIR, f"chunk_analysis_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"詳細データ保存: {json_path}")
    
    # 推奨事項
    print(f"\n=== 分析に基づく推奨事項 ===")
    avg_chunk_kb = np.mean(chunk_sizes_kb)
    
    if avg_chunk_kb < 16:
        print("⚠️  チャンクサイズが小さすぎます（< 16KB）")
        print("   → psycopg3のバッファサイズ設定を確認してください")
        print("   → libpq直接実装やCython実装を検討してください")
    elif avg_chunk_kb < 64:
        print("⚠️  チャンクサイズが最適ではありません（< 64KB）")
        print("   → より大きなバッファサイズでの読み込みが推奨されます")
    else:
        print("✅ チャンクサイズは適切です")
    
    # 実際のpsycopg3実装との比較
    print(f"\n--- psycopg3実装の制限 ---")
    print("• psycopg3は内部的に8KBバッファを使用")
    print("• copy.read()は利用可能なデータを返す（最大8KB）")
    print("• この制限により、大量データ転送時にボトルネックとなる")
    
    return {
        'chunk_count': chunk_count,
        'total_bytes': total_bytes,
        'total_time': total_time,
        'avg_chunk_kb': avg_chunk_kb,
        'throughput_mb_sec': (total_bytes / (1024*1024)) / total_time
    }


def compare_chunk_sizes():
    """異なる方法でのチャンクサイズを比較"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return
    
    print("=== チャンクサイズ比較実験 ===")
    
    test_rows = 100000
    results = {}
    
    # 1. デフォルトpsycopg3
    print("\n1. psycopg3 デフォルト実装:")
    results['psycopg3_default'] = analyze_copy_chunks(test_rows, detailed_output=False)
    
    # 2. 異なる読み込みパターンのテスト
    print("\n2. read()パラメータテスト:")
    conn = psycopg.connect(dsn)
    
    for read_size in [1024, 8192, 65536, 1048576]:  # 1KB, 8KB, 64KB, 1MB
        print(f"\n  read({read_size//1024}KB):")
        chunk_sizes = []
        start_time = time.time()
        
        try:
            copy_sql = f"COPY (SELECT * FROM {TABLE_NAME} LIMIT {test_rows}) TO STDOUT (FORMAT binary)"
            
            with conn.cursor() as cur:
                with cur.copy(copy_sql) as copy_obj:
                    while True:
                        chunk = copy_obj.read(read_size)
                        if not chunk:
                            break
                        chunk_sizes.append(len(chunk))
            
            elapsed = time.time() - start_time
            avg_size = np.mean(chunk_sizes) / 1024
            total_mb = sum(chunk_sizes) / (1024*1024)
            
            print(f"    平均チャンク: {avg_size:.1f} KB")
            print(f"    チャンク数: {len(chunk_sizes)}")
            print(f"    スループット: {total_mb/elapsed:.2f} MB/sec")
            
            results[f'read_{read_size//1024}kb'] = {
                'avg_chunk_kb': avg_size,
                'chunk_count': len(chunk_sizes),
                'throughput_mb_sec': total_mb/elapsed
            }
            
        except Exception as e:
            print(f"    エラー: {e}")
    
    conn.close()
    
    # 結果サマリー
    print("\n=== 比較結果サマリー ===")
    print(f"{'Method':<20} {'Avg Chunk':<12} {'Chunks':<10} {'Throughput':<12}")
    print("-" * 60)
    
    for method, data in results.items():
        if isinstance(data, dict) and 'avg_chunk_kb' in data:
            print(f"{method:<20} {data['avg_chunk_kb']:>8.1f} KB  "
                  f"{data['chunk_count']:>8}  {data['throughput_mb_sec']:>8.2f} MB/s")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='psycopg3 COPYチャンクサイズ分析')
    parser.add_argument('--rows', type=int, default=1000000, help='分析する行数')
    parser.add_argument('--compare', action='store_true', help='異なる読み込み方法を比較')
    parser.add_argument('--no-detail', action='store_true', help='詳細出力を無効化')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_chunk_sizes()
    else:
        analyze_copy_chunks(args.rows, detailed_output=not args.no_detail)


if __name__ == "__main__":
    main()