#!/usr/bin/env python3
"""
圧縮実験結果の可視化スクリプト

compression_experiment.pyで生成されたJSONファイルを読み込み、
グラフとして可視化します。
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(json_file: str) -> list:
    """JSONファイルから結果を読み込み"""
    with open(json_file, 'r') as f:
        return json.load(f)


def plot_results(results: list, output_file: str = None):
    """結果をグラフ化"""
    # 成功した結果のみをフィルタ
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    # データの準備
    compressions = [r['compression'] for r in successful_results]
    times = [r.get('total_time', r['elapsed_time']) for r in successful_results]
    sizes = [r.get('parquet_total_size_mb', 0) for r in successful_results]
    ratios = [r.get('compression_ratio', 0) for r in successful_results]
    throughputs = [r.get('throughput_gb_s', 0) for r in successful_results]
    
    # 図の作成（2x2のサブプロット）
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Parquet Compression Performance Comparison', fontsize=16)
    
    # カラーマップ
    colors = plt.cm.viridis(np.linspace(0, 1, len(compressions)))
    
    # 1. 処理時間
    ax1.bar(compressions, times, color=colors)
    ax1.set_title('Processing Time by Compression')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Compression Method')
    for i, v in enumerate(times):
        ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    # 2. ファイルサイズ
    ax2.bar(compressions, sizes, color=colors)
    ax2.set_title('Output File Size by Compression')
    ax2.set_ylabel('Size (MB)')
    ax2.set_xlabel('Compression Method')
    for i, v in enumerate(sizes):
        ax2.text(i, v + 10, f'{v:.0f}', ha='center', va='bottom')
    
    # 3. 圧縮率
    ax3.bar(compressions, ratios, color=colors)
    ax3.set_title('Compression Ratio by Method')
    ax3.set_ylabel('Compression Ratio (x)')
    ax3.set_xlabel('Compression Method')
    for i, v in enumerate(ratios):
        ax3.text(i, v + 0.1, f'{v:.2f}x', ha='center', va='bottom')
    
    # 4. スループット
    ax4.bar(compressions, throughputs, color=colors)
    ax4.set_title('Throughput by Compression')
    ax4.set_ylabel('Throughput (GB/s)')
    ax4.set_xlabel('Compression Method')
    for i, v in enumerate(throughputs):
        ax4.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # レイアウトの調整
    plt.tight_layout()
    
    # 保存または表示
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()


def create_comparison_table(results: list):
    """比較表を作成してコンソールに出力"""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("No successful results to compare")
        return
    
    # Snappyを基準とした相対比較
    snappy_result = next((r for r in successful_results if r['compression'] == 'snappy'), None)
    
    if snappy_result:
        snappy_time = snappy_result.get('total_time', snappy_result['elapsed_time'])
        snappy_size = snappy_result.get('parquet_total_size_mb', 0)
        
        print("\n" + "="*80)
        print("RELATIVE PERFORMANCE COMPARISON (vs Snappy)")
        print("="*80)
        print(f"{'Compression':<12} {'Time Diff':<12} {'Size Diff':<12} {'Time Ratio':<12} {'Size Ratio'}")
        print("-"*80)
        
        for r in successful_results:
            comp = r['compression']
            time = r.get('total_time', r['elapsed_time'])
            size = r.get('parquet_total_size_mb', 0)
            
            time_diff = time - snappy_time
            size_diff = size - snappy_size
            time_ratio = time / snappy_time if snappy_time > 0 else 0
            size_ratio = size / snappy_size if snappy_size > 0 else 0
            
            time_diff_str = f"{time_diff:+.2f}s"
            size_diff_str = f"{size_diff:+.0f}MB"
            time_ratio_str = f"{time_ratio:.2f}x"
            size_ratio_str = f"{size_ratio:.2f}x"
            
            print(f"{comp:<12} {time_diff_str:<12} {size_diff_str:<12} {time_ratio_str:<12} {size_ratio_str}")
        
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(
        description="圧縮実験結果の可視化"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="入力JSONファイル（compression_experiment.pyの出力）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="グラフの保存先（指定しない場合は表示のみ）"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="グラフを作成せず、テーブルのみ表示"
    )
    
    args = parser.parse_args()
    
    # 結果の読み込み
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        return 1
    
    results = load_results(args.json_file)
    
    # 比較表の作成
    create_comparison_table(results)
    
    # グラフの作成
    if not args.no_plot:
        plot_results(results, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())