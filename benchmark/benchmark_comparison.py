#!/usr/bin/env python3
"""
3パターンのベンチマーク比較実行スクリプト

実行パターン:
1. 直接GPU転送版 (direct) - memoryviewから直接GPU転送
2. バッチ化GPU転送版 (batch) - チャンクをバッチ化してGPU転送
3. 現在の実装 (current) - BytesIO + bytearray経由

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

# ベンチマークスクリプトのパス
BENCHMARK_SCRIPTS = {
    'direct': 'benchmark/benchmark_parallel_ctid_ray_sequential_gpu_direct.py',
    'batch': 'benchmark/benchmark_parallel_ctid_ray_sequential_gpu_batch.py',
    'current': 'benchmark/benchmark_parallel_ctid_ray_sequential_gpu_current.py'
}

# メトリクスファイルのパス
METRICS_FILES = {
    'direct': 'benchmark/lineorder_parallel_ctid_ray_sequential_gpu_direct.output_metrics_direct.json',
    'batch': 'benchmark/lineorder_parallel_ctid_ray_sequential_gpu_batch.output_metrics_batch.json',
    'current': 'benchmark/lineorder_parallel_ctid_ray_sequential_gpu_current.output_metrics_current.json'
}

def ensure_conda_env():
    """conda環境が正しくアクティベートされているか確認"""
    try:
        import cudf
        print("✅ cuDF環境確認OK")
    except ImportError:
        print("❌ エラー: cuDF環境が見つかりません")
        print("以下のコマンドを実行してください:")
        print("source /home/ubuntu/miniconda3/etc/profile.d/conda.sh")
        print("conda activate cudf_dev")
        sys.exit(1)

def run_benchmark(pattern: str, args: List[str]) -> Dict:
    """ベンチマークを実行してメトリクスを取得"""
    script_path = BENCHMARK_SCRIPTS[pattern]
    metrics_path = METRICS_FILES[pattern]
    
    print(f"\n{'='*60}")
    print(f"実行中: {pattern} パターン")
    print(f"スクリプト: {script_path}")
    print(f"引数: {' '.join(args)}")
    print(f"{'='*60}")
    
    # 既存のメトリクスファイルを削除
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    
    # ベンチマーク実行
    start_time = time.time()
    cmd = ['python', script_path] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        execution_time = time.time() - start_time
        
        # 標準出力を保存
        output_file = f"benchmark/comparison_{pattern}_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"\n{'-'*40} STDOUT {'-'*40}\n")
            f.write(result.stdout)
            f.write(f"\n{'-'*40} STDERR {'-'*40}\n")
            f.write(result.stderr)
        
        print(f"✅ 実行完了 ({execution_time:.2f}秒)")
        print(f"出力保存: {output_file}")
        
        # メトリクスファイルを読み込み
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                metrics['execution_time'] = execution_time
                return metrics
        else:
            print(f"⚠️  メトリクスファイルが見つかりません: {metrics_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 実行エラー: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return None

def compare_metrics(metrics_data: Dict[str, Dict]):
    """メトリクスを比較して表形式で表示"""
    
    # メモリ使用量比較
    print("\n=== メモリ使用量比較 ===")
    memory_data = []
    
    for pattern, metrics in metrics_data.items():
        if metrics and 'memory' in metrics:
            memory = metrics['memory']
            memory_data.append({
                'パターン': pattern,
                '初期 (MB)': f"{memory.get('initial_system_mb', 0):.0f}",
                'ピーク (MB)': f"{memory.get('peak_system_mb', 0):.0f}",
                'オーバーヘッド率': f"{memory.get('memory_overhead_ratio', 0):.2f}x",
                'チャンク数': f"{memory.get('chunk_count', 0):,}",
                '平均チャンクサイズ': f"{memory.get('avg_chunk_size', 0):.0f}B"
            })
    
    if memory_data:
        df_memory = pd.DataFrame(memory_data)
        print(df_memory.to_string(index=False))
    
    # パフォーマンス比較
    print("\n=== パフォーマンス比較 ===")
    perf_data = []
    
    for pattern, metrics in metrics_data.items():
        if metrics and 'performance' in metrics:
            perf = metrics['performance']
            perf_data.append({
                'パターン': pattern,
                'COPY時間 (秒)': f"{perf.get('copy_time_sec', 0):.2f}",
                'GPU転送時間 (秒)': f"{perf.get('gpu_transfer_time_sec', 0):.2f}",
                'GPU転送回数': f"{perf.get('gpu_transfer_count', 0):,}",
                'GPU処理時間 (秒)': f"{perf.get('gpu_processing_time_sec', 0):.2f}",
                '総時間 (秒)': f"{perf.get('total_time_sec', 0):.2f}",
                'スループット (MB/s)': f"{perf.get('throughput_mb_sec', 0):.2f}"
            })
    
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        print(df_perf.to_string(index=False))
    
    # バッチ処理特有のメトリクス（バッチ版のみ）
    if 'batch' in metrics_data and metrics_data['batch']:
        batch_memory = metrics_data['batch'].get('memory', {})
        if 'batch_count' in batch_memory:
            print("\n=== バッチ処理メトリクス ===")
            print(f"総バッチ数: {batch_memory.get('batch_count', 0):,}")
            print(f"平均バッチサイズ: {batch_memory.get('avg_batch_size_mb', 0):.1f} MB")

def create_comparison_charts(metrics_data: Dict[str, Dict], output_dir: str):
    """比較チャートを生成"""
    
    # データ準備
    patterns = []
    memory_overhead = []
    throughput = []
    gpu_transfer_count = []
    
    for pattern, metrics in metrics_data.items():
        if metrics:
            patterns.append(pattern)
            memory_overhead.append(metrics.get('memory', {}).get('memory_overhead_ratio', 0))
            throughput.append(metrics.get('performance', {}).get('throughput_mb_sec', 0))
            gpu_transfer_count.append(metrics.get('performance', {}).get('gpu_transfer_count', 0))
    
    if not patterns:
        print("⚠️  チャート生成に必要なデータがありません")
        return
    
    # 図のセットアップ
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. メモリオーバーヘッド比較
    ax1 = axes[0]
    bars1 = ax1.bar(patterns, memory_overhead, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('メモリオーバーヘッド率')
    ax1.set_title('メモリ効率比較')
    ax1.set_ylim(0, max(memory_overhead) * 1.2 if memory_overhead else 2)
    
    # 値をバーの上に表示
    for bar, value in zip(bars1, memory_overhead):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.2f}x', ha='center', va='bottom')
    
    # 2. スループット比較
    ax2 = axes[1]
    bars2 = ax2.bar(patterns, throughput, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_ylabel('スループット (MB/sec)')
    ax2.set_title('処理速度比較')
    ax2.set_ylim(0, max(throughput) * 1.2 if throughput else 100)
    
    for bar, value in zip(bars2, throughput):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 3. GPU転送回数比較
    ax3 = axes[2]
    bars3 = ax3.bar(patterns, gpu_transfer_count, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_ylabel('GPU転送回数')
    ax3.set_title('GPU転送効率比較')
    ax3.set_yscale('log')  # 対数スケール
    
    for bar, value in zip(bars3, gpu_transfer_count):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{value:,}', ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    
    # チャート保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = os.path.join(output_dir, f'comparison_chart_{timestamp}.png')
    plt.savefig(chart_path, dpi=150)
    print(f"\n✅ 比較チャート保存: {chart_path}")
    plt.close()

def generate_report(metrics_data: Dict[str, Dict], output_dir: str):
    """マークダウン形式のレポートを生成"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_path = os.path.join(output_dir, f'comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md')
    
    with open(report_path, 'w') as f:
        f.write(f"# GPUPGParser ベンチマーク比較レポート\n\n")
        f.write(f"実行日時: {timestamp}\n\n")
        
        # 実行条件
        f.write("## 実行条件\n\n")
        f.write("- 並列数: 16\n")
        f.write("- チャンク数: 4\n")
        f.write("- テーブル: lineorder（全件処理）\n\n")
        
        # 結果サマリー
        f.write("## 結果サマリー\n\n")
        
        # 最適なパターンを判定
        best_memory = min(metrics_data.items(), 
                         key=lambda x: x[1].get('memory', {}).get('memory_overhead_ratio', float('inf')))[0]
        best_speed = max(metrics_data.items(), 
                        key=lambda x: x[1].get('performance', {}).get('throughput_mb_sec', 0))[0]
        
        f.write(f"- **最もメモリ効率が良い**: {best_memory}\n")
        f.write(f"- **最も処理速度が速い**: {best_speed}\n\n")
        
        # 詳細結果
        f.write("## 詳細結果\n\n")
        
        for pattern, metrics in metrics_data.items():
            if metrics:
                f.write(f"### {pattern}パターン\n\n")
                
                memory = metrics.get('memory', {})
                perf = metrics.get('performance', {})
                
                f.write("#### メモリ使用量\n")
                f.write(f"- 初期: {memory.get('initial_system_mb', 0):.0f} MB\n")
                f.write(f"- ピーク: {memory.get('peak_system_mb', 0):.0f} MB\n")
                f.write(f"- オーバーヘッド率: {memory.get('memory_overhead_ratio', 0):.2f}x\n")
                f.write(f"- チャンク数: {memory.get('chunk_count', 0):,}\n")
                f.write(f"- 平均チャンクサイズ: {memory.get('avg_chunk_size', 0):.0f} bytes\n\n")
                
                f.write("#### パフォーマンス\n")
                f.write(f"- 総時間: {perf.get('total_time_sec', 0):.2f} 秒\n")
                f.write(f"- スループット: {perf.get('throughput_mb_sec', 0):.2f} MB/sec\n")
                f.write(f"- GPU転送回数: {perf.get('gpu_transfer_count', 0):,}\n\n")
        
        # 推奨事項
        f.write("## 推奨事項\n\n")
        
        if best_memory == best_speed:
            f.write(f"**{best_memory}パターン**が最もバランスが良く、推奨されます。\n")
        else:
            f.write(f"- メモリが制約となる場合: **{best_memory}パターン**\n")
            f.write(f"- 速度を優先する場合: **{best_speed}パターン**\n")
    
    print(f"✅ レポート生成: {report_path}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='3パターンのベンチマーク比較実行')
    parser.add_argument('--patterns', nargs='+', choices=['direct', 'batch', 'current'],
                       default=['direct', 'batch', 'current'],
                       help='実行するパターン（デフォルト: 全パターン）')
    parser.add_argument('--skip-run', action='store_true',
                       help='ベンチマーク実行をスキップし、既存のメトリクスを使用')
    parser.add_argument('--output-dir', default='benchmark/comparison_results',
                       help='結果出力ディレクトリ')
    
    # ベンチマーク共通引数
    parser.add_argument('--parallel', type=int, default=16, help='並列数')
    parser.add_argument('--chunks', type=int, default=4, help='チャンク数')
    parser.add_argument('--no-limit', action='store_true', help='LIMIT無し（全件処理）')
    parser.add_argument('--true-batch', action='store_true', help='真のバッチ処理を有効化')
    
    args = parser.parse_args()
    
    # conda環境確認
    ensure_conda_env()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ベンチマーク引数を構築
    benchmark_args = [
        '--parallel', str(args.parallel),
        '--chunks', str(args.chunks)
    ]
    
    if args.no_limit:
        benchmark_args.append('--no-limit')
    
    if args.true_batch:
        benchmark_args.append('--true-batch')
    
    # 各パターンを実行
    metrics_data = {}
    
    if not args.skip_run:
        for pattern in args.patterns:
            metrics = run_benchmark(pattern, benchmark_args)
            if metrics:
                metrics_data[pattern] = metrics
            else:
                print(f"⚠️  {pattern}パターンのメトリクスが取得できませんでした")
    else:
        # 既存のメトリクスファイルを読み込み
        for pattern in args.patterns:
            metrics_path = METRICS_FILES[pattern]
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data[pattern] = json.load(f)
                print(f"✅ 既存メトリクス読み込み: {pattern}")
            else:
                print(f"⚠️  メトリクスファイルが見つかりません: {metrics_path}")
    
    if not metrics_data:
        print("❌ 比較可能なメトリクスがありません")
        return
    
    # 結果を比較
    compare_metrics(metrics_data)
    
    # チャート生成
    create_comparison_charts(metrics_data, args.output_dir)
    
    # レポート生成
    generate_report(metrics_data, args.output_dir)
    
    # 結果をJSON形式でも保存
    summary_path = os.path.join(args.output_dir, f'comparison_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✅ サマリー保存: {summary_path}")

if __name__ == '__main__':
    main()