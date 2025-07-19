#!/usr/bin/env python3
"""
Parquet圧縮方式の性能実験スクリプト

各圧縮方式（snappy、gzip、lz4、brotli、zstd、none）で
ベンチマークを実行し、性能を比較します。
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import argparse

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))


def cleanup_output_dir():
    """出力ディレクトリのクリーンアップ"""
    output_dir = Path("./output")
    if output_dir.exists():
        parquet_files = list(output_dir.glob("*.parquet"))
        if parquet_files:
            print(f"Cleaning up {len(parquet_files)} parquet files...")
            for f in parquet_files:
                f.unlink()


def run_benchmark(table_name: str, parallel: int, chunks: int, compression: str, timeout: int = 300):
    """単一の圧縮方式でベンチマークを実行"""
    print(f"\n{'='*80}")
    print(f"Testing compression: {compression}")
    print(f"{'='*80}")
    
    # 出力ディレクトリのクリーンアップ
    cleanup_output_dir()
    
    # コマンドを構築
    cmd = [
        sys.executable,
        "cu_pg_parquet.py",
        "--table", table_name,
        "--parallel", str(parallel),
        "--chunks", str(chunks),
        "--compression", compression,
        "--yes"  # 自動的にファイルを削除
    ]
    
    # 実行時間計測
    start_time = time.time()
    
    # gzipの場合は警告を表示
    if compression == 'gzip':
        print("⚠️  Warning: gzip is not GPU-accelerated in cuDF and will use CPU fallback")
        print("   This may take significantly longer than other compression methods")
        print(f"   Timeout set to {timeout} seconds")
    
    try:
        # サブプロセスで実行（タイムアウト付き）
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # 出力から統計情報を抽出
        output_lines = result.stdout.split('\n')
        stats = {
            'compression': compression,
            'elapsed_time': elapsed_time,
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # 統計情報の抽出
        for line in output_lines:
            if '総データサイズ:' in line:
                size_str = line.split(':')[1].strip().split()[0]
                stats['total_size_gb'] = float(size_str)
            elif '総行数:' in line:
                rows_str = line.split(':')[1].strip().replace(',', '').split()[0]
                stats['total_rows'] = int(rows_str)
            elif '総実行時間:' in line:
                time_str = line.split(':')[1].strip().replace('秒', '')
                stats['total_time'] = float(time_str)
            elif '全体スループット:' in line:
                throughput_str = line.split(':')[1].strip().split()[0]
                stats['throughput_gb_s'] = float(throughput_str)
        
        # Parquetファイルのサイズを測定
        output_dir = Path("./output")
        parquet_files = list(output_dir.glob("*.parquet"))
        if parquet_files:
            total_parquet_size = sum(f.stat().st_size for f in parquet_files)
            stats['parquet_total_size_mb'] = total_parquet_size / (1024**2)
            stats['parquet_file_count'] = len(parquet_files)
            
            # 圧縮率を計算
            if 'total_size_gb' in stats:
                original_size_mb = stats['total_size_gb'] * 1024
                stats['compression_ratio'] = original_size_mb / stats['parquet_total_size_mb']
        
        print(f"\n✅ Benchmark completed successfully")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        if 'parquet_total_size_mb' in stats:
            print(f"Parquet size: {stats['parquet_total_size_mb']:.2f} MB")
            if 'compression_ratio' in stats:
                print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        
        return stats
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed for {compression}")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        
        return {
            'compression': compression,
            'elapsed_time': time.time() - start_time,
            'success': False,
            'error': str(e),
            'stdout': e.stdout if e.stdout else '',
            'stderr': e.stderr if e.stderr else ''
        }
    except subprocess.TimeoutExpired as e:
        print(f"\n⏱️  Benchmark timed out after {timeout} seconds for {compression}")
        print("   This is expected for gzip compression with large datasets")
        
        return {
            'compression': compression,
            'elapsed_time': timeout,
            'success': False,
            'error': f'Timeout after {timeout} seconds',
            'timeout': True
        }
    except Exception as e:
        print(f"\n❌ Unexpected error for {compression}: {e}")
        return {
            'compression': compression,
            'elapsed_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def save_results(results: list, output_file: str):
    """結果をJSONファイルに保存"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def print_summary(results: list):
    """結果のサマリーを表示"""
    print(f"\n{'='*80}")
    print("COMPRESSION BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    # ヘッダー
    print(f"{'Compression':<12} {'Time (s)':<10} {'Size (MB)':<12} {'Ratio':<8} {'Throughput':<12} {'Status'}")
    print("-" * 80)
    
    # 各結果を表示
    for r in results:
        if r['success']:
            time_str = f"{r.get('total_time', r['elapsed_time']):.2f}"
            size_str = f"{r.get('parquet_total_size_mb', 0):.2f}"
            ratio_str = f"{r.get('compression_ratio', 0):.2f}x"
            throughput_str = f"{r.get('throughput_gb_s', 0):.2f} GB/s"
            status = "✅"
        else:
            time_str = f"{r['elapsed_time']:.2f}"
            size_str = "N/A"
            ratio_str = "N/A"
            throughput_str = "N/A"
            status = "❌"
        
        print(f"{r['compression']:<12} {time_str:<10} {size_str:<12} {ratio_str:<8} {throughput_str:<12} {status}")
    
    print("-" * 80)
    
    # 最適な圧縮方式を特定
    successful_results = [r for r in results if r['success']]
    if successful_results:
        # 速度最優先
        fastest = min(successful_results, key=lambda x: x.get('total_time', x['elapsed_time']))
        print(f"\n🚀 Fastest: {fastest['compression']} ({fastest.get('total_time', fastest['elapsed_time']):.2f}s)")
        
        # 圧縮率最優先
        if any('compression_ratio' in r for r in successful_results):
            best_ratio = max(successful_results, key=lambda x: x.get('compression_ratio', 0))
            print(f"📦 Best compression: {best_ratio['compression']} ({best_ratio.get('compression_ratio', 0):.2f}x)")
        
        # スループット最優先
        if any('throughput_gb_s' in r for r in successful_results):
            best_throughput = max(successful_results, key=lambda x: x.get('throughput_gb_s', 0))
            print(f"⚡ Best throughput: {best_throughput['compression']} ({best_throughput.get('throughput_gb_s', 0):.2f} GB/s)")


def main():
    parser = argparse.ArgumentParser(
        description="Parquet圧縮方式の性能実験"
    )
    parser.add_argument(
        "--table",
        type=str,
        default="lineorder",
        help="対象テーブル名（デフォルト: lineorder）"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="並列接続数（デフォルト: 16）"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=16,
        help="チャンク数（デフォルト: 16）"
    )
    parser.add_argument(
        "--compressions",
        type=str,
        nargs='+',
        default=["snappy", "lz4", "zstd", "none"],
        help="テストする圧縮方式（デフォルト: snappy lz4 zstd none）※gzipは非推奨"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果ファイルの出力先"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="各ベンチマークのタイムアウト秒数（デフォルト: 300秒）"
    )
    parser.add_argument(
        "--include-gzip",
        action="store_true",
        help="gzip圧縮をテストに含める（非推奨、非常に遅い）"
    )
    
    args = parser.parse_args()
    
    # 環境変数の確認
    if "GPUPASER_PG_DSN" not in os.environ:
        print("❌ Error: GPUPASER_PG_DSN environment variable not set")
        print("Example: export GPUPASER_PG_DSN=\"dbname=postgres user=postgres host=localhost port=5432\"")
        return 1
    
    # gzipを含める場合の処理
    compressions = args.compressions.copy()
    if args.include_gzip and 'gzip' not in compressions:
        compressions.append('gzip')
        print("\n⚠️  Note: gzip compression included. This will be MUCH slower than other methods.")
    
    # 結果を格納するリスト
    results = []
    
    # 各圧縮方式でベンチマークを実行
    for compression in compressions:
        stats = run_benchmark(
            table_name=args.table,
            parallel=args.parallel,
            chunks=args.chunks,
            compression=compression,
            timeout=args.timeout
        )
        results.append(stats)
        
        # 少し待機（GPUのクールダウン）
        if compression != args.compressions[-1]:
            print("\nWaiting 5 seconds for GPU cooldown...")
            time.sleep(5)
    
    # 結果のサマリーを表示
    print_summary(results)
    
    # 結果を保存
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"compression_benchmark_{timestamp}.json"
    
    save_results(results, output_file)
    
    return 0


if __name__ == "__main__":
    exit(main())