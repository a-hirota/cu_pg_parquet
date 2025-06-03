#!/usr/bin/env python3
"""
Decimal変換のpass2性能分析用ベンチマークスクリプト

このスクリプトは、decimal変換のpass2が遅い原因を分析するために、
詳細なタイミング情報とパフォーマンス指標を出力します。
"""

import os
import sys
import time
import numpy as np

# パッケージのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_decimal_benchmark():
    """
    Decimal変換のpass2ベンチマークを実行
    """
    print("=" * 60)
    print("DECIMAL PASS2 PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # 環境変数を設定してデバッグログを有効化
    os.environ['USE_DECIMAL_OPTIMIZATION'] = '1'
    # NUMERIC列を文字列として扱う（高速化）
    os.environ['NUMERIC_AS_STRING'] = '1'
    
    try:
        # lineorderベンチマークを直接import
        try:
            from benchmark.benchmark_lineorder_5m import run_benchmark
            print("Running lineorder benchmark from benchmark.benchmark_lineorder_5m")
            print("-" * 40)
            run_benchmark()
        except ImportError:
            # fallback: パスを追加してimport
            benchmark_path = os.path.join(os.path.dirname(__file__), 'benchmark')
            if benchmark_path not in sys.path:
                sys.path.insert(0, benchmark_path)
            from benchmark_lineorder_5m import run_benchmark
            print("Running lineorder benchmark from benchmark_lineorder_5m")
            print("-" * 40)
            run_benchmark()
            
    except Exception as e:
        print(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)
    
    print("\nDECIMAL PASS2 ANALYSIS SUMMARY:")
    print("- Check the timing output above for:")
    print("  1. GPU vs CPU timing differences")
    print("  2. Memory bandwidth utilization")
    print("  3. Per-row processing time")
    print("  4. Sync overhead")
    print("\nPOSSIBLE BOTTLENECKS TO INVESTIGATE:")
    print("- If GPU bandwidth < 100 GB/s: Memory bandwidth limited")
    print("- If sync overhead > 50% of total time: CPU-GPU synchronization issue")
    print("- If per-row time > 0.1ms: Computational bottleneck")
    print("- If free GPU memory < 1GB: Memory pressure")

if __name__ == "__main__":
    run_decimal_benchmark()