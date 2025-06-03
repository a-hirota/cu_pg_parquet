#!/usr/bin/env python3
"""
DECIMAL vs STRING 性能比較ベンチマーク

このスクリプトは、NUMERIC列をDECIMAL128として処理する場合と
文字列として処理する場合の性能を比較します。
"""

import os
import sys
import time

# パッケージのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_comparison_benchmark():
    """
    DECIMAL vs STRING の性能比較を実行
    """
    print("=" * 70)
    print("DECIMAL vs STRING PERFORMANCE COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # 1. DECIMAL128処理のテスト
    print("\n1. Testing DECIMAL128 processing...")
    print("-" * 50)
    os.environ['NUMERIC_AS_STRING'] = '0'
    os.environ['USE_DECIMAL_OPTIMIZATION'] = '1'
    
    try:
        from benchmark.benchmark_lineorder_5m import run_benchmark
        
        start_time = time.time()
        run_benchmark()
        decimal_time = time.time() - start_time
        results['decimal'] = decimal_time
        print(f"\nDECIMAL128 processing completed in {decimal_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in DECIMAL128 test: {e}")
        results['decimal'] = None
    
    # 2. STRING処理のテスト
    print("\n" + "=" * 70)
    print("\n2. Testing STRING processing...")
    print("-" * 50)
    os.environ['NUMERIC_AS_STRING'] = '1'
    
    try:
        # モジュールを再インポートして設定を反映
        import importlib
        import benchmark.benchmark_lineorder_5m
        importlib.reload(benchmark.benchmark_lineorder_5m)
        from benchmark.benchmark_lineorder_5m import run_benchmark
        
        start_time = time.time()
        run_benchmark()
        string_time = time.time() - start_time
        results['string'] = string_time
        print(f"\nSTRING processing completed in {string_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in STRING test: {e}")
        results['string'] = None
    
    # 3. 結果の比較
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 70)
    
    if results['decimal'] and results['string']:
        speedup = results['decimal'] / results['string']
        print(f"DECIMAL128 processing: {results['decimal']:.2f} seconds")
        print(f"STRING processing:     {results['string']:.2f} seconds")
        print(f"Speedup:               {speedup:.1f}x faster with STRING")
        
        if speedup > 10:
            print("\n🚀 EXCELLENT: STRING processing is significantly faster!")
        elif speedup > 2:
            print("\n✅ GOOD: STRING processing shows meaningful improvement")
        else:
            print("\n⚠️  MINIMAL: Performance difference is small")
            
    else:
        print("⚠️  Could not complete both tests for comparison")
        if results['decimal']:
            print(f"DECIMAL128 processing: {results['decimal']:.2f} seconds")
        if results['string']:
            print(f"STRING processing: {results['string']:.2f} seconds")
    
    print("\nTest completed!")

if __name__ == "__main__":
    run_comparison_benchmark()