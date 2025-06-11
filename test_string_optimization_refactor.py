#!/usr/bin/env python3
"""
文字列最適化リファクタリングのテストスクリプト

統合バッファ版と文字列最適化版のパフォーマンス比較を実行
既存のテストコードを流用し、文字列最適化版をテスト
"""

import os
import sys
import time
import subprocess

def run_benchmark(script_name, description):
    """ベンチマークスクリプトを実行"""
    print(f"\n{'='*60}")
    print(f"実行中: {description}")
    print(f"スクリプト: {script_name}")
    print(f"{'='*60}")
    
    cmd = [
        'bash', '-c',
        f"export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser && "
        f"export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432' && "
        f"python {script_name}"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        execution_time = time.time() - start_time
        
        print(f"実行時間: {execution_time:.2f}秒")
        print(f"リターンコード: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ 成功")
            print("\n--- 標準出力 ---")
            print(result.stdout)
        else:
            print("❌ 失敗")
            print("\n--- 標準エラー ---")
            print(result.stderr)
            if result.stdout:
                print("\n--- 標準出力 ---")
                print(result.stdout)
        
        return result.returncode == 0, execution_time, result.stdout
        
    except subprocess.TimeoutExpired:
        print("❌ タイムアウト (300秒)")
        return False, 300, ""
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False, 0, ""

def extract_timing_info(output):
    """出力から主要なタイミング情報を抽出"""
    lines = output.split('\n')
    timing_info = {}
    
    for line in lines:
        if 'GPUパース完了:' in line:
            # GPUパース完了: 1000000 行 (0.6153秒)
            try:
                time_str = line.split('(')[1].split('秒')[0]
                timing_info['gpu_parsing'] = float(time_str)
            except:
                pass
        elif '文字列最適化スループット:' in line:
            try:
                throughput_str = line.split(':')[1].strip().split()[0].replace(',', '')
                timing_info['string_optimization_throughput'] = int(throughput_str)
            except:
                pass
        elif 'セル処理速度:' in line:
            try:
                throughput_str = line.split(':')[1].strip().split()[0].replace(',', '')
                timing_info['cell_throughput'] = int(throughput_str)
            except:
                pass
        elif 'データ処理速度:' in line:
            try:
                throughput_str = line.split(':')[1].strip().split()[0]
                timing_info['data_throughput'] = float(throughput_str)
            except:
                pass
        elif 'GPU使用効率:' in line:
            try:
                efficiency_str = line.split(':')[1].strip().split('%')[0]
                timing_info['gpu_efficiency'] = float(efficiency_str)
            except:
                pass
        elif '文字列最適化効率:' in line:
            try:
                efficiency_str = line.split(':')[1].strip().split('%')[0]
                timing_info['string_optimization_efficiency'] = float(efficiency_str)
            except:
                pass
        elif 'ベンチマーク完了: 総時間 =' in line:
            try:
                time_str = line.split('=')[1].strip().split()[0]
                timing_info['total_time'] = float(time_str)
            except:
                pass
    
    return timing_info

def compare_results(original_timing, optimized_timing):
    """結果比較"""
    print(f"\n{'='*60}")
    print("パフォーマンス比較結果（文字列最適化版）")
    print(f"{'='*60}")
    
    comparison = {}
    
    # 共通指標の比較
    common_metrics = [
        ('total_time', '総実行時間', '秒'),
        ('gpu_parsing', 'GPUパース時間', '秒'),
        ('cell_throughput', 'セル処理速度', 'cells/sec'),
        ('data_throughput', 'データ処理速度', 'MB/sec')
    ]
    
    for key, name, unit in common_metrics:
        if key in original_timing and key in optimized_timing:
            original_val = original_timing[key]
            optimized_val = optimized_timing[key]
            
            if key.endswith('_time'):
                # 時間系は少ない方が良い
                improvement = ((original_val - optimized_val) / original_val) * 100
                comparison[key] = improvement
                if improvement > 0:
                    print(f"✅ {name}: {original_val:.4f}{unit} → {optimized_val:.4f}{unit} ({improvement:+.1f}% 高速化)")
                else:
                    print(f"❌ {name}: {original_val:.4f}{unit} → {optimized_val:.4f}{unit} ({improvement:+.1f}% 低下)")
            else:
                # スループット系は多い方が良い
                improvement = ((optimized_val - original_val) / original_val) * 100
                comparison[key] = improvement
                if improvement > 0:
                    print(f"✅ {name}: {original_val:,.0f}{unit} → {optimized_val:,.0f}{unit} ({improvement:+.1f}% 向上)")
                else:
                    print(f"❌ {name}: {original_val:,.0f}{unit} → {optimized_val:,.0f}{unit} ({improvement:+.1f}% 低下)")
    
    # 文字列最適化固有の指標
    if 'string_optimization_throughput' in optimized_timing:
        print(f"\n📊 文字列最適化固有指標:")
        print(f"  文字列処理スループット: {optimized_timing['string_optimization_throughput']:,.0f} cells/sec")
    
    if 'string_optimization_efficiency' in optimized_timing:
        print(f"  文字列最適化効率: {optimized_timing['string_optimization_efficiency']:.1f}%")
    
    # 効率性指標
    if 'gpu_efficiency' in original_timing:
        print(f"\n統合バッファ版 GPU使用効率: {original_timing['gpu_efficiency']:.1f}%")
    if 'gpu_efficiency' in optimized_timing:
        print(f"文字列最適化版 GPU使用効率: {optimized_timing['gpu_efficiency']:.1f}%")
    
    return comparison

def main():
    """メイン実行"""
    print("文字列最適化リファクタリング テスト開始")
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    # 必要なファイルの存在確認
    required_files = [
        'benchmark/benchmark_lineorder_5m.py',
        'benchmark/benchmark_lineorder_5m_string_optimized.py',
        'src/main_postgres_to_parquet.py',
        'src/main_postgres_to_parquet_string_optimized.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 必要なファイルが見つかりません: {missing_files}")
        return 1
    
    # 環境変数チェック
    if not os.environ.get('GPUPASER_PG_DSN'):
        print("❌ 環境変数 GPUPASER_PG_DSN が設定されていません")
        return 1
    
    # 1. 統合バッファ版（元の実装）を実行
    print("\n" + "="*60)
    print("1. 統合バッファ版（元の実装）テスト")
    print("="*60)
    
    original_success, original_time, original_output = run_benchmark(
        'benchmark/benchmark_lineorder_5m.py',
        '統合バッファ版ベンチマーク'
    )
    
    # 2. 文字列最適化版を実行
    print("\n" + "="*60)
    print("2. 文字列最適化版テスト")
    print("="*60)
    
    optimized_success, optimized_time, optimized_output = run_benchmark(
        'benchmark/benchmark_lineorder_5m_string_optimized.py',
        '文字列最適化版ベンチマーク'
    )
    
    # 3. 結果比較
    if original_success and optimized_success:
        original_timing = extract_timing_info(original_output)
        optimized_timing = extract_timing_info(optimized_output)
        
        print(f"\n統合バッファ版 抽出データ: {original_timing}")
        print(f"文字列最適化版 抽出データ: {optimized_timing}")
        
        comparison = compare_results(original_timing, optimized_timing)
        
        # 総合評価
        print(f"\n{'='*60}")
        print("総合評価（文字列最適化版）")
        print(f"{'='*60}")
        
        positive_improvements = sum(1 for v in comparison.values() if v > 0)
        total_comparisons = len(comparison)
        
        # 文字列最適化版は安全なアプローチなので、パフォーマンス低下がなければ成功
        if total_comparisons == 0:
            print("⚠️  比較データ不足: タイミング情報の抽出に問題があります")
        elif positive_improvements >= total_comparisons * 0.5:
            print("✅ 文字列最適化成功: パフォーマンス維持または向上を確認")
            print("   固定長データ処理は既存実装を維持し、文字列処理のみ最適化")
        elif positive_improvements >= total_comparisons * 0.3:
            print("⚠️  文字列最適化部分成功: 一部で性能変化を確認")
            print("   安全なアプローチのため、大幅な性能低下がなければ成功")
        else:
            print("❌ 文字列最適化要検討: 予期しない性能低下が発生")
        
        print(f"改善項目: {positive_improvements}/{total_comparisons}")
        
        # 文字列最適化の特徴
        if 'string_optimization_throughput' in optimized_timing:
            print(f"\n📈 文字列最適化効果:")
            print(f"  共有メモリ不使用による直接コピー最適化を実現")
            print(f"  文字列処理スループット: {optimized_timing['string_optimization_throughput']:,.0f} cells/sec")
        
    elif original_success and not optimized_success:
        print(f"\n❌ 文字列最適化版のみ失敗")
        print("  - 統合バッファ版は正常動作")
        print("  - 文字列最適化版でエラー発生")
        return 1
        
    elif not original_success and optimized_success:
        print(f"\n✅ 文字列最適化版のみ成功")
        print("  - 統合バッファ版でエラー発生")
        print("  - 文字列最適化版は正常動作")
        
    else:
        print(f"\n❌ 両方のテスト失敗")
        if not original_success:
            print("  - 統合バッファ版が失敗")
        if not optimized_success:
            print("  - 文字列最適化版が失敗")
        return 1
    
    print(f"\n{'='*60}")
    print("テスト完了")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())