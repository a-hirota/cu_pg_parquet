#!/usr/bin/env python3
"""
個別バッファリファクタリングのテストスクリプト

統合バッファ版と個別バッファ版のパフォーマンス比較を実行
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
        elif '個別バッファスループット:' in line:
            try:
                throughput_str = line.split(':')[1].strip().split()[0].replace(',', '')
                timing_info['individual_buffer_throughput'] = int(throughput_str)
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
        elif '個別バッファ効率:' in line:
            try:
                efficiency_str = line.split(':')[1].strip().split('%')[0]
                timing_info['individual_buffer_efficiency'] = float(efficiency_str)
            except:
                pass
        elif 'ベンチマーク完了: 総時間 =' in line:
            try:
                time_str = line.split('=')[1].strip().split()[0]
                timing_info['total_time'] = float(time_str)
            except:
                pass
    
    return timing_info

def compare_results(original_timing, refactored_timing):
    """結果比較"""
    print(f"\n{'='*60}")
    print("パフォーマンス比較結果")
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
        if key in original_timing and key in refactored_timing:
            original_val = original_timing[key]
            refactored_val = refactored_timing[key]
            
            if key.endswith('_time'):
                # 時間系は少ない方が良い
                improvement = ((original_val - refactored_val) / original_val) * 100
                comparison[key] = improvement
                if improvement > 0:
                    print(f"✅ {name}: {original_val:.4f}{unit} → {refactored_val:.4f}{unit} ({improvement:+.1f}% 高速化)")
                else:
                    print(f"❌ {name}: {original_val:.4f}{unit} → {refactored_val:.4f}{unit} ({improvement:+.1f}% 低下)")
            else:
                # スループット系は多い方が良い
                improvement = ((refactored_val - original_val) / original_val) * 100
                comparison[key] = improvement
                if improvement > 0:
                    print(f"✅ {name}: {original_val:,.0f}{unit} → {refactored_val:,.0f}{unit} ({improvement:+.1f}% 向上)")
                else:
                    print(f"❌ {name}: {original_val:,.0f}{unit} → {refactored_val:,.0f}{unit} ({improvement:+.1f}% 低下)")
    
    # 効率性指標
    if 'gpu_efficiency' in original_timing:
        print(f"\n統合バッファ版 GPU使用効率: {original_timing['gpu_efficiency']:.1f}%")
    if 'individual_buffer_efficiency' in refactored_timing:
        print(f"個別バッファ版 バッファ効率: {refactored_timing['individual_buffer_efficiency']:.1f}%")
    
    return comparison

def main():
    """メイン実行"""
    print("個別バッファリファクタリング テスト開始")
    print(f"作業ディレクトリ: {os.getcwd()}")
    
    # 必要なファイルの存在確認
    required_files = [
        'benchmark/benchmark_lineorder_5m.py',
        'benchmark/benchmark_lineorder_5m_individual.py',
        'src/main_postgres_to_parquet.py',
        'src/main_postgres_to_parquet_refactored.py'
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
    
    # 2. 個別バッファ版（リファクタリング版）を実行
    print("\n" + "="*60)
    print("2. 個別バッファ版（リファクタリング版）テスト")
    print("="*60)
    
    refactored_success, refactored_time, refactored_output = run_benchmark(
        'benchmark/benchmark_lineorder_5m_individual.py',
        '個別バッファ版ベンチマーク'
    )
    
    # 3. 結果比較
    if original_success and refactored_success:
        original_timing = extract_timing_info(original_output)
        refactored_timing = extract_timing_info(refactored_output)
        
        print(f"\n統合バッファ版 抽出データ: {original_timing}")
        print(f"個別バッファ版 抽出データ: {refactored_timing}")
        
        comparison = compare_results(original_timing, refactored_timing)
        
        # 総合評価
        print(f"\n{'='*60}")
        print("総合評価")
        print(f"{'='*60}")
        
        positive_improvements = sum(1 for v in comparison.values() if v > 0)
        total_comparisons = len(comparison)
        
        if positive_improvements >= total_comparisons * 0.7:
            print("✅ リファクタリング成功: 大幅な性能向上を確認")
        elif positive_improvements >= total_comparisons * 0.5:
            print("⚠️  リファクタリング部分成功: 一部で性能向上を確認")
        else:
            print("❌ リファクタリング要検討: 性能低下が発生")
        
        print(f"改善項目: {positive_improvements}/{total_comparisons}")
        
    else:
        print(f"\n❌ テスト失敗")
        if not original_success:
            print("  - 統合バッファ版が失敗")
        if not refactored_success:
            print("  - 個別バッファ版が失敗")
        return 1
    
    print(f"\n{'='*60}")
    print("テスト完了")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())