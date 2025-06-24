"""lineorderテーブル全体での比較検証

基本ベンチマークと直接抽出版の両方で文字列破損が発生するか確認
"""

import os
import sys
import subprocess
import time
import cudf
import numpy as np
from pathlib import Path

print("=== lineorder全体での比較検証 ===")
print("基本ベンチマーク vs 直接抽出版")

# 環境設定
os.environ['GPUPASER_PG_DSN'] = "dbname=postgres user=postgres host=localhost port=5432"

def run_benchmark(script_path, output_prefix):
    """ベンチマークを実行し結果を返す"""
    print(f"\n実行中: {script_path}")
    start_time = time.time()
    
    try:
        # Python経由で実行（現在の環境を引き継ぐ）
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"エラー: {result.stderr}")
            return None, elapsed_time
        
        # 出力ファイルを確認
        output_files = []
        for i in range(8):  # 8チャンク
            file_path = f"benchmark/chunk_{i}_{output_prefix}.parquet"
            if os.path.exists(file_path):
                output_files.append(file_path)
        
        return output_files, elapsed_time
        
    except Exception as e:
        print(f"実行エラー: {e}")
        return None, time.time() - start_time

def check_string_corruption(parquet_files, sample_size=100000):
    """Parquetファイルから文字列破損をチェック"""
    print("\n文字列破損チェック中...")
    
    total_rows = 0
    even_errors = 0
    odd_errors = 0
    first_error = None
    
    for i, file_path in enumerate(parquet_files[:4]):  # 最初の4チャンクのみ
        print(f"\nチャンク{i}: {file_path}")
        
        try:
            # cuDFで読み込み
            df = cudf.read_parquet(file_path)
            rows = len(df)
            total_rows += rows
            print(f"  行数: {rows:,}")
            
            # lo_orderpriority列をチェック
            if 'lo_orderpriority' in df.columns:
                # 最初の部分をチェック
                check_rows = min(sample_size, rows)
                
                for row_idx in range(min(10000, check_rows)):
                    try:
                        value = df['lo_orderpriority'].iloc[row_idx]
                        
                        # 正常なパターン
                        expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                        is_valid = any(value.startswith(p) for p in expected_patterns)
                        
                        if not is_valid:
                            if row_idx % 2 == 0:
                                even_errors += 1
                            else:
                                odd_errors += 1
                            
                            if first_error is None:
                                first_error = (i, row_idx, value)
                            
                            # 最初の数個のエラーを表示
                            if even_errors + odd_errors <= 10:
                                print(f"  エラー行{row_idx}({'偶数' if row_idx % 2 == 0 else '奇数'}): {repr(value)}")
                    
                    except Exception as e:
                        pass
                
                # 中間部分もチェック（大規模データの場合）
                if rows > 1000000:
                    middle_start = rows // 2
                    for row_idx in range(middle_start, middle_start + 5000):
                        if row_idx >= rows:
                            break
                        try:
                            value = df['lo_orderpriority'].iloc[row_idx]
                            expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                            is_valid = any(value.startswith(p) for p in expected_patterns)
                            
                            if not is_valid:
                                if row_idx % 2 == 0:
                                    even_errors += 1
                                else:
                                    odd_errors += 1
                        except:
                            pass
            
            # メモリ解放
            del df
            
        except Exception as e:
            print(f"  読み込みエラー: {e}")
    
    return {
        'total_rows': total_rows,
        'even_errors': even_errors,
        'odd_errors': odd_errors,
        'first_error': first_error,
        'corruption_rate': (even_errors + odd_errors) / total_rows if total_rows > 0 else 0
    }

# 1. 基本ベンチマーク実行
print("\n=== 1. 基本ベンチマーク（benchmark/benchmark_rust_gpu.py）===")
basic_files, basic_time = run_benchmark(
    "benchmark/benchmark_rust_gpu.py",
    "basic"
)

if basic_files:
    print(f"\n基本ベンチマーク完了: {len(basic_files)}ファイル生成 ({basic_time:.2f}秒)")
    basic_corruption = check_string_corruption(basic_files)
else:
    print("基本ベンチマーク失敗")
    basic_corruption = None

# 2. 直接抽出版実行
print("\n=== 2. 直接抽出版（benchmark/benchmark_rust_gpu_direct.py）===")
direct_files, direct_time = run_benchmark(
    "benchmark/benchmark_rust_gpu_direct.py", 
    "direct"
)

if direct_files:
    print(f"\n直接抽出版完了: {len(direct_files)}ファイル生成 ({direct_time:.2f}秒)")
    direct_corruption = check_string_corruption(direct_files)
else:
    print("直接抽出版失敗")
    direct_corruption = None

# 結果比較
print("\n=== 比較結果 ===")

if basic_corruption:
    print("\n基本ベンチマーク:")
    print(f"  総行数: {basic_corruption['total_rows']:,}")
    print(f"  偶数行エラー: {basic_corruption['even_errors']}")
    print(f"  奇数行エラー: {basic_corruption['odd_errors']}")
    print(f"  破損率: {basic_corruption['corruption_rate']:.6%}")
    if basic_corruption['first_error']:
        chunk, row, value = basic_corruption['first_error']
        print(f"  最初のエラー: チャンク{chunk}, 行{row}, 値={repr(value)}")

if direct_corruption:
    print("\n直接抽出版:")
    print(f"  総行数: {direct_corruption['total_rows']:,}")
    print(f"  偶数行エラー: {direct_corruption['even_errors']}")
    print(f"  奇数行エラー: {direct_corruption['odd_errors']}")
    print(f"  破損率: {direct_corruption['corruption_rate']:.6%}")
    if direct_corruption['first_error']:
        chunk, row, value = direct_corruption['first_error']
        print(f"  最初のエラー: チャンク{chunk}, 行{row}, 値={repr(value)}")

# 診断
print("\n=== 診断 ===")
if basic_corruption and direct_corruption:
    if basic_corruption['odd_errors'] == 0 and direct_corruption['odd_errors'] > 0:
        print("❌ 直接抽出版のみで奇数行破損が発生")
        print("→ 直接抽出版の実装に問題がある可能性が高い")
    elif basic_corruption['odd_errors'] > 0 and direct_corruption['odd_errors'] > 0:
        print("⚠️ 両方で奇数行破損が発生")
        print("→ 共通のコンポーネント（GPUパーサーまたは文字列処理）に問題がある可能性")
    else:
        print("✅ 破損パターンに有意な差はない")

# クリーンアップ
print("\n生成ファイルをクリーンアップ中...")
for files in [basic_files, direct_files]:
    if files:
        for f in files:
            if os.path.exists(f):
                os.remove(f)

print("\n=== 検証完了 ===")