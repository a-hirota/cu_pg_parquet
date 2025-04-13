"""
PostgreSQL-GPU処理パイプライン テストスクリプト

モジュール化されたコードを使用するエントリーポイント
"""

import time
from gpupaser.main import load_table_optimized

def run_tests():
    """テスト実行関数"""
    
    # date1テーブルのテスト
    print("=== date1テーブル ===")
    print("\n[最適化GPU実装]")
    start_time = time.time()
    try:
        results_date1 = load_table_optimized('date1')
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
        print("\n最初の5行:")
        for col_name, data in results_date1.items():
            print(f"{col_name}: {data[:5]}")
    except Exception as e:
        print(f"Error processing date1 table: {e}")
    
    # customerテーブル（小規模）のテスト
    print("\n=== customerテーブル（小規模） ===")
    print("\n[最適化GPU実装]")
    start_time = time.time()
    try:
        # チャンクサイズ以下の行数でテスト (65000行以下)
        test_rows = 60000
        print(f"テスト行数: {test_rows}行")
        results_customer = load_table_optimized('customer', test_rows)
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
        print("\n最初の5行:")
        for col_name, data in results_customer.items():
            print(f"{col_name}: {data[:5]}")
    except Exception as e:
        print(f"Error processing customer table: {e}")
    
    # customerテーブル（70,000行）のテスト - チャンク分割処理のテスト
    print("\n=== customerテーブル（70,000行） ===")
    print("\n[最適化GPU実装 - 複数チャンク処理]")
    start_time = time.time()
    try:
        # 70,000行でテスト（チャンクサイズ65535を超える量）
        test_rows = 70000
        print(f"テスト行数: {test_rows}行")
        results_customer_70k = load_table_optimized('customer', test_rows)
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
        
        # 実際に処理された行数を確認
        if results_customer_70k:
            first_col_name = next(iter(results_customer_70k))
            actual_rows = len(results_customer_70k[first_col_name])
            print(f"実際に処理された行数: {actual_rows}行")
            
            print("\n最初の5行と最後の5行:")
            for col_name, data in results_customer_70k.items():
                print(f"{col_name}: 先頭5行={data[:5]}, 末尾5行={data[-5:]}")
    except Exception as e:
        print(f"Error processing 70k customer table: {e}")
    
    # customerテーブル（100,000行）のテスト - チャンク分割処理のテスト
    print("\n=== customerテーブル（100,000行） ===")
    print("\n[最適化GPU実装 - 複数チャンク処理]")
    start_time = time.time()
    try:
        # チャンクサイズを超える行数でテスト (65535行超)
        test_rows = 100000
        print(f"テスト行数: {test_rows}行")
        results_customer_large = load_table_optimized('customer', test_rows)
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
        
        # 実際に処理された行数を確認
        if results_customer_large:
            first_col_name = next(iter(results_customer_large))
            actual_rows = len(results_customer_large[first_col_name])
            print(f"実際に処理された行数: {actual_rows}行")
            
            print("\n最初の5行と最後の5行:")
            for col_name, data in results_customer_large.items():
                print(f"{col_name}: 先頭5行={data[:5]}, 末尾5行={data[-5:]}")
    except Exception as e:
        print(f"Error processing large customer table: {e}")

def test_increasing_rows():
    """増分テスト - 徐々に行数を増やしていく"""
    print("\n=== 増分テスト ===")
    
    start_rows = 60000    # 現在成功している行数
    increment = 20000     # 増分
    max_attempt = 200000  # 最大試行行数
    
    current_rows = start_rows
    while current_rows <= max_attempt:
        print(f"\n--- customerテーブル {current_rows}行テスト ---")
        try:
            start_time = time.time()
            results = load_table_optimized('customer', current_rows)
            elapsed = time.time() - start_time
            
            # 実際に処理された行数を確認
            if results:
                first_col_name = next(iter(results))
                actual_rows = len(results[first_col_name])
                print(f"✅ {actual_rows}行の処理に成功: {elapsed:.2f}秒")
            else:
                print(f"✅ 処理に成功（結果なし）: {elapsed:.2f}秒") 
            
            # 次の試行行数
            current_rows += increment
        except Exception as e:
            print(f"❌ {current_rows}行の処理でエラー: {e}")
            # エラー時は増分を小さくして再試行
            increment = max(5000, increment // 2)
            current_rows -= increment
            if increment < 5000:
                print(f"最大処理可能行数: 約{current_rows}行")
                break
    
    return current_rows

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PostgreSQL GPU Parser Test')
    parser.add_argument('--incremental', action='store_true', help='Run incremental row test')
    args = parser.parse_args()
    
    if args.incremental:
        test_increasing_rows()
    else:
        run_tests()
