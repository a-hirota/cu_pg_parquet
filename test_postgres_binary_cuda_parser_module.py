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
    
    # customerテーブルのテスト
    print("\n=== customerテーブル ===")
    print("\n[最適化GPU実装]")
    start_time = time.time()
    try:
        results_customer = load_table_optimized('customer', 4000)  # 行数を制限して高速にテスト
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
        print("\n最初の5行:")
        for col_name, data in results_customer.items():
            print(f"{col_name}: {data[:5]}")
    except Exception as e:
        print(f"Error processing customer table: {e}")

if __name__ == "__main__":
    run_tests()
