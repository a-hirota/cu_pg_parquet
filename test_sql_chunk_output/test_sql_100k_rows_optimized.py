#!/usr/bin/env python

"""
SQLクエリを用いた大規模データ処理のテスト (10万行) - パフォーマンス最適化版
"""

import time
import os
import sys
from typing import Optional

# モジュールパスの追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpupaser.main import PgGpuProcessor

def test_process_custom_query(sql_query: str, output_file: Optional[str] = None):
    """カスタムSQLクエリを実行して結果を確認する

    Args:
        sql_query: 実行するSQLクエリ
        output_file: 出力Parquetファイルパス
    """
    print(f"=== テスト: SQLクエリを使った大規模データ処理 ({sql_query}) ===")
    start_time = time.time()
    
    # デバッグ環境変数の設定（デバッグファイル出力を無効化して処理を高速化）
    os.environ['GPUPASER_DEBUG_FILES'] = '0'
    os.environ['GPUPASER_VERBOSE'] = '0'  # 詳細ログを抑制
    
    # 処理クラスのインスタンス化
    processor = PgGpuProcessor()
    
    try:
        # SQLクエリ処理
        print("SQLクエリを実行中...")
        result = processor.process_custom_query(sql_query, output_file)
        
        # 処理時間
        elapsed_time = time.time() - start_time
        print(f"\n処理時間: {elapsed_time:.3f}秒")
        
        # 結果確認
        if output_file and os.path.exists(output_file):
            print(f"\nParquetファイル出力: {output_file}")
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"ファイルサイズ: {file_size:.2f} MB")
            
            # cuDFで結果検証
            try:
                import cudf
                print("\ncuDFで読み込みテスト...")
                df = cudf.read_parquet(output_file)
                num_rows = len(df)
                print(f"データフレーム行数: {num_rows}")
                print("\n先頭10行:")
                print(df.head(10))
                
                if num_rows > 20:
                    print("\n最後の10行:")
                    print(df.tail(10))
                    
                return True
            except ImportError:
                print("cuDFがインストールされていないため、読み込みテストをスキップします")
            except Exception as e:
                print(f"Parquetファイル読み込み中にエラー: {e}")
                return False
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        processor.close()
        
    return True

if __name__ == "__main__":
    # 10万行のテスト（小さいチャンクサイズを使用）
    sql_query = "SELECT * FROM lineorder LIMIT 100000"
    output_file = "test_sql_chunk_output/lineorder_100k_optimized.parquet"
    
    # チャンクサイズの環境変数設定（より小さいチャンクで処理）
    os.environ['GPUPASER_MAX_CHUNK_SIZE'] = '10000'  # チャンクサイズを小さく設定
    
    success = test_process_custom_query(sql_query, output_file)
    
    if success:
        print("\n=== テスト成功: 10万行のデータ処理が正常に完了しました ===")
    else:
        print("\n=== テスト失敗: 10万行のデータ処理中にエラーが発生しました ===")
