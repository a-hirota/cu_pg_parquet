#!/usr/bin/env python

"""
SQLクエリを用いた大規模データ処理のテスト - デバッグモード
"""

import time
import os
import sys
from typing import Optional
import traceback
import signal

# モジュールパスの追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpupaser.main import PgGpuProcessor

# タイムアウトハンドラー
def timeout_handler(signum, frame):
    print("\n*** タイムアウトが発生しました! GPUカーネルが応答していない可能性があります ***")
    traceback.print_stack(frame)
    raise TimeoutError("GPUカーネル実行がタイムアウトしました")

def test_process_custom_query(sql_query: str, output_file: Optional[str] = None, timeout_seconds: int = 60):
    """カスタムSQLクエリを実行して結果を確認する

    Args:
        sql_query: 実行するSQLクエリ
        output_file: 出力Parquetファイルパス
        timeout_seconds: 実行タイムアウト（秒）
    """
    print(f"=== テスト: SQLクエリを使った大規模データ処理 ({sql_query}) ===")
    start_time = time.time()
    
    # デバッグ環境変数の設定（デバッグログを有効化）
    os.environ['GPUPASER_DEBUG_FILES'] = '1'  # デバッグファイル出力を有効化
    os.environ['GPUPASER_VERBOSE'] = '1'      # 詳細ログを有効化
    
    # タイムアウト設定
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    # 処理クラスのインスタンス化
    processor = PgGpuProcessor()
    
    try:
        # SQLクエリ処理
        print(f"SQLクエリを実行中... （最大{timeout_seconds}秒のタイムアウト設定）")
        
        # デバッグログにタイムスタンプを追加
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - クエリ実行開始")
        
        # 実行前にGPUメモリ状態をチェック
        try:
            from numba import cuda
            mem_info = cuda.current_context().get_memory_info()
            print(f"[DEBUG] GPU Memory: Free {mem_info[0]/1024/1024:.2f}MB / Total {mem_info[1]/1024/1024:.2f}MB")
        except Exception as e:
            print(f"[DEBUG] GPUメモリ情報取得エラー: {e}")
        
        # ここから実行開始
        result = processor.process_custom_query(sql_query, output_file)
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - クエリ実行完了")
        
        # タイムアウト解除
        signal.alarm(0)
        
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
                print("\n先頭5行:")
                print(df.head(5))
                
                if num_rows > 10:
                    print("\n最後の5行:")
                    print(df.tail(5))
                    
                return True
            except ImportError:
                print("cuDFがインストールされていないため、読み込みテストをスキップします")
            except Exception as e:
                print(f"Parquetファイル読み込み中にエラー: {e}")
                return False
    except TimeoutError as e:
        print(f"\n*** {e} ***")
        print("GPUカーネル実行が長時間応答しません。カーネルが無限ループまたはデッドロックしている可能性があります。")
        return False
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        traceback.print_exc()
        return False
    finally:
        # タイムアウト設定を解除
        signal.alarm(0)
        # プロセッサクローズ
        processor.close()
        
        # 終了時間表示
        print(f"[DEBUG] {time.strftime('%H:%M:%S')} - 処理終了")
        
    return True

if __name__ == "__main__":
    # 非常に少ない行数のテスト
    sql_query = "SELECT * FROM lineorder LIMIT 100"
    output_file = "test_sql_chunk_output/lineorder_debug_test.parquet"
    
    # チャンクサイズの設定（さらに小さく設定）
    os.environ['GPUPASER_MAX_CHUNK_SIZE'] = '100'  # 100行に制限
    
    # GPUカーネル設定
    os.environ['GPUPASER_MAX_THREADS'] = '256'
    os.environ['GPUPASER_MAX_BLOCKS'] = '128'
    
    print(f"環境変数設定:")
    print(f"  GPUPASER_MAX_CHUNK_SIZE = {os.environ.get('GPUPASER_MAX_CHUNK_SIZE')}")
    print(f"  GPUPASER_MAX_THREADS = {os.environ.get('GPUPASER_MAX_THREADS')}")
    print(f"  GPUPASER_MAX_BLOCKS = {os.environ.get('GPUPASER_MAX_BLOCKS')}")
    
    # タイムアウト30秒で実行
    success = test_process_custom_query(sql_query, output_file, timeout_seconds=30)
    
    if success:
        print("\n=== テスト成功: デバッグ実行が正常に完了しました ===")
    else:
        print("\n=== テスト失敗: デバッグ実行中にエラーまたはタイムアウトが発生しました ===")
