"""
PostgreSQL → GPU直接コピー（シンプル版）

ユーザー提供例を基にした、最小限の実装:
```python
import psycopg, rmm, kvikio
from kvikio import CuFile

rows_est   = 50_000_000
row_bytes  = 17*8 + 17*4   # 概算
dbuf       = rmm.DeviceBuffer(size=19 + rows_est*row_bytes + 2)

with psycopg.connect("dbname=bench") as conn:
    with conn.cursor() as cur:
        with cur.copy("COPY lineorder TO STDOUT (FORMAT BINARY)") as copy:
            offset = 0
            for chunk in copy:
                dbuf.copy_from_host(buffer=chunk, dst_offset=offset)
                offset += len(chunk)
```

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列 (例: "dbname=postgres user=postgres host=localhost port=5432")
"""

import os
import time
import psycopg
import rmm
import numpy as np
from numba import cuda
import argparse

def simple_direct_gpu_copy(table_name="lineorder", limit_rows=50_000_000):
    """
    シンプルなGPU直接コピー実装
    
    Args:
        table_name: テーブル名
        limit_rows: 処理行数制限
    """
    # 環境変数から接続文字列を取得
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        print("例: export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'")
        return None

    print(f"=== シンプルGPU直接コピー実装 ===")
    print(f"テーブル: {table_name}")
    print(f"行数制限: {limit_rows:,}")
    print(f"接続先: {dsn}")

    # RMM 初期化
    try:
        if not rmm.is_initialized():
            rmm.reinitialize(pool_allocator=True, initial_pool_size=8*1024**3)  # 8GB
            print("✅ RMM メモリプール初期化完了")
    except Exception as e:
        print(f"❌ RMM初期化エラー: {e}")
        return None

    # データサイズ推定（ユーザー例と同じ計算方法）
    rows_est = limit_rows
    row_bytes = 17*8 + 17*4  # 17列想定: 8バイト×8列 + 4バイト×9列
    header_bytes = 19        # PostgreSQL COPY BINARYヘッダー
    buffer_size = header_bytes + rows_est * row_bytes + 1024  # 少し余裕

    print(f"推定バッファサイズ: {buffer_size / (1024*1024):.2f} MB")

    # 注：GPU バッファは COPY 完了後に実データサイズで確保
    print(f"推定バッファサイズ: {buffer_size / (1024*1024):.2f} MB")

    # PostgreSQL接続 & COPY実行
    print("PostgreSQL接続 & GPU直接コピー開始...")
    start_time = time.time()
    
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                # COPY クエリ（ユーザー例と同様）
                copy_sql = f"COPY (SELECT * FROM {table_name} LIMIT {limit_rows}) TO STDOUT (FORMAT BINARY)"
                print(f"実行SQL: {copy_sql}")
                
                with cur.copy(copy_sql) as copy:
                    offset = 0
                    chunk_count = 0
                    total_bytes = 0
                    
                    # チャンクを逐次受け取り、GPU に直接書き込み
                    for chunk in copy:
                        if chunk:
                            chunk_size = len(chunk)
                            
                            # バッファオーバーフロー チェック
                            if offset + chunk_size > dbuf.size:
                                print(f"⚠️  警告: バッファサイズ不足")
                                print(f"   現在オフセット: {offset:,}")
                                print(f"   チャンクサイズ: {chunk_size:,}")
                                print(f"   バッファサイズ: {dbuf.size:,}")
                                break
                            
                            # 【重要】GPU バッファに直接コピー（RMM 25.x対応）
                            # Numba CUDA Driver で host → GPU + offset
                            cuda.cudadrv.driver.memcpy_htod(
                                int(dbuf.ptr) + offset,  # dst GPU ptr + オフセット
                                chunk,                   # src host bytes
                                chunk_size               # サイズ
                            )
                            offset += chunk_size
                            chunk_count += 1
                            total_bytes += chunk_size
                            
                            # 進捗表示を無効化
                            # if chunk_count % 500 == 0:
                            #     print(f"  📊 チャンク {chunk_count:,} | {total_bytes / (1024*1024):.2f} MB")
    
    except Exception as e:
        print(f"❌ PostgreSQL処理エラー: {e}")
        return None

    copy_time = time.time() - start_time
    
    # GPU バッファ確保 & 1回コピー
    print("GPU バッファ確保 & 1回コピー実行中...")
    start_gpu_time = time.time()
    
    # 実データサイズに合わせてバッファ確保
    dbuf = rmm.DeviceBuffer(size=total_bytes)
    
    # RMM 25.x 正しいAPI - 位置引数1個のみ
    dbuf.copy_from_host(host_bytes)
    
    # ホストメモリ解放
    del host_bytes
    
    gpu_time = time.time() - start_gpu_time
    
    print(f"✅ GPU 1回コピー完了!")
    print(f"--- 結果 ---")
    print(f"  COPY実行時間: {copy_time:.4f} 秒")
    print(f"  GPU転送時間 : {gpu_time:.4f} 秒")
    print(f"  処理チャンク: {chunk_count:,} 個")
    print(f"  データサイズ: {total_bytes / (1024*1024):.2f} MB")
    print(f"  ネットワーク速度: {total_bytes / (1024*1024) / copy_time:.2f} MB/sec")
    print(f"  GPU転送速度   : {total_bytes / (1024*1024) / gpu_time:.2f} MB/sec")

    # GPU バッファの内容確認（最初の数バイト）
    print(f"\n--- GPUバッファ内容確認 ---")
    try:
        # 最初の32バイトをホストにコピーして確認
        sample_size = min(32, offset)
        if sample_size > 0:
            sample_buf = rmm.DeviceBuffer(size=sample_size)
            sample_buf.copy_from_device(dbuf, size=sample_size)
            
            # numba GPU アレイとして取得
            gpu_array = cuda.as_cuda_array(sample_buf).view(dtype=np.uint8)
            sample_bytes = gpu_array.copy_to_host()
            
            print(f"先頭 {sample_size} バイト:")
            hex_str = ' '.join(f'{b:02x}' for b in sample_bytes)
            print(f"  Hex: {hex_str}")
            print(f"  Dec: {list(sample_bytes)}")
            
    except Exception as e:
        print(f"バッファ内容確認エラー: {e}")

    return {
        'dbuf': dbuf,
        'size': offset,
        'chunks': chunk_count,
        'time': copy_time,
        'throughput_mbps': total_bytes / (1024*1024) / copy_time
    }

def save_gpu_buffer_to_file(dbuf, size, output_path):
    """
    GPU バッファをファイルに保存する例
    （kvikio CuFile または通常のファイル保存）
    """
    print(f"\n--- GPUバッファをファイル保存 ---")
    print(f"出力パス: {output_path}")
    
    try:
        # kvikio CuFile を試行
        from kvikio import CuFile
        
        start_time = time.time()
        with CuFile(output_path, 'w') as f:
            # 実際のデータサイズのみ書き込み
            trimmed_buf = rmm.DeviceBuffer(size=size)
            trimmed_buf.copy_from_device(dbuf, size=size)
            f.pwrite(trimmed_buf)
        
        save_time = time.time() - start_time
        print(f"✅ CuFile保存完了 ({save_time:.4f}秒)")
        print(f"   サイズ: {size / (1024*1024):.2f} MB")
        print(f"   速度: {size / (1024*1024) / save_time:.2f} MB/sec")
        
    except ImportError:
        print("⚠️  kvikio (CuFile) が利用できません。通常の方法で保存します...")
        
        # フォールバック: GPU→ホスト→ファイル
        start_time = time.time()
        gpu_array = cuda.as_cuda_array(dbuf).view(dtype=np.uint8)
        host_data = gpu_array[:size].copy_to_host()
        
        with open(output_path, 'wb') as f:
            f.write(host_data.tobytes())
        
        save_time = time.time() - start_time
        print(f"✅ 通常保存完了 ({save_time:.4f}秒)")
        print(f"   サイズ: {size / (1024*1024):.2f} MB")
        print(f"   速度: {size / (1024*1024) / save_time:.2f} MB/sec")
        
    except Exception as e:
        print(f"❌ ファイル保存エラー: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU直接コピー（シンプル版）')
    parser.add_argument('--table', type=str, default='lineorder', help='テーブル名')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    parser.add_argument('--save', type=str, help='GPU生データ保存パス (optional)')
    
    args = parser.parse_args()
    
    # CUDA初期化確認
    try:
        cuda.current_context()
        print("✅ CUDA context OK")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        exit(1)
    
    # シンプル GPU 直接コピー実行
    result = simple_direct_gpu_copy(
        table_name=args.table,
        limit_rows=args.rows
    )
    
    if result and args.save:
        save_gpu_buffer_to_file(
            result['dbuf'], 
            result['size'], 
            args.save
        )
    
    print(f"\n🎉 シンプルGPU直接コピー完了!")

if __name__ == "__main__":
    main()