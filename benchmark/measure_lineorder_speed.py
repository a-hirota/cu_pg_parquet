# benchmark/measure_lineorder_speed.py
import os
import time
import numpy as np
import psycopg
import pyarrow as pa
from numba import cuda

# Import necessary functions from the correct modules using absolute paths from root
from src.meta_fetch import fetch_column_meta
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v2 import decode_chunk
# Import the CPU row start calculator from its new location
from src.cpu_parse_utils import calculate_row_starts_cpu

ROWS = 5_000_000  # 計測対象の行数
TABLE_NAME = "lineorder"
CHUNK_SIZE_MB = 256 # GPUへのコピーチャンクサイズ（参考値、現状一括）

def measure_lineorder_speed(verbose=False): # Add verbose flag, default to False
    """lineorderテーブルの指定行数をGPUで処理する速度を計測する"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    if verbose:
        print(f"接続情報: {dsn}")
        print(f"対象テーブル: {tbl}")
        print(f"処理行数: {ROWS:,}")

    conn = None
    try:
        conn = psycopg.connect(dsn)
        if verbose:
            print("PostgreSQL接続完了")

        # -------------------------------
        # 1. メタデータ取得
        # -------------------------------
        start_meta = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {tbl} LIMIT 1") # LIMIT 1で十分
        ncols = len(columns)
        end_meta = time.time()
        if verbose:
            print(f"メタデータ取得完了 ({end_meta - start_meta:.4f} 秒)")
            print(f"カラム数: {ncols}")
        # if verbose: print("カラム情報:")
        # if verbose: for i, c in enumerate(columns):
        #     print(f"  {i}: {c.name} (OID: {c.pg_oid}, ArrowID: {getattr(c, 'arrow_id', 'N/A')})")


        # -------------------------------
        # 2. COPY BINARY でデータ取得
        # -------------------------------
        if verbose: print("COPY BINARY データ取得開始...")
        start_copy = time.time()
        copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {ROWS}) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        # TODO: ストリーミング処理とチャンク転送の実装 (現状は一括読み込み)
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
        raw_host = np.frombuffer(buf, dtype=np.uint8)
        end_copy = time.time()
        if verbose:
            print(f"COPY BINARY データ取得完了 ({end_copy - start_copy:.4f} 秒)")
            print(f"データサイズ: {len(raw_host) / (1024*1024):,.2f} MB")

        # -------------------------------
        # 3. GPUへデータ転送
        # -------------------------------
        if verbose: print("GPUへのデータ転送開始...")
        start_transfer = time.time()
        raw_dev = cuda.to_device(raw_host)
        cuda.synchronize() # 転送完了待ち
        end_transfer = time.time()
        if verbose: print(f"GPUへのデータ転送完了 ({end_transfer - start_transfer:.4f} 秒)")

        # -------------------------------
        # 4. ヘッダーサイズ検出 & 行開始位置計算 (CPU)
        # -------------------------------
        if verbose: print("ヘッダーサイズ検出と行開始位置計算 (CPU) 開始...")
        start_preproc = time.time()
        header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
        header_size = detect_pg_header_size(header_sample)
        if verbose: print(f"  検出ヘッダーサイズ: {header_size} バイト")

        # CPUで行開始位置を計算
        row_start_positions_host = calculate_row_starts_cpu(raw_host, header_size, ROWS)
        row_start_positions_dev = cuda.to_device(row_start_positions_host)
        cuda.synchronize()
        end_preproc = time.time()
        if verbose: print(f"ヘッダーサイズ検出と行開始位置計算完了 ({end_preproc - start_preproc:.4f} 秒)")
        # if verbose: print(f"  行開始位置 (先頭5件): {row_start_positions_host[:5]}")
        # if verbose: print(f"  行開始位置 (末尾5件): {row_start_positions_host[-5:]}")


        # -------------------------------
        # 5. GPU パース (Pass 0: オフセット/長さ計算)
        # -------------------------------
        if verbose: print("GPU パース (Pass 0) 開始...")
        start_pass0 = time.time()
        field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
            raw_dev, ROWS, ncols, header_size=header_size,
            row_start_positions=row_start_positions_dev
        )
        cuda.synchronize() # Pass 0 完了待ち
        end_pass0 = time.time()
        if verbose: print(f"GPU パース (Pass 0) 完了 ({end_pass0 - start_pass0:.4f} 秒)")

        # -------------------------------
        # 6. GPU デコード (Pass 1 & 2 & Arrow組立)
        # -------------------------------
        if verbose: print("GPU デコード (Pass 1 & 2, Arrow組立) 開始...")
        start_decode = time.time()
        # decode_chunk内でPass 1, Prefix Sum, Mem Alloc, Pass 2, from_buffers が実行される
        batch = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
        cuda.synchronize() # デコード完了待ち
        end_decode = time.time()
        if verbose:
            print(f"GPU デコード (Pass 1 & 2, Arrow組立) 完了 ({end_decode - start_decode:.4f} 秒)")
            print(f"  生成された RecordBatch: {batch.num_rows} 行, {batch.num_columns} 列")

        # -------------------------------
        # 7. 総時間
        # -------------------------------
        total_time = end_decode - start_copy # COPY開始からデコード完了まで
        gpu_time = end_decode - start_transfer # GPU転送開始からデコード完了まで
        print("\n--- 計測結果 ---")
        print(f"データ取得時間 (COPY BINARY): {end_copy - start_copy:.4f} 秒")
        print(f"GPU転送時間:                {end_transfer - start_transfer:.4f} 秒")
        print(f"CPU前処理時間:              {end_preproc - start_preproc:.4f} 秒")
        print(f"GPU Pass 0 時間:            {end_pass0 - start_pass0:.4f} 秒")
        print(f"GPU Decode時間 (P1,P2,Arrow): {end_decode - start_decode:.4f} 秒")
        print(f"-----------------------------------")
        print(f"総処理時間 (COPY開始～Arrow完了): {total_time:.4f} 秒")
        print(f"GPU関連時間 (転送開始～Arrow完了): {gpu_time:.4f} 秒")
        print(f"スループット (総時間ベース): {ROWS / total_time:,.2f} 行/秒")
        print(f"スループット (GPU時間ベース): {ROWS / gpu_time:,.2f} 行/秒")

    except psycopg.Error as e:
        print(f"PostgreSQLエラー: {e}")
    except cuda.cudadrv.driver.CudaAPIError as e:
         print(f"CUDAエラー: {e}")
         print("GPUメモリ不足の可能性があります。ROWSを減らして試してください。")
    except Exception as e:
        print(f"予期せぬエラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            if verbose: print("PostgreSQL接続切断")

if __name__ == "__main__":
    # 環境変数 GPUPASER_PG_DSN が設定されていることを確認
    if "GPUPASER_PG_DSN" not in os.environ:
        print("エラー: 環境変数 GPUPASER_PG_DSN を設定してください。")
        print("例: export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'")
    else:
        # verbose=True を渡すと詳細ログが表示される
        measure_lineorder_speed(verbose=False)
