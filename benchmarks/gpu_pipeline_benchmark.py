"""
PostgreSQL → Rust → GPU キューベース並列処理版
Producer-Consumerパターンで真の並列処理を実現
"""

import os
import time
import subprocess
import json
import numpy as np
from numba import cuda
import rmm
import cudf
import cupy as cp
import kvikio
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
import pandas as pd

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet_direct
from src.cuda_kernels.postgres_binary_parser import detect_pg_header_size
import pyarrow.parquet as pq

TABLE_NAME = "lineorder"  # デフォルト値（実行時に上書きされる）
OUTPUT_DIR = "/dev/shm"

# スクリプトのディレクトリを基準に相対パスを解決
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # benchmarks -> project_root
RUST_BINARY = os.environ.get('GPUPGPARSER_RUST_BINARY') or os.path.join(
    project_root, "rust_pg_binary_extractor/target/release/pg_chunk_extractor"
)

MAX_QUEUE_SIZE = 3  # キューの最大サイズ

# グローバル変数
chunk_stats = []
gpu_row_counts = {}  # GPU処理行数を保存（チャンクIDをキーとする辞書）
shutdown_flag = threading.Event()


def signal_handler(sig, frame):
    """Ctrl+Cハンドラー"""
    print("\n\n⚠️  処理を中断しています...")
    shutdown_flag.set()
    sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


def setup_rmm_pool():
    """RMMメモリプールを適切に設定"""
    try:
        if rmm.is_initialized():
            return
        
        # GPUメモリの90%を使用可能に設定
        gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
        pool_size = int(gpu_memory * 0.9)
        
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=pool_size,
            maximum_pool_size=pool_size
        )
    except Exception as e:
        print(f"⚠️ RMM初期化警告: {e}")


def convert_rust_metadata_to_column_meta(rust_columns):
    """Rust出力のメタデータをColumnMetaオブジェクトに変換"""
    # src/types.pyから必要な定数をインポート
    from src.types import (
        INT16, INT32, INT64, FLOAT32, FLOAT64, DECIMAL128,
        UTF8, BINARY, DATE32, TS64_US, BOOL
    )
    
    columns = []
    for col in rust_columns:
        # Rust側のarrow_typeを正しいarrow_idに変換
        arrow_type_map = {
            'int16': (INT16, 2),          # INT16 = 0, elem_size = 2
            'int32': (INT32, 4),          # INT32 = 1, elem_size = 4
            'int64': (INT64, 8),          # INT64 = 2, elem_size = 8
            'float32': (FLOAT32, 4),      # FLOAT32 = 3, elem_size = 4
            'float64': (FLOAT64, 8),      # FLOAT64 = 4, elem_size = 8
            'decimal128': (DECIMAL128, 16), # DECIMAL128 = 5, elem_size = 16
            'string': (UTF8, None),       # UTF8 = 6, elem_size = None (可変長)
            'binary': (BINARY, None),     # BINARY = 7, elem_size = None (可変長)
            'date32': (DATE32, 4),        # DATE32 = 8, elem_size = 4
            'timestamp[ns]': (TS64_US, 8),       # TS64_US = 9, elem_size = 8
            'timestamp[ns, tz=UTC]': (TS64_US, 8), # TS64_US = 9, elem_size = 8
            'bool': (BOOL, 1),            # BOOL = 10, elem_size = 1
        }
        
        arrow_type = col['arrow_type']
        arrow_id, elem_size = arrow_type_map.get(arrow_type, (UNKNOWN, None))
        
        # elem_sizeがNoneの場合は0に変換（可変長を示す）
        if elem_size is None:
            elem_size = 0
        
        # arrow_paramの設定（DECIMAL128の場合はデフォルト値を設定）
        arrow_param = None
        if arrow_id == DECIMAL128:
            # Rustからprecision/scaleが渡されていない場合はデフォルト値を使用
            # PostgreSQL numericのデフォルトは(38, 0)
            arrow_param = (38, 0)
        
        # ColumnMetaオブジェクトを作成
        meta = ColumnMeta(
            name=col['name'],
            pg_oid=col['pg_oid'],
            pg_typmod=-1,  # Rustから取得できないため、デフォルト値
            arrow_id=arrow_id,
            elem_size=elem_size,
            arrow_param=arrow_param
        )
        columns.append(meta)
    
    return columns


def cleanup_files(total_chunks=8, table_name=None):
    """ファイルをクリーンアップ"""
    # テーブル名が指定されていない場合はグローバル変数を使用
    if table_name is None:
        table_name = TABLE_NAME
    
    # 通常のクリーンアップ処理
    files = [
        f"{OUTPUT_DIR}/{table_name}_meta_0.json",
        f"{OUTPUT_DIR}/{table_name}_data_0.ready"
    ] + [f"{OUTPUT_DIR}/{table_name}_chunk_{i}.bin" for i in range(total_chunks)]
    
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    
    # 追加の安全対策: OUTPUT_DIR内の全ての.binファイルをクリーンアップ（テーブル名が一致するもののみ）
    try:
        output_path = Path(OUTPUT_DIR)
        if output_path.exists():
            # このテーブルに関連する全ての.binファイルを削除
            for bin_file in output_path.glob(f"{table_name}_*.bin"):
                if bin_file.is_file():
                    bin_file.unlink()
    except Exception as e:
        print(f"⚠️ 追加クリーンアップ中の警告: {e}")


def rust_producer(chunk_queue: queue.Queue, total_chunks: int, stats_queue: queue.Queue, table_name: str, metadata_queue: queue.Queue):
    """Rust転送を実行するProducerスレッド"""
    for chunk_id in range(total_chunks):
        if shutdown_flag.is_set():
            break
            
        try:
            print(f"\n[Producer] チャンク {chunk_id + 1}/{total_chunks} Rust転送開始...")
            
            env = os.environ.copy()
            env['CHUNK_ID'] = str(chunk_id)
            env['TOTAL_CHUNKS'] = str(total_chunks)
            env['TABLE_NAME'] = table_name
            
            rust_start = time.time()
            process = subprocess.run(
                [RUST_BINARY],
                capture_output=True,
                text=True,
                env=env
            )
            
            if process.returncode != 0:
                print(f"❌ Rustエラー: {process.stderr}")
                continue
            
            # デバッグ: Rustの全出力を確認
            if os.environ.get("GPUPGPARSER_TEST_MODE") == "1" and chunk_id == 0:
                print(f"[Producer] Rust stdout長さ: {len(process.stdout)}文字")
                print(f"[Producer] Rust stderr: {process.stderr}" if process.stderr else "[Producer] stderrは空")
            
            # テストモードの場合、Rustの出力を表示
            if os.environ.get("GPUPGPARSER_TEST_MODE") == "1":
                for line in process.stdout.split('\n'):
                    if 'チャンク' in line or 'ページ' in line or 'COPY範囲' in line:
                        print(f"[Rust Debug] {line}")
                # JSON出力を確認
                if "===CHUNK_RESULT_JSON===" not in process.stdout:
                    print("[Producer] 警告: RustからJSON出力がありません")
                    print(f"[Producer] Rust出力の末尾100文字: ...{process.stdout[-100:]}")
            
            # JSON結果を抽出
            output = process.stdout
            json_start = output.find("===CHUNK_RESULT_JSON===")
            json_end = output.find("===END_CHUNK_RESULT_JSON===")
            
            # デバッグ: 最初のチャンクでJSONが見つからない場合
            if chunk_id == 0 and json_start == -1 and os.environ.get("GPUPGPARSER_TEST_MODE") == "1":
                print(f"[Producer] Rust stdout全体:\n{output}")
                print(f"[Producer] JSON検索結果: start={json_start}, end={json_end}")
            
            if json_start != -1 and json_end != -1:
                json_str = output[json_start + len("===CHUNK_RESULT_JSON==="):json_end].strip()
                result = json.loads(json_str)
                rust_time = result['elapsed_seconds']
                file_size = result['total_bytes']
                chunk_file = result['chunk_file']
                
                # 最初のチャンクの場合、メタデータも抽出して保存
                if chunk_id == 0 and 'columns' in result:
                    metadata_queue.put(result['columns'])
            else:
                rust_time = time.time() - rust_start
                chunk_file = f"{OUTPUT_DIR}/{table_name}_chunk_{chunk_id}.bin"
                file_size = os.path.getsize(chunk_file)
            
            chunk_info = {
                'chunk_id': chunk_id,
                'chunk_file': chunk_file,
                'file_size': file_size,
                'rust_time': rust_time
            }
            
            print(f"[Producer] チャンク {chunk_id + 1} 転送完了 ({rust_time:.1f}秒, {file_size / 1024**3:.1f}GB)")
            
            # キューに追加（ブロッキング）
            chunk_queue.put(chunk_info)
            stats_queue.put(('rust_time', rust_time))
            
        except Exception as e:
            print(f"[Producer] エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # 終了シグナル
    chunk_queue.put(None)
    print("[Producer] 全チャンク転送完了")


def gpu_consumer(chunk_queue: queue.Queue, columns: List[ColumnMeta], consumer_id: int, stats_queue: queue.Queue, total_chunks: int, table_name: str, test_mode: bool = False):
    """GPU処理を実行するConsumerスレッド"""
    while not shutdown_flag.is_set():
        try:
            # キューからチャンクを取得（ブロッキング）
            chunk_info = chunk_queue.get(timeout=1)
            
            if chunk_info is None:  # 終了シグナル
                break
                
            chunk_id = chunk_info['chunk_id']
            chunk_file = chunk_info['chunk_file']
            file_size = chunk_info['file_size']
            
            print(f"[Consumer-{consumer_id}] チャンク {chunk_id + 1} GPU処理開始...")
            
            # GPUメモリクリア
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            gpu_start = time.time()
            
            # kvikio+RMMで直接GPU転送
            transfer_start = time.time()
            
            
            # RMM DeviceBufferを使用
            gpu_buffer = rmm.DeviceBuffer(size=file_size)
            
            # kvikioで直接読み込み
            with kvikio.CuFile(chunk_file, "rb") as f:
                gpu_array = cp.asarray(gpu_buffer).view(dtype=cp.uint8)
                bytes_read = f.read(gpu_array)
            
            if bytes_read != file_size:
                raise RuntimeError(f"読み込みサイズ不一致: {bytes_read} != {file_size}")
            
            # ファイル存在確認（kvikio読み込み後） - テストモードのみ
            if os.environ.get("GPUPGPARSER_TEST_MODE") == "1":
                if os.path.exists(chunk_file):
                    print(f"[Consumer-{consumer_id}] kvikio読み込み後: {chunk_file} はまだ存在します")
                else:
                    print(f"[Consumer-{consumer_id}] kvikio読み込み後: {chunk_file} が削除されました")
            
            # numba cuda配列に変換（ゼロコピー）
            raw_dev = cuda.as_cuda_array(gpu_buffer).view(dtype=np.uint8)
            
            transfer_time = time.time() - transfer_start
            
            # ヘッダーサイズ検出
            header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
            header_size = detect_pg_header_size(header_sample)
            
            # 直接抽出処理
            chunk_output = f"output/{table_name}_chunk_{chunk_id}_queue.parquet"
            
            # チャンクIDと最後のチャンクかどうかを環境変数で設定
            os.environ['GPUPGPARSER_CURRENT_CHUNK'] = str(chunk_id)
            if chunk_id == total_chunks - 1:
                os.environ['GPUPGPARSER_LAST_CHUNK'] = '1'
            else:
                os.environ['GPUPGPARSER_LAST_CHUNK'] = '0'
            
            cudf_df, detailed_timing = postgresql_to_cudf_parquet_direct(
                raw_dev=raw_dev,
                columns=columns,
                ncols=len(columns),
                header_size=header_size,
                output_path=chunk_output,
                compression='snappy',
                use_rmm=True,
                optimize_gpu=True,
                verbose=False,
                test_mode=(os.environ.get('GPUPGPARSER_TEST_MODE', '0') == '1')
            )
            
            gpu_time = time.time() - gpu_start
            
            # 処理統計
            rows = len(cudf_df) if cudf_df is not None else 0
            
            # ファイル存在確認（GPU処理後）
            if os.path.exists(chunk_file):
                print(f"[Consumer-{consumer_id}] GPU処理後: {chunk_file} はまだ存在します")
            else:
                print(f"[Consumer-{consumer_id}] GPU処理後: {chunk_file} が削除されました")
            
            # GPU処理行数を保存
            gpu_row_counts[chunk_id] = rows
            
            print(f"[Consumer-{consumer_id}] チャンク {chunk_id + 1} GPU処理完了 ({gpu_time:.1f}秒, {rows:,}行)")
            
            # テストモードで追加情報を表示
            if os.environ.get('GPUPGPARSER_TEST_MODE', '0') == '1':
                print(f"[CHUNK DEBUG] チャンク {chunk_id + 1}: ")
                print(f"  - ファイルサイズ: {file_size / 1024**2:.1f} MB")
                print(f"  - 検出行数: {rows:,}行")
                print(f"  - GPUパース時間: {detailed_timing.get('gpu_parsing', 0):.2f}秒")
                print(f"  - 行あたり: {file_size/rows if rows > 0 else 0:.1f} bytes/row")
            
            # 統計情報を送信
            stats_queue.put(('gpu_time', gpu_time))
            stats_queue.put(('transfer_time', transfer_time))
            stats_queue.put(('rows', rows))
            stats_queue.put(('size', file_size))
            
            # 詳細統計を保存
            chunk_stats.append({
                'chunk_id': chunk_id,
                'consumer_id': consumer_id,
                'rust_time': chunk_info['rust_time'],
                'gpu_time': gpu_time,
                'transfer_time': transfer_time,
                'parse_time': detailed_timing.get('gpu_parsing', 0),
                'string_time': detailed_timing.get('string_buffer_creation', 0),
                'write_time': detailed_timing.get('parquet_export', 0),
                'rows': rows,
                'size_gb': file_size / 1024**3
            })
            
            # メモリ解放
            del raw_dev
            del gpu_buffer
            del gpu_array
            if cudf_df is not None:
                del cudf_df
            
            mempool.free_all_blocks()
            
            # ガベージコレクション
            import gc
            gc.collect()
            
            # テストモードの場合、削除前にバイナリファイルを保存
            if test_mode:
                # テストモード用のディレクトリとタイムスタンプを取得
                if not hasattr(gpu_consumer, 'test_save_dir'):
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    gpu_consumer.test_save_dir = f"test_binaries/{timestamp}"
                    os.makedirs(gpu_consumer.test_save_dir, exist_ok=True)
                    print(f"\n📁 テストモード: バイナリファイル保存先 - {gpu_consumer.test_save_dir}")
                
                # バイナリファイルを保存
                import shutil
                dst = f"{gpu_consumer.test_save_dir}/{table_name}_chunk_{chunk_id}.bin"
                shutil.copy2(chunk_file, dst)
                size = os.path.getsize(chunk_file) / (1024**3)
                print(f"  ✓ {table_name}_chunk_{chunk_id}.bin を保存 ({size:.2f} GB)")
            
            # チャンクファイル削除（中間ファイルのみ）
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            # Parquetファイルは出力ファイルなので保持
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Consumer-{consumer_id}] エラー: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[Consumer-{consumer_id}] 終了")


def validate_parquet_output(file_path: str, num_rows: int = 5, gpu_rows: int = None) -> bool:
    """
    Parquetファイルの検証とサンプル表示
    
    Args:
        file_path: 検証するParquetファイルのパス
        num_rows: 表示する行数
        gpu_rows: GPU処理で検出した行数（比較用）
    
    Returns:
        検証成功の場合True
    """
    try:
        # PyArrowでParquetファイルを読み込む
        table = pq.read_table(file_path)
        
        print(f"\n📊 Parquetファイル検証: {os.path.basename(file_path)}")
        print(f"├─ 行数: {table.num_rows:,}", end="")
        if gpu_rows is not None:
            if table.num_rows == gpu_rows:
                print(f" ✅ OK (GPU処理行数と一致)")
            else:
                print(f" ❌ NG (GPU処理行数: {gpu_rows:,})")
        else:
            print()
        print(f"├─ 列数: {table.num_columns}")
        print(f"└─ ファイルサイズ: {os.path.getsize(file_path) / 1024**2:.2f} MB")
        
        # サンプルデータを表示
        print(f"\n📝 サンプルデータ（先頭{num_rows}行）:")
        print("─" * 80)
        df_sample = table.slice(0, num_rows).to_pandas()
        print(df_sample.to_string(index=False, max_colwidth=20))
        print("─" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Parquet検証エラー: {e}")
        return False


def run_parallel_pipeline(total_chunks: int, table_name: str, test_mode: bool = False):
    """真の並列パイプライン実行"""
    # キューとスレッド管理
    chunk_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    stats_queue = queue.Queue()
    metadata_queue = queue.Queue()
    
    start_time = time.time()
    
    # Producerスレッド開始
    producer_thread = threading.Thread(
        target=rust_producer,
        args=(chunk_queue, total_chunks, stats_queue, table_name, metadata_queue)
    )
    producer_thread.start()
    
    # 最初のチャンクからメタデータを取得
    print("メタデータを取得中...")
    try:
        rust_metadata = metadata_queue.get(timeout=30)  # 30秒タイムアウト
        columns = convert_rust_metadata_to_column_meta(rust_metadata)
        print(f"✅ メタデータ取得完了: {len(columns)} 列")
    except queue.Empty:
        print("❌ メタデータ取得タイムアウト")
        print("ヒント: RustバイナリがJSONを出力しているか確認してください")
        # Producerスレッドの終了を待つ
        producer_thread.join()
        raise RuntimeError("メタデータ取得に失敗しました")
    
    # GPUウォーミングアップ（メタデータ取得後）
    warmup_thread = threading.Thread(
        target=gpu_warmup,
        args=(columns,)
    )
    warmup_thread.start()
    
    # Consumerスレッド開始（1つのみ - GPUメモリ制約）
    consumer_thread = threading.Thread(
        target=gpu_consumer,
        args=(chunk_queue, columns, 1, stats_queue, total_chunks, table_name, test_mode)
    )
    consumer_thread.start()
    
    # 統計収集
    total_rust_time = 0
    total_gpu_time = 0
    total_transfer_time = 0
    total_rows = 0
    total_size = 0
    
    # スレッドの終了を待機しながら統計を収集
    warmup_thread.join()  # ウォーミングアップ完了を待つ
    producer_thread.join()
    consumer_thread.join()
    
    # 統計キューから結果を収集
    while not stats_queue.empty():
        stat_type, value = stats_queue.get()
        if stat_type == 'rust_time':
            total_rust_time += value
        elif stat_type == 'gpu_time':
            total_gpu_time += value
        elif stat_type == 'transfer_time':
            total_transfer_time += value
        elif stat_type == 'rows':
            total_rows += value
        elif stat_type == 'size':
            total_size += value
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'total_rust_time': total_rust_time,
        'total_gpu_time': total_gpu_time,
        'total_transfer_time': total_transfer_time,
        'total_rows': total_rows,
        'total_size': total_size,
        'processed_chunks': len(chunk_stats)
    }


def gpu_warmup(columns):
    """GPUウォーミングアップ - JITコンパイルとCUDA初期化"""
    try:
        print("\n🔥 GPUウォーミングアップ中...")
        
        # PostgreSQL COPY BINARYヘッダー（19バイト）
        header = [
            0x50, 0x47, 0x43, 0x4F, 0x50, 0x59, 0x0A, 0xFF, 0x0D, 0x0A, 0x00,  # PGCOPY
            0x00, 0x00, 0x00, 0x00,  # flags
            0x00, 0x00, 0x00, 0x00   # header extension
        ]
        
        # 1行分のダミーデータ（17列）を作成
        row_data = []
        row_data.extend([0x00, 0x11])  # 17フィールド
        
        for i in range(17):
            if i < 8:  # 数値フィールド（int64）
                row_data.extend([0x00, 0x00, 0x00, 0x08])  # 長さ8
                row_data.extend([0x00] * 8)  # 8バイトのゼロ
            else:  # 文字列フィールド
                row_data.extend([0x00, 0x00, 0x00, 0x04])  # 長さ4
                row_data.extend([0x54, 0x45, 0x53, 0x54])  # "TEST"
        
        # ダミーデータ作成（100KB程度 - より現実的なサイズ）
        dummy_list = header + row_data * 1000  # 1000行分
        # 終端マーカー追加（0xFFFF）
        dummy_list.extend([0xFF, 0xFF])
        dummy_data = np.array(dummy_list, dtype=np.uint8)
        
        # GPU処理実行（JITコンパイルとCUDA初期化）
        # 実際の処理と同じ環境変数を設定
        os.environ['GPUPGPARSER_ROWS_PER_THREAD'] = os.environ.get('GPUPGPARSER_ROWS_PER_THREAD', '32')
        os.environ['GPUPGPARSER_STRING_ROWS_PER_THREAD'] = os.environ.get('GPUPGPARSER_STRING_ROWS_PER_THREAD', '1')
        
        # GPU転送
        import cupy as cp
        gpu_buffer = cp.asarray(dummy_data).view(dtype=cp.uint8)
        raw_dev = cuda.as_cuda_array(gpu_buffer).view(dtype=np.uint8)
        
        # ヘッダーサイズ検出
        header_size = detect_pg_header_size(dummy_data)
        
        # シンプルなカーネル実行でJITコンパイルをトリガー
        from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2_lite
        
        # GPUパースカーネルを直接実行（JITコンパイル確実に実行）
        row_positions, field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2_lite(
            raw_dev, columns, header_size, debug=False, test_mode=False
        )
        
        # 結果を破棄
        del row_positions
        del field_offsets
        del field_lengths
        del gpu_buffer
        del raw_dev
        
        print("✅ GPUウォーミングアップ完了\n")
        
    except Exception as e:
        print(f"⚠️  GPUウォーミングアップ警告: {e}\n")


def main(total_chunks=8, table_name=None, test_mode=False, test_duplicate_keys=None):
    global TABLE_NAME
    if table_name:
        TABLE_NAME = table_name
    else:
        table_name = TABLE_NAME  # デフォルト値を使用
    
    # テストモードの場合、GPU特性を表示
    if test_mode:
        from src.cuda_kernels.postgres_binary_parser import print_gpu_properties
        print_gpu_properties()
    
    # kvikio設定確認
    is_compat = os.environ.get("KVIKIO_COMPAT_MODE", "").lower() in ["on", "1", "true"]
    
    # RMMメモリプール設定
    setup_rmm_pool()
    
    # CUDA context確認
    try:
        cuda.current_context()
    except Exception as e:
        print(f"❌ CUDA context エラー: {e}")
        return
    
    # クリーンアップ
    cleanup_files(total_chunks, table_name)
    
    try:
        # 並列パイプライン実行
        print("\n並列処理を開始します...")
        print("=" * 80)
        
        results = run_parallel_pipeline(total_chunks, table_name, test_mode)
        
        # 最終統計を構造化表示
        total_gb = results['total_size'] / 1024**3
        
        print(f"\n{'='*80}")
        print(f" ✅ 処理完了サマリー")
        print(f"{'='*80}")
        
        # 全体統計
        print(f"\n【全体統計】")
        print(f"├─ 総データサイズ: {total_gb:.2f} GB")
        print(f"├─ 総行数: {results['total_rows']:,} 行")
        print(f"├─ 総実行時間: {results['total_time']:.2f}秒")
        print(f"└─ 全体スループット: {total_gb / results['total_time']:.2f} GB/秒")
        
        # 処理時間内訳
        print(f"\n【処理時間内訳】")
        if results['total_rust_time'] > 0:
            print(f"├─ Rust転送合計: {results['total_rust_time']:.2f}秒 ({total_gb / results['total_rust_time']:.2f} GB/秒)")
        else:
            print(f"├─ Rust転送合計: {results['total_rust_time']:.2f}秒")
        if results['total_gpu_time'] > 0:
            print(f"└─ GPU処理合計: {results['total_gpu_time']:.2f}秒 ({total_gb / results['total_gpu_time']:.2f} GB/秒)")
        else:
            print(f"└─ GPU処理合計: {results['total_gpu_time']:.2f}秒")
        print(f"   └─ kvikio転送: {results['total_transfer_time']:.2f}秒")
        
        
        # チャンク毎の詳細統計テーブル
        if chunk_stats:
            print(f"\n【チャンク毎の処理時間】")
            print(f"┌{'─'*7}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*10}┬{'─'*12}┐")
            print(f"│チャンク│ Rust転送 │kvikio転送│ GPUパース│ 文字列処理│ Parquet │   処理行数  │")
            print(f"├{'─'*7}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*10}┼{'─'*12}┤")
            
            # チャンクIDでソート
            sorted_stats = sorted(chunk_stats, key=lambda x: x['chunk_id'])
            for stat in sorted_stats:
                print(f"│  {stat['chunk_id']:^3}  │ {stat['rust_time']:>6.2f}秒 │ {stat['transfer_time']:>6.2f}秒 │"
                      f" {stat['parse_time']:>6.2f}秒 │ {stat['string_time']:>6.2f}秒 │"
                      f" {stat['write_time']:>6.2f}秒 │{stat['rows']:>10,}行│")
            
            print(f"└{'─'*7}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*10}┴{'─'*12}┘")
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 性能測定完了後、サンプル検証を実行
        sample_parquet = f"output/{table_name}_chunk_0_queue.parquet"
        if os.path.exists(sample_parquet):
            # chunk_0のGPU処理行数を取得
            gpu_rows_chunk0 = gpu_row_counts.get(0, None)
            validate_parquet_output(sample_parquet, num_rows=5, gpu_rows=gpu_rows_chunk0)
            # Parquetファイルは出力ファイルなので保持
        
        # 全Parquetファイルの実際の行数を確認
        print("\n【Parquetファイル統計】")
        actual_total_rows = 0
        parquet_files = sorted(Path("output").glob(f"{table_name}_chunk_*_queue.parquet"))
        for pf in parquet_files:
            try:
                table = pq.read_table(pf)
                actual_total_rows += table.num_rows
                print(f"├─ {pf.name}: {table.num_rows:,} 行")
            except Exception as e:
                print(f"├─ {pf.name}: 読み込みエラー - {e}")
        print(f"└─ 実際の総行数: {actual_total_rows:,} 行")
        
        # --testモードの場合、PostgreSQLテーブル行数と比較
        if os.environ.get("GPUPGPARSER_TEST_MODE") == "1":
            print("\n【PostgreSQLテーブル行数との比較】")
            print("※ psycopg依存を削除したため、この機能は一時的に無効です")
            # TODO: Rustバイナリから行数情報を取得するか、別の方法を実装
            
            # 重複チェックは一時的に無効化（psycopg依存のため）
            if test_duplicate_keys:
                print("\n【重複チェック】")
                print("※ psycopg依存を削除したため、この機能は一時的に無効です")
        
        if actual_total_rows != results['total_rows']:
            print(f"\n⚠️  行数不一致: GPU報告値 {results['total_rows']:,} vs Parquet実際値 {actual_total_rows:,}")
        
        # テストモードの場合、メタデータファイルを保存
        if test_mode:
            import shutil
            
            # gpu_consumerで使用したディレクトリを取得（存在しない場合は新規作成）
            if hasattr(gpu_consumer, 'test_save_dir'):
                save_dir = gpu_consumer.test_save_dir
                print(f"\n📁 テストモード: メタデータファイルを保存中...")
            else:
                # gpu_consumerが実行されなかった場合の処理
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = f"test_binaries/{timestamp}"
                os.makedirs(save_dir, exist_ok=True)
                print(f"\n📁 テストモード: バイナリファイルを確認中...")
                
                # 現時点で存在するバイナリファイルを保存（通常はgpu_consumerで保存済み）
                saved_count = 0
                for i in range(total_chunks):
                    src = f"{OUTPUT_DIR}/{table_name}_chunk_{i}.bin"
                    if os.path.exists(src):
                        dst = f"{save_dir}/{table_name}_chunk_{i}.bin"
                        shutil.copy2(src, dst)
                        size = os.path.getsize(src) / (1024**3)
                        print(f"  ✓ {table_name}_chunk_{i}.bin を保存 ({size:.2f} GB)")
                        saved_count += 1
            
            # メタファイルを保存
            meta_src = f"{OUTPUT_DIR}/{table_name}_meta_0.json"
            if os.path.exists(meta_src):
                shutil.copy2(meta_src, f"{save_dir}/{table_name}_meta_0.json")
                print(f"  ✓ {table_name}_meta_0.json を保存")
            
            # 実行情報をメタデータファイルとして保存
            from datetime import datetime
            metadata = {
                "timestamp": save_dir.split('/')[-1],  # ディレクトリ名からタイムスタンプを取得
                "table_name": table_name,
                "total_chunks": total_chunks,
                "saved_chunks": total_chunks,  # gpu_consumerで全て保存されているはず
                "note": "Binary files saved before deletion in gpu_consumer",
                "parallel_connections": int(os.environ.get('RUST_PARALLEL_CONNECTIONS', 16)),
                "command": f"python cu_pg_parquet.py --test --table {table_name} --parallel {int(os.environ.get('RUST_PARALLEL_CONNECTIONS', 16))} --chunks {total_chunks}"
            }
            metadata_path = f"{save_dir}/execution_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ execution_metadata.json を保存")
            
            print(f"\n📁 バイナリファイルとメタデータを {save_dir} に保存しました")
        
        # 終了時のクリーンアップ
        cleanup_files(total_chunks, table_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PostgreSQL → GPU キューベース並列処理版ベンチマーク')
    parser.add_argument('--table', type=str, default='lineorder', help='対象テーブル名')
    parser.add_argument('--parallel', type=int, default=16, help='並列接続数')
    parser.add_argument('--chunks', type=int, default=8, help='チャンク数')
    parser.add_argument('--test', action='store_true', help='テストモード（GPU特性・カーネル情報表示）')
    args = parser.parse_args()
    
    # 環境変数を設定
    os.environ['RUST_PARALLEL_CONNECTIONS'] = str(args.parallel)
    os.environ['TOTAL_CHUNKS'] = str(args.chunks)
    os.environ['TABLE_NAME'] = args.table  # Rust側にもテーブル名を伝える
    
    # テストモードの場合は環境変数を設定
    if args.test:
        os.environ['GPUPGPARSER_TEST_MODE'] = '1'
    
    main(total_chunks=args.chunks, table_name=args.table, test_mode=args.test)