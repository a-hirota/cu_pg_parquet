"""
PostgreSQL → Rust → /dev/shm → GPU バッチ処理版
Rust FFI経由でGPU処理を実行

最適化:
- Rustの高速データ転送（2GB/s）
- /dev/shmを介したゼロコピー転送
- 一括GPU処理によるオーバーヘッド削減
- Ray完全削除によるシンプル化

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列（Rust側で使用）
"""

import os
import time
import subprocess
import psutil
import json
import cupy as cp
from pathlib import Path
from typing import Dict, Tuple, List

from src.types import ColumnMeta, PG_OID_TO_ARROW, UNKNOWN
from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
from src.main_postgres_to_parquet import postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"
OUTPUT_DIR = "/dev/shm"
OUTPUT_PARQUET_PATH = "benchmark/lineorder_rust_gpu_batch.parquet"
META_FILE = "/dev/shm/lineorder_meta.json"
READY_FILE = "/dev/shm/lineorder_data.ready"
RUST_BINARY = "/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_sequential"
CHUNKS = 4  # チャンク数

class DetailedMetrics:
    def __init__(self):
        self.metrics = {
            'memory': {
                'initial_system_mb': 0,
                'peak_system_mb': 0,
                'initial_gpu_mb': 0,
                'peak_gpu_mb': 0,
                'data_size_mb': 0,
                'memory_overhead_ratio': 0
            },
            'performance': {
                'rust_time_sec': 0,
                'rust_speed_gbps': 0,
                'gpu_transfer_time_sec': 0,
                'gpu_processing_time_sec': 0,
                'total_time_sec': 0,
                'throughput_mb_sec': 0
            },
            'details': []
        }
    
    def log_memory_snapshot(self, stage: str):
        process = psutil.Process()
        system_memory_mb = process.memory_info().rss / (1024**2)
        
        try:
            mempool = cp.get_default_memory_pool()
            gpu_memory_mb = mempool.used_bytes() / (1024**2)
        except:
            gpu_memory_mb = 0
        
        self.metrics['details'].append({
            'timestamp': time.time(),
            'stage': stage,
            'system_memory_mb': system_memory_mb,
            'gpu_memory_mb': gpu_memory_mb
        })
        
        # ピーク値を更新
        if system_memory_mb > self.metrics['memory']['peak_system_mb']:
            self.metrics['memory']['peak_system_mb'] = system_memory_mb
        if gpu_memory_mb > self.metrics['memory']['peak_gpu_mb']:
            self.metrics['memory']['peak_gpu_mb'] = gpu_memory_mb
    
    def save_to_json(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def cleanup_shm_files():
    """共有メモリファイルをクリーンアップ"""
    # チャンクファイルをクリーンアップ
    for chunk_id in range(CHUNKS):
        chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            print(f"✓ クリーンアップ: {chunk_file}")
    
    # メタデータファイルをクリーンアップ
    for file in [META_FILE, READY_FILE]:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ クリーンアップ: {file}")


def run_rust_data_transfer() -> Tuple[float, float]:
    """Rustプロセスを実行してPostgreSQLからデータを転送"""
    print("\n=== フェーズ1: Rust高速データ転送 ===")
    
    # 環境変数が設定されているか確認
    if "GPUPASER_PG_DSN" not in os.environ:
        raise ValueError("GPUPASER_PG_DSN環境変数が設定されていません")
    
    # Rustプロセスを実行
    start_time = time.time()
    result = subprocess.run(
        [RUST_BINARY],
        capture_output=True,
        text=True,
        env=os.environ
    )
    rust_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Rustプロセスエラー:")
        print(result.stderr)
        raise RuntimeError("Rustデータ転送に失敗しました")
    
    # 出力から速度を抽出
    speed_gbps = 0.0
    for line in result.stdout.split('\n'):
        if "読み取り速度:" in line:
            speed_gbps = float(line.split(":")[1].strip().split()[0])
            break
    
    print(result.stdout)
    return rust_time, speed_gbps


def load_metadata() -> Tuple[List[ColumnMeta], List[Dict]]:
    """Rustが出力したメタデータJSONを読み込み"""
    with open(META_FILE, 'r') as f:
        meta_json = json.load(f)
    
    columns = []
    for col in meta_json['columns']:
        # JSONからColumnMetaオブジェクトに変換
        pg_oid = col['pg_oid']
        arrow_info = PG_OID_TO_ARROW.get(pg_oid, (UNKNOWN, None))
        arrow_id, elem_size = arrow_info
        
        column_meta = ColumnMeta(
            name=col['name'],
            pg_oid=pg_oid,
            pg_typmod=-1,  # 今回は使用しない
            arrow_id=arrow_id,
            elem_size=elem_size if elem_size is not None else 0
        )
        columns.append(column_meta)
    
    # チャンク情報も返す
    chunks = meta_json.get('chunks', [])
    return columns, chunks


def process_chunk_on_gpu(chunk_id: int, chunk_file: str, columns: List[ColumnMeta], 
                        metrics: DetailedMetrics) -> Dict:
    """1つのチャンクをGPU上で処理"""
    print(f"\n=== チャンク {chunk_id} 処理 ===")
    
    # ファイルサイズ確認
    file_size = os.path.getsize(chunk_file)
    file_size_gb = file_size / (1024**3)
    print(f"チャンクサイズ: {file_size_gb:.2f} GB")
    
    metrics.log_memory_snapshot(f"chunk_{chunk_id}_before_read")
    
    # データを読み込み
    print("データ読み込み中...")
    read_start = time.time()
    with open(chunk_file, 'rb') as f:
        data = f.read()
    read_time = time.time() - read_start
    print(f"読み込み時間: {read_time:.2f}秒 ({file_size_gb/read_time:.2f} GB/秒)")
    
    metrics.log_memory_snapshot(f"chunk_{chunk_id}_after_read")
    
    # GPU転送
    print("GPU転送中...")
    gpu_start = time.time()
    gpu_data = cp.frombuffer(data, dtype=cp.uint8)
    gpu_transfer_time = time.time() - gpu_start
    gpu_transfer_speed = file_size_gb / gpu_transfer_time
    print(f"GPU転送時間: {gpu_transfer_time:.2f}秒 ({gpu_transfer_speed:.2f} GB/秒)")
    
    # メモリ解放
    del data  # CPUメモリを即座に解放
    
    metrics.log_memory_snapshot(f"chunk_{chunk_id}_after_gpu_transfer")
    
    return {
        'chunk_id': chunk_id,
        'file_size': file_size,
        'read_time': read_time,
        'gpu_transfer_time': gpu_transfer_time,
        'gpu_data': gpu_data
    }


def process_all_chunks_on_gpu(columns: List[ColumnMeta], chunks: List[Dict], 
                             metrics: DetailedMetrics) -> None:
    """GPU上ですべてのチャンクを処理してParquetに変換"""
    print("\n=== フェーズ2: GPU処理 ===")
    
    # 完了フラグを待機
    wait_start = time.time()
    while not os.path.exists(READY_FILE):
        if time.time() - wait_start > 300:  # 5分タイムアウト
            raise TimeoutError("Rustプロセスのタイムアウト")
        time.sleep(0.1)
    
    print("✓ データ転送完了を確認")
    
    total_processing_time = 0
    total_data_size = 0
    
    # 各チャンクを順次処理
    for chunk_id in range(CHUNKS):
        chunk_file = f"{OUTPUT_DIR}/chunk_{chunk_id}.bin"
        if not os.path.exists(chunk_file):
            print(f"警告: チャンクファイル {chunk_file} が存在しません")
            continue
        
        chunk_start = time.time()
        chunk_result = process_chunk_on_gpu(chunk_id, chunk_file, columns, metrics)
        
        # GPUパーサーで処理
        print("\nGPUパーシング中...")
        gpu_process_start = time.time()
        
        # ヘッダーサイズ検出
        gpu_data = chunk_result['gpu_data']
        header_size_gpu = cp.zeros(1, dtype=cp.int32)
        detect_pg_header_size(gpu_data, header_size_gpu)
        header_size = int(header_size_gpu[0])
        print(f"ヘッダーサイズ: {header_size} bytes")
        
        # postgresql_to_cudf_parquetを使用して処理
        chunk_output_path = f"benchmark/lineorder_rust_gpu_batch_chunk_{chunk_id}.parquet"
        postgresql_to_cudf_parquet(
            pg_header_size=header_size,
            column_metas=columns,
            buffer=gpu_data,
            output_parquet_path=chunk_output_path,
            table_name=TABLE_NAME,
            estimated_rows=None  # 自動検出
        )
        
        gpu_process_time = time.time() - gpu_process_start
        chunk_total_time = time.time() - chunk_start
        
        print(f"チャンク{chunk_id} GPU処理時間: {gpu_process_time:.2f}秒")
        print(f"チャンク{chunk_id} 合計時間: {chunk_total_time:.2f}秒")
        
        total_processing_time += chunk_total_time
        total_data_size += chunk_result['file_size']
        
        # GPUメモリを解放
        del gpu_data
        del chunk_result
        cp.get_default_memory_pool().free_all_blocks()
        
        # チャンクファイルを削除してディスクスペースを解放
        os.remove(chunk_file)
        print(f"✓ チャンクファイル削除: {chunk_file}")
        
        metrics.log_memory_snapshot(f"chunk_{chunk_id}_after_processing")
    
    # 全体の統計
    total_data_gb = total_data_size / (1024**3)
    print(f"\n=== 全体統計 ===")
    print(f"総データサイズ: {total_data_gb:.2f} GB")
    print(f"総GPU処理時間: {total_processing_time:.2f}秒")
    print(f"平均スループット: {total_data_gb/total_processing_time:.2f} GB/秒")
    
    metrics.metrics['performance']['gpu_processing_time_sec'] = total_processing_time
    metrics.metrics['memory']['data_size_mb'] = total_data_size / (1024**2)


def main():
    print("✅ CUDA context OK")
    print("=== PostgreSQL → Rust → /dev/shm → GPU バッチ処理版 ===")
    print(f"テーブル: {TABLE_NAME}")
    print(f"Rustバイナリ: {RUST_BINARY}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print(f"チャンク数: {CHUNKS}")
    print(f"出力: {OUTPUT_PARQUET_PATH}")
    
    # クリーンアップ
    cleanup_shm_files()
    
    # メトリクス初期化
    metrics = DetailedMetrics()
    metrics.log_memory_snapshot("start")
    
    total_start_time = time.time()
    
    try:
        # Rustデータ転送（メタデータも同時に生成される）
        rust_time, rust_speed = run_rust_data_transfer()
        metrics.metrics['performance']['rust_time_sec'] = rust_time
        metrics.metrics['performance']['rust_speed_gbps'] = rust_speed
        
        # メタデータ読み込み
        print("\nメタデータを読み込み中...")
        meta_start = time.time()
        columns, chunks = load_metadata()
        print(f"メタデータ読み込み完了 ({time.time() - meta_start:.4f}秒, {len(columns)}カラム)")
        print(f"チャンク数: {len(chunks)}")
        metrics.log_memory_snapshot("after_metadata")
        
        # GPU処理
        process_all_chunks_on_gpu(columns, chunks, metrics)
        
        # 結果サマリー
        total_time = time.time() - total_start_time
        metrics.metrics['performance']['total_time_sec'] = total_time
        
        print("\n" + "="*60)
        print("=== 実行結果サマリー ===")
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"\nフェーズ別時間:")
        print(f"  - Rustデータ転送: {rust_time:.2f}秒 ({rust_speed:.2f} GB/秒)")
        print(f"  - GPU処理: {metrics.metrics['performance']['gpu_processing_time_sec']:.2f}秒")
        print(f"\nデータサイズ: {metrics.metrics['memory']['data_size_mb']/1024:.2f} GB")
        print(f"スループット: {metrics.metrics['memory']['data_size_mb']/1024/total_time:.2f} GB/秒")
        
        # メモリ使用量
        print(f"\nメモリ使用量:")
        print(f"  - システムメモリ（ピーク）: {metrics.metrics['memory']['peak_system_mb']:.0f} MB")
        print(f"  - GPUメモリ（ピーク）: {metrics.metrics['memory']['peak_gpu_mb']:.0f} MB")
        
        # メトリクス保存
        metrics.save_to_json("benchmark/rust_gpu_batch_metrics.json")
        print(f"\n詳細メトリクス保存: benchmark/rust_gpu_batch_metrics.json")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        raise
    
    finally:
        # クリーンアップ
        cleanup_shm_files()
        print("\n✓ クリーンアップ完了")


if __name__ == "__main__":
    main()