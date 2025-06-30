"""
GPUメモリ制限に基づいて最適なチャンクサイズを自動計算するユーティリティ
"""
import os
import cupy as cp
import psycopg
from ..cuda_kernels.postgres_binary_parser import estimate_row_size_from_columns


def get_gpu_memory_info():
    """GPUメモリ情報を取得"""
    gpu_props = cp.cuda.runtime.getDeviceProperties(0)
    total_memory_gb = gpu_props['totalGlobalMem'] / 1024**3
    
    # メモリプールの使用状況
    mempool = cp.get_default_memory_pool()
    used_memory_gb = mempool.used_bytes() / 1024**3
    available_memory_gb = total_memory_gb - used_memory_gb
    
    return {
        'total_gb': total_memory_gb,
        'used_gb': used_memory_gb,
        'available_gb': available_memory_gb
    }


def get_table_size(table_name):
    """PostgreSQLテーブルのサイズを取得"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise ValueError("GPUPASER_PG_DSN環境変数が設定されていません")
    
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT pg_relation_size('{table_name}'::regclass)")
            return cursor.fetchone()[0]


def calculate_optimal_chunks(table_name, columns):
    """
    テーブルとGPUメモリに基づいて最適なチャンク数を計算
    
    Args:
        table_name: テーブル名
        columns: カラムメタデータのリスト
    
    Returns:
        (chunk_size_bytes, num_chunks)
    """
    # テーブルサイズを取得
    total_data_size = get_table_size(table_name)
    
    # GPUメモリ情報を取得
    gpu_info = get_gpu_memory_info()
    
    # 利用可能メモリの50%を使用（より安全なマージン）
    available_memory = gpu_info['available_gb'] * 1024**3 * 0.5
    
    # 推定行サイズを計算
    estimated_row_size = estimate_row_size_from_columns(columns)
    
    # メタデータのメモリ使用量
    ncols = len(columns)
    metadata_per_row = 8 + ncols * 12  # row_positions + field_offsets + field_lengths
    
    # 処理時の追加メモリを考慮（3倍のマージン - ソート、文字列処理、DataFrame作成）
    total_memory_per_row = (estimated_row_size + metadata_per_row) * 3
    
    # 最大処理可能行数を計算
    max_rows_per_chunk = int(available_memory / total_memory_per_row)
    
    # チャンクサイズを計算
    max_chunk_size = max_rows_per_chunk * estimated_row_size
    
    # チャンクサイズの制限（1GB～4GB）- より保守的な上限
    min_chunk_size = 1 * 1024**3
    max_chunk_size_limit = 4 * 1024**3
    
    if max_chunk_size < min_chunk_size:
        # GPUメモリが少ない場合
        chunk_size = max_chunk_size
        print(f"警告: GPUメモリが少ないため、小さなチャンクサイズ（{chunk_size/1024**3:.2f}GB）を使用します")
    else:
        chunk_size = min(max_chunk_size, max_chunk_size_limit)
        chunk_size = max(chunk_size, min_chunk_size)
    
    # チャンク数を計算
    num_chunks = max(1, int((total_data_size + chunk_size - 1) / chunk_size))
    
    # 並列度を考慮したチャンク数の調整
    # Rust側の16並列接続に適した数に調整
    # 大規模テーブル用により多くのチャンクを推奨
    if total_data_size > 30 * 1024**3:  # 30GB以上
        if num_chunks < 16:
            num_chunks = 16  # 最小16チャンク
        elif num_chunks > 32:
            num_chunks = 32  # 最大32チャンク
    elif total_data_size > 10 * 1024**3:  # 10GB以上
        if num_chunks < 8:
            num_chunks = 8
        elif num_chunks > 16:
            num_chunks = 16
    else:
        # 小規模テーブル（10GB未満）
        if num_chunks > 8:
            num_chunks = 8
        elif num_chunks < 1:
            num_chunks = 1
    
    # 実際のチャンクサイズを再計算
    actual_chunk_size = total_data_size / num_chunks
    
    # 情報を出力
    print(f"\n=== チャンク構成の自動計算 ===")
    print(f"テーブル名: {table_name}")
    print(f"テーブルサイズ: {total_data_size / 1024**3:.2f} GB")
    print(f"GPU総メモリ: {gpu_info['total_gb']:.2f} GB")
    print(f"GPU利用可能メモリ: {gpu_info['available_gb']:.2f} GB")
    print(f"推定行サイズ: {estimated_row_size} バイト")
    print(f"行あたり総メモリ: {total_memory_per_row} バイト")
    print(f"推奨チャンク数: {num_chunks}")
    print(f"チャンクサイズ: {actual_chunk_size / 1024**3:.2f} GB/チャンク")
    print(f"最大処理可能行数/チャンク: {int(actual_chunk_size / estimated_row_size):,}")
    print("=" * 40)
    
    return int(actual_chunk_size), num_chunks


def get_chunk_recommendation(table_name, columns):
    """
    チャンク数の推奨値を取得（環境変数でオーバーライド可能）
    
    Returns:
        num_chunks: 推奨チャンク数
    """
    # 環境変数でオーバーライド
    manual_chunks = os.environ.get("GPUPGPARSER_CHUNKS")
    if manual_chunks:
        return int(manual_chunks)
    
    # 自動計算
    _, num_chunks = calculate_optimal_chunks(table_name, columns)
    return num_chunks