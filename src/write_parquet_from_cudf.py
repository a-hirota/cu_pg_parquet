"""
cuDF DataFrame から Parquet ファイルへの書き込み処理

GPU上のcuDF DataFrameを効率的にParquetファイルとして保存します。
"""

import warnings
import pyarrow.parquet as pq
import cudf
from typing import Dict, Any, Optional


def write_cudf_to_parquet(
    cudf_df: cudf.DataFrame,
    output_path: str,
    compression: str = 'snappy',
    **parquet_kwargs
) -> Dict[str, float]:
    """
    cuDF DataFrameをParquetファイルに書き込み
    
    Args:
        cudf_df: 書き込み対象のcuDF DataFrame
        output_path: 出力Parquetファイルパス
        compression: 圧縮方式 ('snappy', 'gzip', 'lz4', etc.)
        **parquet_kwargs: 追加のParquetオプション
    
    Returns:
        Dict[str, float]: 書き込み時間の詳細情報
    """
    
    import time
    timing_info = {}
    start_time = time.time()
    
    try:
        # cuDFの直接Parquet書き出し（GPU上で圧縮処理）
        cudf_df.to_parquet(
            output_path,
            compression=compression,
            engine='cudf',  # cuDFエンジン使用
            **parquet_kwargs
        )
        
        timing_info['cudf_direct'] = time.time() - start_time
        timing_info['method'] = 'cudf_direct'
        
    except Exception as e:
        warnings.warn(f"cuDF直接書き出し失敗, PyArrowにフォールバック: {e}")
        
        # フォールバック: PyArrow経由
        fallback_start = time.time()
        arrow_table = cudf_df.to_arrow()
        pq.write_table(arrow_table, output_path, compression=compression)
        
        timing_info['cudf_direct'] = time.time() - start_time  # 失敗までの時間
        timing_info['pyarrow_fallback'] = time.time() - fallback_start
        timing_info['total'] = time.time() - start_time
        timing_info['method'] = 'pyarrow_fallback'
    
    timing_info['total'] = time.time() - start_time
    
    return timing_info


def write_cudf_to_parquet_with_options(
    cudf_df: cudf.DataFrame,
    output_path: str,
    compression: str = 'snappy',
    optimize_for_spark: bool = False,
    row_group_size: Optional[int] = None,
    **parquet_kwargs
) -> Dict[str, float]:
    """
    オプション付きcuDF → Parquet書き込み
    
    Args:
        cudf_df: 書き込み対象のcuDF DataFrame
        output_path: 出力Parquetファイルパス
        compression: 圧縮方式
        optimize_for_spark: Spark読み込み用設定
        row_group_size: 行グループサイズ（None=自動）
        **parquet_kwargs: 追加のParquetオプション
    
    Returns:
        Dict[str, float]: 詳細なタイミング情報
    """
    
    import time
    timing_info = {}
    start_time = time.time()
    
    # オプションの設定
    enhanced_kwargs = parquet_kwargs.copy()
    
    if optimize_for_spark:
        # Spark読み込み用設定（cuDFエンジン対応）
        enhanced_kwargs.update({
            'use_dictionary': True
            # 注: write_statisticsはcuDFエンジンでサポートされていないため除外
        })
    
    if row_group_size is not None:
        enhanced_kwargs['row_group_size'] = row_group_size
    
    # GPU書き込み実行
    try:
        cudf_df.to_parquet(
            output_path,
            compression=compression,
            engine='cudf',
            **enhanced_kwargs
        )
        
        timing_info['method'] = 'cudf_enhanced'
        
    except Exception as e:
        warnings.warn(f"拡張書き込み失敗, 標準方式にフォールバック: {e}")
        
        # 標準方式でリトライ
        fallback_result = write_cudf_to_parquet(
            cudf_df, output_path, compression, **parquet_kwargs
        )
        timing_info.update(fallback_result)
        timing_info['enhancement_failed'] = True
    
    timing_info['total'] = time.time() - start_time
    
    return timing_info


__all__ = [
    "write_cudf_to_parquet",
    "write_cudf_to_parquet_with_options"
]
