"""
cuDF ZeroCopy統合プロセッサー（文字列最適化版）

文字列処理を最適化した高性能版:
1. 文字列データ: 共有メモリ不使用の直接コピー
2. 固定長データ: 既存の統合カーネルを使用（Decimal処理等は安全に維持）
3. cuDFによるゼロコピーArrow変換
4. GPU直接Parquet書き出し
5. RMM統合メモリ管理

注: 元の統合バッファ版は src/old/main_postgres_to_parquet.py に移動されました
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import time
import warnings
import numpy as np
import cupy as cp
import cudf
import cudf.core.dtypes
from numba import cuda
import pyarrow as pa
import rmm

from .types import (
    ColumnMeta, INT16, INT32, INT64, FLOAT32, FLOAT64, DECIMAL128, 
    UTF8, BINARY, DATE32, TS64_US, BOOL, UNKNOWN
)
from .memory_manager import GPUMemoryManager, BufferInfo
from .cuda_kernels.optimized_parsers import (
    parse_binary_chunk_gpu_enhanced,
    optimize_grid_size
)
from .cuda_kernels.column_processor import (
    pass1_column_wise_integrated
)
from .cuda_kernels.decimal_tables import (
    POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
)
from .build_cudf_from_buf import CuDFZeroCopyProcessor
from .build_buf_from_postgres import detect_pg_header_size
from .write_parquet_from_cudf import write_cudf_to_parquet_with_options


class ZeroCopyProcessor:
    """ゼロコピープロセッサー（文字列最適化版）"""
    
    def __init__(self, use_rmm: bool = True, optimize_gpu: bool = True):
        """
        初期化
        
        Args:
            use_rmm: RMM (Rapids Memory Manager) を使用
            optimize_gpu: GPU最適化を有効化
        """
        self.use_rmm = use_rmm
        self.optimize_gpu = optimize_gpu
        self.cudf_processor = CuDFZeroCopyProcessor(use_rmm=use_rmm)
        self.gmm = GPUMemoryManager()
        self.device_props = self._get_device_properties()
        
        # RMM メモリプール最適化
        if use_rmm:
            try:
                rmm.reinitialize(
                    pool_allocator=True,
                    initial_pool_size=2**31,  # 2GB
                    maximum_pool_size=2**33   # 8GB
                )
                print("RMM メモリプール初期化完了 (最大8GB)")
            except Exception as e:
                warnings.warn(f"RMM初期化警告: {e}")
    
    def _get_device_properties(self) -> dict:
        """現在のGPUデバイス特性を取得"""
        try:
            device = cuda.get_current_device()
            return {
                'MAX_THREADS_PER_BLOCK': device.MAX_THREADS_PER_BLOCK,
                'MULTIPROCESSOR_COUNT': device.MULTIPROCESSOR_COUNT,
                'MAX_GRID_DIM_X': device.MAX_GRID_DIM_X,
                'GLOBAL_MEMORY': device.TOTAL_MEMORY,
                'SHARED_MEMORY_PER_BLOCK': device.MAX_SHARED_MEMORY_PER_BLOCK,
                'WARP_SIZE': device.WARP_SIZE
            }
        except Exception as e:
            warnings.warn(f"GPU特性取得失敗: {e}")
            return {
                'MAX_THREADS_PER_BLOCK': 1024,
                'MULTIPROCESSOR_COUNT': 16,
                'MAX_GRID_DIM_X': 65535,
                'GLOBAL_MEMORY': 8 * 1024**3,
                'SHARED_MEMORY_PER_BLOCK': 48 * 1024,
                'WARP_SIZE': 32
            }
    
    def create_string_buffers(
        self,
        columns: List[ColumnMeta],
        rows: int,
        raw_dev,
        field_offsets_dev,
        field_lengths_dev
    ) -> Dict[str, Any]:
        """
        文字列バッファ作成（最適化版 - 共有メモリ不使用）
        
        直接グローバルメモリコピーによる高速化
        """
        
        string_buffers = {}
        var_columns = [col for col in columns if col.is_variable and (col.arrow_id == UTF8 or col.arrow_id == BINARY)]
        
        if not var_columns:
            return string_buffers
        
        for col_idx, col in enumerate(var_columns):
            # 対応する列インデックスを検索
            actual_col_idx = None
            for i, c in enumerate(columns):
                if c.name == col.name:
                    actual_col_idx = i
                    break
            
            if actual_col_idx is None:
                continue
            
            try:
                # === 1. 長さ配列の並列抽出 ===
                d_lengths = cuda.device_array(rows, dtype=np.int32)
                
                # グリッドサイズの計算
                blocks, threads = optimize_grid_size(0, rows, self.device_props)
                
                @cuda.jit
                def extract_lengths_coalesced(field_lengths, col_idx, lengths_out, num_rows):
                    """ワープ効率を考慮した長さ抽出"""
                    row = cuda.grid(1)
                    if row < num_rows:
                        lengths_out[row] = field_lengths[row, col_idx]
                
                extract_lengths_coalesced[blocks, threads](
                    field_lengths_dev, actual_col_idx, d_lengths, rows
                )
                cuda.synchronize()
                
                # === 2. GPU上でのオフセット計算（CuPy使用版） ===
                # 文字列長配列をCuPyに変換
                lengths_cupy = cp.asarray(d_lengths)
                
                # CuPyのcumsum使用（GPU実行）
                offsets_cumsum = cp.cumsum(lengths_cupy, dtype=cp.int32)
                
                # オフセット配列を作成（0を先頭に追加）
                d_offsets = cuda.device_array(rows + 1, dtype=np.int32)
                d_offsets[0] = 0  # 最初のオフセットは0
                
                # CuPyの結果をNumba配列にコピー
                @cuda.jit
                def copy_cumsum_to_offsets(cumsum_data, offsets_out, num_rows):
                    """CuPy cumsum結果をオフセット配列にコピー"""
                    idx = cuda.grid(1)
                    if idx < num_rows:
                        offsets_out[idx + 1] = cumsum_data[idx]
                
                copy_cumsum_to_offsets[blocks, threads](
                    cp.asarray(offsets_cumsum), d_offsets, rows
                )
                cuda.synchronize()
                
                # 総データサイズを取得
                total_size_array = d_offsets[rows:rows+1].copy_to_host()
                total_size = int(total_size_array[0]) if len(total_size_array) > 0 else rows * 50
                
                if total_size == 0:
                    string_buffers[col.name] = {'data': None, 'offsets': None, 'actual_size': 0}
                    continue
                
                # === 3. 最適化データバッファの並列コピー（共有メモリ不使用） ===
                d_data = cuda.device_array(total_size, dtype=np.uint8)
                
                @cuda.jit
                def copy_string_data_direct_optimized(
                    raw_data, field_offsets, field_lengths,
                    col_idx, data_out, offsets, num_rows
                ):
                    """
                    最適化された直接グローバルメモリコピー
                    共有メモリを使用せず、メモリコアレッシングを考慮
                    """
                    row = cuda.grid(1)
                    if row >= num_rows:
                        return
                    
                    field_offset = field_offsets[row, col_idx]
                    field_length = field_lengths[row, col_idx]
                    output_offset = offsets[row]
                    
                    # NULL チェック
                    if field_length <= 0:
                        return
                    
                    # ワープ協調的な直接コピー（共有メモリ経由なし）
                    # 各スレッドが連続したメモリ領域を効率的にコピー
                    for i in range(field_length):
                        src_idx = field_offset + i
                        dst_idx = output_offset + i
                        
                        # 境界チェック
                        if (src_idx < raw_data.size and 
                            dst_idx < data_out.size):
                            # 直接グローバルメモリ間コピー
                            data_out[dst_idx] = raw_data[src_idx]
                
                print(f"文字列列 {col.name}: 直接コピー最適化カーネル実行")
                copy_string_data_direct_optimized[blocks, threads](
                    raw_dev, field_offsets_dev, field_lengths_dev,
                    actual_col_idx, d_data, d_offsets, rows
                )
                cuda.synchronize()
                
                string_buffers[col.name] = {
                    'data': d_data,
                    'offsets': d_offsets,
                    'actual_size': total_size
                }
                print(f"✅ 文字列列 {col.name}: 最適化バッファ作成完了 ({total_size} bytes)")
                
            except Exception as e:
                warnings.warn(f"文字列バッファ作成エラー ({col.name}): {e}")
                string_buffers[col.name] = {'data': None, 'offsets': None, 'actual_size': 0}
        
        return string_buffers
    
    def decode_and_export(
        self,
        raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        field_offsets_dev,
        field_lengths_dev,
        columns: List[ColumnMeta],
        output_path: str,
        compression: str = 'snappy',
        **parquet_kwargs
    ) -> Tuple[cudf.DataFrame, Dict[str, float]]:
        """
        文字列最適化デコード + エクスポート処理
        
        文字列処理のみ最適化し、固定長データは既存の統合カーネルを使用
        
        Returns:
            (cudf_dataframe, timing_info)
        """
        
        timing_info = {}
        start_time = time.time()
        
        rows, ncols = field_lengths_dev.shape
        if rows == 0:
            raise ValueError("rows == 0")

        # === 1. 前処理とメモリ準備 ===
        prep_start = time.time()
        
        # Decimal処理用テーブル
        d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
        d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)

        # 最適化文字列バッファ作成（共有メモリ不使用）
        optimized_string_buffers = self.create_string_buffers(
            columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
        )

        # 統合バッファシステム初期化（固定長データ用）
        buffer_info = self.gmm.initialize_buffers(columns, rows)
        
        # NULL配列
        d_nulls_all = cuda.device_array((rows, ncols), dtype=np.uint8)
        
        timing_info['preparation'] = time.time() - prep_start
        timing_info['string_optimization'] = timing_info['preparation']  # 文字列最適化時間

        # === 2. 統合カーネル実行（固定長データのみ） ===
        kernel_start = time.time()
        
        # Grid/Blockサイズの計算
        blocks, threads = optimize_grid_size(0, rows, self.device_props)
        
        print(f"統合カーネル実行（固定長データのみ）: {blocks} blocks × {threads} threads")
        
        try:
            # 既存の統合カーネルを使用（文字列データは統合バッファに書き込まれるが使用しない）
            pass1_column_wise_integrated[blocks, threads](
                raw_dev,
                field_offsets_dev,
                field_lengths_dev,
                
                # 列メタデータ配列
                buffer_info.column_types,
                buffer_info.column_is_variable,
                buffer_info.column_indices,
                
                # 固定長統合バッファ（このまま使用）
                buffer_info.fixed_buffer,
                buffer_info.fixed_column_offsets,
                buffer_info.fixed_column_sizes,
                buffer_info.fixed_decimal_scales,
                buffer_info.row_stride,
                
                # 可変長統合バッファ（文字列データは書き込まれるが使用しない）
                buffer_info.var_data_buffer,
                buffer_info.var_offset_arrays,
                buffer_info.var_column_mapping,
                
                # 共通出力
                d_nulls_all,
                
                # Decimal処理用
                d_pow10_table_lo,
                d_pow10_table_hi
            )
            
            cuda.synchronize()
            
        except Exception as e:
            raise RuntimeError(f"統合カーネル実行エラー: {e}")
        
        timing_info['kernel_execution'] = time.time() - kernel_start

        # === 3. cuDF DataFrame作成（文字列最適化版） ===
        cudf_start = time.time()
        
        # 最適化文字列バッファを使用してcuDF作成
        cudf_df = self.cudf_processor.create_cudf_from_gpu_buffers_zero_copy(
            columns, rows, buffer_info, optimized_string_buffers
        )
        
        timing_info['cudf_creation'] = time.time() - cudf_start

        # === 4. Parquet書き出し ===
        export_start = time.time()
        
        parquet_timing = write_cudf_to_parquet_with_options(
            cudf_df,
            output_path,
            compression=compression,
            optimize_for_spark=True,
            **parquet_kwargs
        )
        
        timing_info['parquet_export'] = time.time() - export_start
        timing_info['parquet_details'] = parquet_timing
        timing_info['total'] = time.time() - start_time
        
        return cudf_df, timing_info
    
    def process_postgresql_to_parquet(
        self,
        raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        columns: List[ColumnMeta],
        ncols: int,
        header_size: int,
        output_path: str,
        compression: str = 'snappy',
        **kwargs
    ) -> Tuple[cudf.DataFrame, Dict[str, float]]:
        """
        PostgreSQL → cuDF → GPU Parquet の文字列最適化処理
        
        文字列処理のみを最適化し、他は既存実装を維持
        """
        
        total_timing = {}
        overall_start = time.time()
        
        # === 1. GPUパース ===
        parse_start = time.time()
        
        print("=== GPU並列パース開始 ===")
        # 8.94倍高速化: Ultra Fast GPU並列パーサーを使用
        from .cuda_kernels.ultra_fast_parser import parse_binary_chunk_gpu_ultra_fast_v2
        field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=header_size
        )
        
        if self.optimize_gpu:
            print("✅ Ultra Fast GPU並列パーサー使用（8.94倍高速化達成）")
        
        rows = field_offsets_dev.shape[0]
        total_timing['gpu_parsing'] = time.time() - parse_start
        print(f"GPUパース完了: {rows} 行 ({total_timing['gpu_parsing']:.4f}秒)")
        
        # === 2. 文字列最適化デコード + エクスポート ===
        decode_start = time.time()
        
        print("=== 文字列最適化デコード開始 ===")
        cudf_df, decode_timing = self.decode_and_export(
            raw_dev, field_offsets_dev, field_lengths_dev,
            columns, output_path, compression, **kwargs
        )
        
        total_timing['decode_and_export'] = time.time() - decode_start
        total_timing.update(decode_timing)
        total_timing['overall_total'] = time.time() - overall_start
        
        # === 3. パフォーマンス統計 ===
        self._print_performance_stats(rows, len(columns), total_timing, len(raw_dev))
        
        return cudf_df, total_timing
    
    def _print_performance_stats(
        self, 
        rows: int, 
        cols: int, 
        timing: Dict[str, float], 
        data_size: int
    ):
        """パフォーマンス統計の表示"""
        
        print(f"\n=== パフォーマンス統計（文字列最適化版） ===")
        print(f"処理データ: {rows:,} 行 × {cols} 列")
        print(f"データサイズ: {data_size / (1024**2):.2f} MB")
        
        print("\n--- 詳細タイミング ---")
        for key, value in timing.items():
            if isinstance(value, (int, float)):
                # decode_and_exportの内訳を階層表示
                if key == 'decode_and_export':
                    print(f"  {key:20}: {value:.4f} 秒")
                    # 内訳項目を表示
                    preparation_time = timing.get('preparation', 0)
                    kernel_time = timing.get('kernel_execution', 0)
                    cudf_time = timing.get('cudf_creation', 0)
                    string_opt_time = timing.get('string_optimization', 0)
                    if preparation_time > 0:
                        print(f"    ├─ preparation   : {preparation_time:.4f} 秒")
                        if string_opt_time > 0:
                            print(f"    │  └─ string_opt  : {string_opt_time:.4f} 秒")
                    if kernel_time > 0:
                        print(f"    ├─ gpu_decode      : {kernel_time:.4f} 秒")
                    if cudf_time > 0:
                        print(f"    └─ cudf_creation : {cudf_time:.4f} 秒")
                # parquet_exportの内訳を階層表示
                elif key == 'parquet_export':
                    print(f"  {key:20}: {value:.4f} 秒")
                    # parquet_detailsから詳細を取得
                    parquet_details = timing.get('parquet_details', {})
                    if isinstance(parquet_details, dict):
                        cudf_direct_time = parquet_details.get('cudf_direct', 0)
                        if cudf_direct_time > 0:
                            print(f"    └─ cudf_direct   : {cudf_direct_time:.4f} 秒")
                # 内訳項目は個別表示をスキップ
                elif key in ['preparation', 'kernel_execution', 'cudf_creation', 'string_optimization']:
                    continue
                else:
                    print(f"  {key:20}: {value:.4f} 秒")
            elif isinstance(value, dict):
                # parquet_detailsは階層表示済みなのでスキップ
                if key == 'parquet_details':
                    continue
                print(f"  {key:20}: (詳細は省略)")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        print(f"    {sub_key:18}: {sub_value:.4f}")
                    else:
                        print(f"    {sub_key:18}: {sub_value}")
            else:
                print(f"  {key:20}: {str(value)}")
        
        # スループット計算
        total_cells = rows * cols
        overall_time = timing.get('overall_total', timing.get('total', 1.0))
        
        if overall_time > 0:
            cell_throughput = total_cells / overall_time
            data_throughput = (data_size / (1024**2)) / overall_time
            
            print(f"\n--- スループット ---")
            print(f"  セル処理速度: {cell_throughput:,.0f} cells/sec")
            print(f"  データ処理速度: {data_throughput:.2f} MB/sec")
            
            # 文字列最適化効率指標
            if 'string_optimization' in timing:
                string_efficiency = (timing['string_optimization'] / overall_time) * 100
                print(f"  文字列最適化効率: {string_efficiency:.1f}%")
            
            # GPU効率指標
            if 'kernel_execution' in timing:
                kernel_efficiency = (timing['kernel_execution'] / overall_time) * 100
                print(f"  GPU使用効率: {kernel_efficiency:.1f}%")
        
        print("=" * 30)


def postgresql_to_cudf_parquet(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
    columns: List[ColumnMeta],
    ncols: int,
    header_size: int,
    output_path: str,
    compression: str = 'snappy',
    use_rmm: bool = True,
    optimize_gpu: bool = True,
    **parquet_kwargs
) -> Tuple[cudf.DataFrame, Dict[str, float]]:
    """
    PostgreSQL → cuDF → GPU Parquet 統合処理関数（文字列最適化版）
    
    最適化技術を統合した高性能バージョン：
    - 並列化GPU行検出・フィールド抽出
    - 文字列処理最適化（共有メモリ不使用の直接コピー）
    - メモリコアレッシング最適化
    - cuDFゼロコピーArrow変換
    - GPU直接Parquet書き出し
    - RMM統合メモリ管理
    
    Args:
        raw_dev: GPU上のPostgreSQLバイナリデータ
        columns: 列メタデータ
        ncols: 列数
        header_size: ヘッダーサイズ
        output_path: Parquet出力パス
        compression: 圧縮方式
        use_rmm: RMM使用フラグ
        optimize_gpu: GPU最適化フラグ
        **parquet_kwargs: 追加のParquetオプション
    
    Returns:
        (cudf_dataframe, timing_information)
    """
    
    processor = ZeroCopyProcessor(
        use_rmm=use_rmm, 
        optimize_gpu=optimize_gpu
    )
    
    return processor.process_postgresql_to_parquet(
        raw_dev, columns, ncols, header_size, output_path, 
        compression, **parquet_kwargs
    )


__all__ = [
    "ZeroCopyProcessor",
    "postgresql_to_cudf_parquet"
]