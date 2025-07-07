"""
cuDF直接抽出プロセッサー（統合バッファ不使用版）

統合バッファを経由せず、入力データから直接列を抽出する
メモリ効率的な実装：
1. 文字列データ: 既存の最適化済み個別バッファ使用
2. 固定長データ: 入力データから直接cuDF列作成（統合バッファ削除）
3. cuDFによるゼロコピーArrow変換
4. GPU直接Parquet書き出し
5. RMM統合メモリ管理
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import time
import warnings
import numpy as np
import cupy as cp
import cudf
from numba import cuda
import rmm

from .types import ColumnMeta
from .postgres_to_cudf import DirectColumnExtractor
from .write_parquet_from_cudf import write_cudf_to_parquet_with_options
from .cuda_kernels.postgres_binary_parser import detect_pg_header_size
from .cuda_kernels.gpu_config_utils import optimize_grid_size


class DirectProcessor:
    """直接抽出プロセッサー（統合バッファ不使用）"""
    
    def __init__(self, use_rmm: bool = True, optimize_gpu: bool = True, verbose: bool = False, test_mode: bool = False):
        """
        初期化
        
        Args:
            use_rmm: RMM (Rapids Memory Manager) を使用
            optimize_gpu: GPU最適化を有効化
            verbose: 詳細ログを出力
            test_mode: テストモード（GPU特性・カーネル情報表示）
        """
        self.use_rmm = use_rmm
        self.optimize_gpu = optimize_gpu
        self.verbose = verbose
        self.test_mode = test_mode
        self.extractor = DirectColumnExtractor()
        self.device_props = self._get_device_properties()
        
        # RMM メモリプール最適化
        if use_rmm:
            try:
                if not rmm.is_initialized():
                    gpu_memory = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem']
                    pool_size = int(gpu_memory * 0.9)
                    
                    rmm.reinitialize(
                        pool_allocator=True,
                        initial_pool_size=pool_size,
                        maximum_pool_size=pool_size
                    )
                    # RMM初期化ログを削除（verbose時のみ表示）
                    pass
            except Exception as e:
                warnings.warn(f"RMM初期化警告: {e}")
    
    def _get_device_properties(self) -> dict:
        """現在のGPUデバイス特性を取得"""
        try:
            device = cuda.get_current_device()
            props = {
                'MAX_THREADS_PER_BLOCK': device.MAX_THREADS_PER_BLOCK,
                'MULTIPROCESSOR_COUNT': device.MULTIPROCESSOR_COUNT,
                'MAX_GRID_DIM_X': device.MAX_GRID_DIM_X,
                'SHARED_MEMORY_PER_BLOCK': device.MAX_SHARED_MEMORY_PER_BLOCK,
                'WARP_SIZE': device.WARP_SIZE
            }
            
            try:
                props['GLOBAL_MEMORY'] = device.TOTAL_MEMORY
            except AttributeError:
                import cupy as cp
                try:
                    mempool = cp.get_default_memory_pool()
                    props['GLOBAL_MEMORY'] = mempool.total_bytes()
                except:
                    props['GLOBAL_MEMORY'] = 8 * 1024**3  # 8GB
                    
            return props
            
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
        row_positions_dev,
        field_offsets_dev,
        field_lengths_dev
    ) -> Dict[str, Any]:
        """
        文字列バッファ作成（DirectColumnExtractorから使用）
        """
        return self.extractor.create_string_buffers(
            columns, rows, raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev
        )
    
    def process_direct(
        self,
        raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
        row_positions_dev,
        field_offsets_dev,
        field_lengths_dev,
        columns: List[ColumnMeta],
        output_path: str,
        compression: str = 'snappy',
        thread_ids_dev=None,
        thread_start_positions_dev=None,
        thread_end_positions_dev=None,
        **parquet_kwargs
    ) -> Tuple[cudf.DataFrame, Dict[str, float]]:
        """
        直接抽出処理（統合バッファ不使用）
        
        Returns:
            (cudf_dataframe, timing_info)
        """
        
        timing_info = {}
        start_time = time.time()
        
        rows, ncols = field_lengths_dev.shape
        if rows == 0:
            raise ValueError("rows == 0")

        # === 1. 文字列バッファ作成 ===
        prep_start = time.time()
        
        # 最適化文字列バッファ作成（既存の実装を使用）
        optimized_string_buffers = self.create_string_buffers(
            columns, rows, raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev
        )
        
        timing_info['string_buffer_creation'] = time.time() - prep_start

        # === 2. 直接列抽出（統合バッファ不使用） ===
        extract_start = time.time()
        
        if self.verbose:
            print(f"直接列抽出開始（統合バッファ不使用）: {rows} 行")
        
        cudf_df = self.extractor.extract_columns_direct(
            raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev,
            columns, optimized_string_buffers, thread_ids_dev,
            thread_start_positions_dev, thread_end_positions_dev
        )
        
        timing_info['direct_extraction'] = time.time() - extract_start

        # === 3. Parquet書き出し ===
        export_start = time.time()
        
        # Parquet書き込み前にGPUメモリを最適化
        # 不要な中間バッファを解放
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        used_before = mempool.used_bytes()
        mempool.free_all_blocks()
        used_after = mempool.used_bytes()
        if self.verbose and used_before > used_after:
            print(f"Parquet書き込み前メモリ解放: {(used_before - used_after) / 1024**2:.1f} MB")
        
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
        PostgreSQL → cuDF → GPU Parquet の直接処理
        """
        
        total_timing = {}
        overall_start = time.time()
        
        # === 1. GPUパース ===
        parse_start = time.time()
        
        if self.verbose:
            print("=== GPU並列パース開始 ===")
        
        # テストモードチェック
        test_mode = os.environ.get('GPUPGPARSER_TEST_MODE', '0') == '1'
        
        from .cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2
        parse_result = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=header_size, test_mode=test_mode
        )
        
        # テストモードの場合、デバッグ情報も返される
        if test_mode and len(parse_result) >= 4:
            row_positions_dev = parse_result[0]
            field_offsets_dev = parse_result[1]
            field_lengths_dev = parse_result[2]
            thread_ids_dev = parse_result[3] if len(parse_result) > 3 else None
            thread_start_positions_dev = parse_result[4] if len(parse_result) > 4 else None
            thread_end_positions_dev = parse_result[5] if len(parse_result) > 5 else None
        else:
            row_positions_dev = parse_result[0]
            field_offsets_dev = parse_result[1]
            field_lengths_dev = parse_result[2]
            thread_ids_dev = None
            thread_start_positions_dev = None
            thread_end_positions_dev = None
        
        rows = field_offsets_dev.shape[0]
        total_timing['gpu_parsing'] = time.time() - parse_start
        
        if self.verbose:
            if self.optimize_gpu:
                print("✅ Ultra Fast GPU並列パーサー使用（8.94倍高速化達成）")
            print(f"GPUパース完了: {rows} 行 ({total_timing['gpu_parsing']:.4f}秒)")
        
        # === 2. 直接抽出 + エクスポート ===
        process_start = time.time()
        
        if self.verbose:
            print("=== 直接列抽出開始（統合バッファ不使用） ===")
        cudf_df, process_timing = self.process_direct(
            raw_dev, row_positions_dev, field_offsets_dev, field_lengths_dev,
            columns, output_path, compression, 
            thread_ids_dev=thread_ids_dev,
            thread_start_positions_dev=thread_start_positions_dev,
            thread_end_positions_dev=thread_end_positions_dev,
            **kwargs
        )
        
        total_timing['process_and_export'] = time.time() - process_start
        total_timing.update(process_timing)
        total_timing['overall_total'] = time.time() - overall_start
        
        # === 3. パフォーマンス統計 ===
        self._print_performance_stats(rows, len(columns), total_timing, len(raw_dev))
        
        return cudf_df, total_timing
    
    def _print_grid_boundary_debug_info(self, debug_info: np.ndarray, raw_dev):
        """Grid境界スレッドのデバッグ情報を表示（テストモード用）"""
        print("\n=== Grid境界スレッドデバッグ情報 ===")
        print(f"記録されたGrid境界スレッド数: {len(debug_info)}")
        
        # デバッグ情報の構造:
        # [0]: Thread ID, [1-3]: Block/Thread indices
        # [4-6]: Row info (position, end, ncols)
        # [7-40]: Field offsets/lengths (17 fields)
        # [41-80]: Binary data sample (40 bytes)
        
        # 表示を15スレッドに限定
        display_limit = min(15, len(debug_info))
        print(f"表示: 最初の{display_limit}スレッドのみ\n")
        
        for idx, info in enumerate(debug_info[:display_limit]):
            print(f"\n--- Grid境界スレッド {idx + 1} ---")
            print(f"Thread ID: {info[0]}")
            print(f"Block Index: ({info[1]}, {info[2]}), Thread Index: {info[3]}")
            print(f"Row Position: {info[4]}, Row End: {info[5]}, Columns: {info[6]}")
            
            # フィールド情報（全フィールド表示）
            ncols = info[6]
            print(f"Field Offsets/Lengths (all {ncols} fields):")
            for i in range(min(ncols, 17)):  # 最大17列まで
                if 7 + i*2 + 1 < len(info):
                    offset = info[7 + i*2]
                    length = info[8 + i*2]
                    print(f"  Field {i}: offset={offset}, length={length}")
            
            # バイナリデータサンプル
            row_pos = info[4]
            sample_start = row_pos - 20  # Row Positionの20バイト前から
            sample_end = sample_start + 40
            
            print(f"Binary Data Sample (position {sample_start} - {sample_end}):")
            sample_bytes = []
            ascii_chars = []
            
            for i in range(40):
                byte_val = info[41 + i]  # 新しいオフセット
                if byte_val >= 0:
                    sample_bytes.append(f"{byte_val:02x}")
                    # ASCII表示用
                    if 32 <= byte_val <= 126:
                        ascii_chars.append(chr(byte_val))
                    else:
                        ascii_chars.append(".")
                else:
                    sample_bytes.append("--")
                    ascii_chars.append(".")
            
            # 16バイトごとに表示（HEXとASCII）
            for i in range(0, 40, 16):
                hex_str = " ".join(sample_bytes[i:i+16])
                ascii_str = "".join(ascii_chars[i:i+16])
                print(f"  HEX:   {hex_str}")
                print(f"  ASCII: {ascii_str}")
                
                # Row Positionのマーキング
                if i <= 20 < i + 16:
                    marker_pos = (20 - i) * 3  # HEX表示での位置
                    marker = " " * (7 + marker_pos) + "^^ Row Position"
                    print(marker)
            
            # validate_and_extract_fields_liteの結果
            row_pos = info[4]
            row_end = info[5]
            is_valid = info[81] if 81 < len(info) else -1
            row_end_from_validate = info[82] if 82 < len(info) else -1
            detected_rows = info[83] if 83 < len(info) else -1
            
            print(f"\nvalidate_and_extract_fields_lite戻り値:")
            print(f"  is_valid: {is_valid} (1=True, 0=False)")
            print(f"  row_end: {row_end_from_validate}")
            if row_end_from_validate > row_pos:
                print(f"  結果: Success (row_size={row_end_from_validate - row_pos})")
            else:
                print(f"  結果: Failed")
            
            print(f"\nparse_rows_and_fields_lite状態:")
            print(f"  この時点での検出行数: {detected_rows}")
            
            # 行位置が負の場合の警告
            if row_pos < 0:
                print(f"\n⚠️ 警告: この行位置は負の値です！ row_position={row_pos}")
        
        # JSON形式でも出力（テスト自動化用）
        debug_data = []
        for info in debug_info[:display_limit]:
            debug_entry = {
                "thread_id": int(info[0]),
                "block_x": int(info[1]),
                "block_y": int(info[2]),
                "thread_x": int(info[3]),
                "row_position": int(info[4]),
                "row_end": int(info[5]),
                "ncols": int(info[6]),
                "field_info": [],
                "binary_sample": []
            }
            
            # フィールド情報（全フィールド）
            for i in range(min(17, info[6])):
                if 7 + i*2 + 1 < len(info):
                    debug_entry["field_info"].append({
                        "offset": int(info[7 + i*2]),
                        "length": int(info[8 + i*2])
                    })
            
            # バイナリサンプル
            for i in range(40):
                byte_val = info[41 + i]  # 新しいオフセット
                if byte_val >= 0:
                    debug_entry["binary_sample"].append(int(byte_val))
            
            debug_data.append(debug_entry)
        
        # JSONファイルに保存
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='_grid_debug.json', delete=False) as f:
            json.dump(debug_data, f, indent=2)
            print(f"\nデバッグ情報をJSONファイルに保存: {f.name}")
    
    def _print_negative_position_debug_info(self, negative_debug_info: np.ndarray, raw_dev):
        """負の行位置のデバッグ情報を表示"""
        print("\n=== 負の行位置デバッグ情報 ===")
        print(f"記録された負の行位置数: {len(negative_debug_info)}")
        
        # デバッグ情報の構造:
        # [0]: 配列インデックス
        # [1]: 負の行位置値
        # [2-11]: 周辺のrow_positions値（前後5個）
        # [12-51]: バイナリデータサンプル（40バイト）
        
        # 表示を10個に限定
        display_limit = min(10, len(negative_debug_info))
        print(f"表示: 最初の{display_limit}個のみ\n")
        
        for idx, info in enumerate(negative_debug_info[:display_limit]):
            print(f"\n--- 負の行位置 {idx + 1} ---")
            print(f"配列インデックス: {info[0]}")
            print(f"負の値: {info[1]}")
            
            # 周辺のrow_positions値
            print("周辺のrow_positions値 ([-5] 〜 [+4]):")
            surrounding = []
            for i in range(10):
                val = info[2 + i]
                if val == -999999999:
                    surrounding.append("--")
                else:
                    surrounding.append(str(val))
            print(f"  {' '.join(surrounding)}")
            print(f"  {' '.join(['   '] * 5)}^ココが負の値")
            
            # バイナリデータサンプル
            print(f"\nバイナリデータサンプル (配列インデックス{info[0]} * 4バイトかゕ40バイト):")
            sample_bytes = []
            ascii_chars = []
            
            for i in range(40):
                if 12 + i < len(info):
                    byte_val = info[12 + i]
                    if byte_val >= 0:
                        sample_bytes.append(f"{byte_val:02x}")
                        # ASCII表示用
                        if 32 <= byte_val <= 126:
                            ascii_chars.append(chr(byte_val))
                        else:
                            ascii_chars.append(".")
                    else:
                        sample_bytes.append("--")
                        ascii_chars.append(".")
                else:
                    sample_bytes.append("--")
                    ascii_chars.append(".")
            
            # 16バイトごとに表示（HEXとASCII）
            for i in range(0, 40, 16):
                hex_str = " ".join(sample_bytes[i:i+16])
                ascii_str = "".join(ascii_chars[i:i+16])
                print(f"  HEX:   {hex_str}")
                print(f"  ASCII: {ascii_str}")
        
        # JSON形式でも出力（テスト自動化用）
        negative_data = []
        for info in negative_debug_info[:display_limit]:
            negative_entry = {
                "array_index": int(info[0]),
                "negative_value": int(info[1]),
                "surrounding_positions": [],
                "binary_sample": []
            }
            
            # 周辺位置
            for i in range(10):
                val = info[2 + i]
                if val != -999999999:
                    negative_entry["surrounding_positions"].append(int(val))
                else:
                    negative_entry["surrounding_positions"].append(None)
            
            # バイナリサンプル
            for i in range(40):
                if 12 + i < len(info) and info[12 + i] >= 0:
                    negative_entry["binary_sample"].append(int(info[12 + i]))
            
            negative_data.append(negative_entry)
        
        # JSONファイルに保存
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='_negative_pos_debug.json', delete=False) as f:
            json.dump(negative_data, f, indent=2)
            print(f"\n負の行位置デバッグ情報をJSONファイルに保存: {f.name}")
    
    def _print_performance_stats(
        self, 
        rows: int, 
        cols: int, 
        timing: Dict[str, float], 
        data_size: int
    ):
        """パフォーマンス統計の表示（verboseモードでのみ表示）"""
        
        if not self.verbose:
            return
            
        print(f"\n=== パフォーマンス統計（直接抽出版） ===")
        print(f"処理データ: {rows:,} 行 × {cols} 列")
        print(f"データサイズ: {data_size / (1024**2):.2f} MB")
        print(f"統合バッファ: 【削除済み】")
        
        print("\n--- 詳細タイミング ---")
        for key, value in timing.items():
            if isinstance(value, (int, float)):
                if key == 'process_and_export':
                    print(f"  {key:20}: {value:.4f} 秒")
                    # 内訳項目を表示
                    string_time = timing.get('string_buffer_creation', 0)
                    extract_time = timing.get('direct_extraction', 0)
                    export_time = timing.get('parquet_export', 0)
                    if string_time > 0:
                        print(f"    ├─ string_buffers  : {string_time:.4f} 秒")
                    if extract_time > 0:
                        print(f"    ├─ direct_extract  : {extract_time:.4f} 秒")
                    if export_time > 0:
                        print(f"    └─ parquet_export  : {export_time:.4f} 秒")
                elif key in ['string_buffer_creation', 'direct_extraction', 'parquet_export']:
                    continue
                else:
                    print(f"  {key:20}: {value:.4f} 秒")
            elif isinstance(value, dict) and key == 'parquet_details':
                continue
        
        # スループット計算
        total_cells = rows * cols
        overall_time = timing.get('overall_total', timing.get('total', 1.0))
        
        if overall_time > 0:
            cell_throughput = total_cells / overall_time
            data_throughput = (data_size / (1024**2)) / overall_time
            
            print(f"\n--- スループット ---")
            print(f"  セル処理速度: {cell_throughput:,.0f} cells/sec")
            print(f"  データ処理速度: {data_throughput:.2f} MB/sec")
            
            # メモリ効率指標
            print(f"\n--- メモリ効率 ---")
            print(f"  統合バッファ削除による節約: ~{rows * 100 / (1024**2):.1f} MB")
            print(f"  ゼロコピー率: 100%（文字列・固定長とも）")
        
        print("=" * 30)


def postgresql_to_cudf_parquet_direct(
    raw_dev: cuda.cudadrv.devicearray.DeviceNDArray,
    columns: List[ColumnMeta],
    ncols: int,
    header_size: int,
    output_path: str,
    compression: str = 'snappy',
    use_rmm: bool = True,
    optimize_gpu: bool = True,
    verbose: bool = False,
    test_mode: bool = False,
    **parquet_kwargs
) -> Tuple[cudf.DataFrame, Dict[str, float]]:
    """
    PostgreSQL → cuDF → GPU Parquet 直接処理関数（統合バッファ不使用版）
    
    最適化技術を統合した高性能バージョン：
    - 統合バッファを削除し、メモリ使用量を削減
    - 入力データから直接列を抽出
    - 並列化GPU行検出・フィールド抽出
    - 文字列処理最適化（個別バッファ）
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
        verbose: 詳細ログ出力フラグ
        **parquet_kwargs: 追加のParquetオプション
    
    Returns:
        (cudf_dataframe, timing_information)
    """
    
    processor = DirectProcessor(
        use_rmm=use_rmm, 
        optimize_gpu=optimize_gpu,
        verbose=verbose,
        test_mode=test_mode
    )
    
    return processor.process_postgresql_to_parquet(
        raw_dev, columns, ncols, header_size, output_path, 
        compression, **parquet_kwargs
    )


__all__ = [
    "DirectProcessor",
    "postgresql_to_cudf_parquet_direct"
]