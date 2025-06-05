"""
究極のcuDF ZeroCopy統合ベンチマーク

PostgreSQL → COPY BINARY → 最適化GPU並列処理 → cuDF ZeroCopy → GPU直接Parquet

全ての最適化技術を統合した究極パフォーマンス版:
1. 並列化GPU行検出・フィールド抽出  
2. メモリコアレッシング最適化
3. cuDFによるゼロコピーArrow変換
4. GPU直接Parquet書き出し
5. RMM統合メモリ管理

環境変数:
GPUPASER_PG_DSN  : PostgreSQL接続文字列
PG_TABLE_PREFIX  : テーブルプレフィックス (optional)
"""

import os
import sys
import time
import numpy as np
import psycopg
import cudf
from numba import cuda
import argparse

from src.metadata import fetch_column_meta
from src.types import ColumnMeta
from src.binary_parser import detect_pg_header_size
from src.ultimate_zero_copy_processor import ultimate_postgresql_to_cudf_parquet

TABLE_NAME = "lineorder"

def create_output_filename(method: str, compression: str = "snappy") -> str:
    """出力ファイル名を生成"""
    timestamp = int(time.time())
    return f"benchmark/lineorder_{method}_{compression}_{timestamp}.parquet"

def run_ultimate_benchmark(
    limit_rows: int = 1000000,
    compression: str = "snappy", 
    use_rmm: bool = True,
    optimize_gpu: bool = True,
    output_path: str = None
):
    """究極版ベンチマーク実行"""
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return None

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME
    
    if output_path is None:
        output_path = create_output_filename("ultimate", compression)

    print("=" * 80)
    print("🚀 究極のcuDF ZeroCopy統合ベンチマーク 🚀")
    print("=" * 80)
    print(f"テーブル        : {tbl}")
    print(f"制限行数        : {limit_rows:,}")
    print(f"圧縮方式        : {compression}")
    print(f"RMM使用         : {use_rmm}")
    print(f"GPU最適化       : {optimize_gpu}")
    print(f"出力パス        : {output_path}")
    print("-" * 80)
    
    overall_start_time = time.time()
    benchmark_results = {}
    
    # === PostgreSQL接続・メタデータ取得 ===
    conn = psycopg.connect(dsn)
    try:
        print("📊 メタデータ取得中...")
        meta_start = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {tbl}")
        benchmark_results['metadata_time'] = time.time() - meta_start
        print(f"✅ メタデータ取得完了 ({benchmark_results['metadata_time']:.4f}秒)")
        
        ncols = len(columns)
        decimal_cols = sum(1 for col in columns if col.arrow_id == 5)
        
        print(f"   列数: {ncols}, Decimal列数: {decimal_cols}")

        # === COPY BINARY実行 ===
        print("📥 COPY BINARY実行中...")
        copy_start = time.time()
        copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {limit_rows}) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
        
        benchmark_results['copy_binary_time'] = time.time() - copy_start
        data_size_mb = len(raw_host) / (1024*1024)
        print(f"✅ COPY BINARY完了 ({benchmark_results['copy_binary_time']:.4f}秒)")
        print(f"   データサイズ: {data_size_mb:.2f} MB")

    finally:
        conn.close()

    # === GPU転送 ===
    print("🚀 GPU転送中...")
    transfer_start = time.time()
    raw_dev = cuda.to_device(raw_host)
    benchmark_results['gpu_transfer_time'] = time.time() - transfer_start
    print(f"✅ GPU転送完了 ({benchmark_results['gpu_transfer_time']:.4f}秒)")

    # === ヘッダーサイズ検出 ===
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"📏 ヘッダーサイズ: {header_size} バイト")

    # === 究極統合処理実行 ===
    print("⚡ 究極統合処理開始...")
    ultimate_start = time.time()
    
    try:
        cudf_df, detailed_timing = ultimate_postgresql_to_cudf_parquet(
            raw_dev=raw_dev,
            columns=columns,
            ncols=ncols,
            header_size=header_size,
            output_path=output_path,
            compression=compression,
            use_rmm=use_rmm,
            optimize_gpu=optimize_gpu
        )
        
        benchmark_results['ultimate_processing_time'] = time.time() - ultimate_start
        benchmark_results.update(detailed_timing)
        
        rows = len(cudf_df)
        cols = len(cudf_df.columns)
        
        print(f"✅ 究極統合処理完了 ({benchmark_results['ultimate_processing_time']:.4f}秒)")
        print(f"   結果: {rows:,} 行 × {cols} 列")
        
    except Exception as e:
        print(f"❌ 究極統合処理でエラー: {e}")
        return None

    # === 総合時間計算 ===
    benchmark_results['total_time'] = time.time() - overall_start_time

    # === 結果検証 ===
    print("🔍 結果検証中...")
    try:
        verify_start = time.time()
        verification_df = cudf.read_parquet(output_path)
        benchmark_results['verification_time'] = time.time() - verify_start
        
        print(f"✅ 検証完了 ({benchmark_results['verification_time']:.4f}秒)")
        print(f"   読み込み結果: {len(verification_df):,} 行 × {len(verification_df.columns)} 列")
        
        # データ型確認
        print("   データ型:")
        for col_name, dtype in verification_df.dtypes.items():
            print(f"     {col_name}: {dtype}")
        
        # データ内容のサンプル表示
        print("\n   データサンプル（最初の5行）:")
        try:
            sample_df = verification_df.head()
            for i in range(min(5, len(sample_df))):
                row_data = []
                for col in sample_df.columns:
                    value = sample_df[col].iloc[i]
                    # 値の型と内容を適切に表示
                    if hasattr(value, 'item'):
                        value = value.item()  # numpy/cudf scalar to python
                    row_data.append(str(value)[:20])  # 長い値は切り詰め
                print(f"     行{i+1}: {row_data}")
        except Exception as e:
            print(f"     サンプル表示エラー: {e}")
        
        # 基本統計情報
        print("\n   基本統計:")
        try:
            numeric_cols = verification_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:5]:  # 最初の5つの数値列のみ
                    col_data = verification_df[col]
                    if len(col_data) > 0:
                        print(f"     {col}: 平均={float(col_data.mean()):.2f}, 最小={float(col_data.min()):.2f}, 最大={float(col_data.max()):.2f}")
        except Exception as e:
            print(f"     統計情報エラー: {e}")
            
    except Exception as e:
        print(f"⚠️  検証でエラー: {e}")
        benchmark_results['verification_time'] = 0

    # === 最終結果表示 ===
    print_final_results(benchmark_results, rows, cols, data_size_mb, decimal_cols)
    
    return {
        'dataframe': cudf_df,
        'timing': benchmark_results,
        'output_path': output_path,
        'rows': rows,
        'columns': cols,
        'data_size_mb': data_size_mb
    }

def print_final_results(timing: dict, rows: int, cols: int, data_size_mb: float, decimal_cols: int):
    """最終結果の詳細表示"""
    
    print("\n" + "=" * 80)
    print("🏆 究極ベンチマーク結果")
    print("=" * 80)
    
    # 基本統計
    print("📊 処理統計:")
    print(f"   処理行数      : {rows:,} 行")
    print(f"   処理列数      : {cols} 列")
    print(f"   Decimal列数   : {decimal_cols} 列")
    print(f"   データサイズ  : {data_size_mb:.2f} MB")
    
    # 時間内訳
    print("\n⏱️  詳細タイミング:")
    timing_keys = [
        ('metadata_time', 'メタデータ取得'),
        ('copy_binary_time', 'COPY BINARY'),
        ('gpu_transfer_time', 'GPU転送'),
        ('gpu_parsing', 'GPU並列パース'),
        ('preparation', '前処理・バッファ準備'),
        ('kernel_execution', 'GPU統合カーネル'),
        ('cudf_creation', 'cuDF作成'),
        ('parquet_export', 'Parquet書き出し'),
        ('verification_time', '結果検証'),
        ('total_time', '総実行時間')
    ]
    
    for key, label in timing_keys:
        if key in timing:
            print(f"   {label:20}: {timing[key]:8.4f} 秒")
    
    # パフォーマンス指標
    total_time = timing.get('total_time', 1.0)
    processing_time = timing.get('ultimate_processing_time', timing.get('overall_total', 1.0))
    
    if total_time > 0 and processing_time > 0:
        print("\n🚀 パフォーマンス指標:")
        
        # スループット
        total_cells = rows * cols
        cell_throughput = total_cells / processing_time
        data_throughput = data_size_mb / processing_time
        
        print(f"   セル処理速度  : {cell_throughput:,.0f} cells/sec")
        print(f"   データ処理速度: {data_throughput:.2f} MB/sec") 
        
        # 効率指標
        gpu_time = timing.get('kernel_execution', 0)
        if gpu_time > 0:
            gpu_efficiency = (gpu_time / processing_time) * 100
            print(f"   GPU使用効率   : {gpu_efficiency:.1f}%")
        
        # 全体効率
        processing_ratio = (processing_time / total_time) * 100
        print(f"   処理時間比率  : {processing_ratio:.1f}%")
    
    print("=" * 80)

def run_comparison_with_traditional():
    """従来版との詳細比較"""
    
    print("\n" + "🔥" * 20 + " 究極 vs 従来 比較ベンチマーク " + "🔥" * 20)
    
    # 1. 従来版実行
    print("\n[1/2] 📊 従来版実行中...")
    traditional_start = time.time()
    
    try:
        from benchmark.benchmark_lineorder_5m import run_benchmark as run_traditional
        run_traditional()
        traditional_time = time.time() - traditional_start
        print(f"✅ 従来版完了: {traditional_time:.4f}秒")
    except Exception as e:
        print(f"❌ 従来版でエラー: {e}")
        traditional_time = None
    
    # 2. 究極版実行
    print("\n[2/2] 🚀 究極版実行中...")
    ultimate_start = time.time()
    ultimate_result = run_ultimate_benchmark()
    ultimate_time = time.time() - ultimate_start
    
    if ultimate_result:
        print(f"✅ 究極版完了: {ultimate_time:.4f}秒")
    
    # 3. 比較結果
    if traditional_time and ultimate_result:
        print("\n" + "🏁" * 30)
        print("🏆 最終比較結果")
        print("🏁" * 30)
        print(f"従来版時間     : {traditional_time:8.4f} 秒")
        print(f"究極版時間     : {ultimate_time:8.4f} 秒")
        
        if traditional_time > 0:
            speedup = traditional_time / ultimate_time
            improvement = ((traditional_time - ultimate_time) / traditional_time) * 100
            print(f"高速化倍率     : {speedup:8.2f}x")
            print(f"性能向上       : {improvement:8.1f}%")
            
            if speedup > 1:
                print("🎉 究極版の勝利！")
            else:
                print("🤔 要調査...")
        
        print("🏁" * 30)

def main():
    """メイン関数"""
    
    parser = argparse.ArgumentParser(description='究極のcuDF ZeroCopyベンチマーク')
    parser.add_argument('--rows', type=int, default=1000000, help='処理行数制限')
    parser.add_argument('--compression', choices=['snappy', 'gzip', 'lz4', 'none'], 
                       default='snappy', help='圧縮方式')
    parser.add_argument('--no-rmm', action='store_true', help='RMMを無効化')
    parser.add_argument('--no-gpu-opt', action='store_true', help='GPU最適化を無効化')
    parser.add_argument('--compare', action='store_true', help='従来版との比較実行')
    parser.add_argument('--output', type=str, help='出力ファイルパス')
    
    args = parser.parse_args()
    
    # CUDA初期化確認
    try:
        cuda.current_context()
        print("✅ CUDA context 初期化成功")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        sys.exit(1)
    
    # 実行モード選択
    if args.compare:
        run_comparison_with_traditional()
    else:
        result = run_ultimate_benchmark(
            limit_rows=args.rows,
            compression=args.compression,
            use_rmm=not args.no_rmm,
            optimize_gpu=not args.no_gpu_opt,
            output_path=args.output
        )
        
        if result:
            print(f"\n🎯 出力ファイル: {result['output_path']}")
            print("🎉 究極ベンチマーク完了！")
        else:
            print("❌ ベンチマーク失敗")
            sys.exit(1)

if __name__ == "__main__":
    main()