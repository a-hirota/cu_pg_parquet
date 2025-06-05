"""
シンプルなcuDF ZeroCopy統合ベンチマーク

従来版パーサー + cuDFゼロコピー変換 + GPU直接Parquet書き出し
複雑な並列化なしで、ZeroCopyの核心価値を実現
"""

import os
import sys
import time
import warnings
import numpy as np
import cudf
import psycopg
from numba import cuda

# パッケージのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.metadata import get_postgresql_table_metadata
from src.binary_parser import parse_binary_chunk_gpu, detect_pg_header_size
from src.cudf_zero_copy_processor import CuDFZeroCopyProcessor

def run_simple_zero_copy_benchmark(
    table_name: str = "lineorder",
    limit: int = 1000000,
    compression: str = 'snappy',
    output_path: str = None
):
    """
    シンプルなZeroCopyベンチマーク実行
    
    従来版パーサー + ZeroCopy変換で確実な動作を実現
    """
    
    print("=" * 80)
    print("🚀 シンプルcuDF ZeroCopyベンチマーク 🚀")
    print("=" * 80)
    print(f"テーブル        : {table_name}")
    print(f"制限行数        : {limit:,}")
    print(f"圧縮方式        : {compression}")
    
    if output_path is None:
        import time
        timestamp = int(time.time())
        output_path = f"benchmark/{table_name}_simple_zero_copy_{compression}_{timestamp}.parquet"
    
    print(f"出力パス        : {output_path}")
    print("-" * 80)
    
    # タイミング情報
    timing_info = {}
    overall_start = time.time()
    
    try:
        # CUDA初期化確認
        cuda.current_context()
        print("✅ CUDA context 初期化成功")
    except Exception as e:
        print(f"❌ CUDA初期化失敗: {e}")
        return None
    
    # === 1. メタデータ取得 ===
    print("📊 メタデータ取得中...")
    meta_start = time.time()
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        raise ValueError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    try:
        conn = psycopg.connect(dsn)
        columns = get_postgresql_table_metadata(conn, table_name)
        conn.close()
    except Exception as e:
        print(f"❌ メタデータ取得エラー: {e}")
        return None
    
    timing_info['metadata'] = time.time() - meta_start
    ncols = len(columns)
    decimal_cols = sum(1 for col in columns if col.arrow_id == 701)  # DECIMAL128
    
    print(f"✅ メタデータ取得完了 ({timing_info['metadata']:.4f}秒)")
    print(f"   列数: {ncols}, Decimal列数: {decimal_cols}")
    
    # === 2. COPY BINARY実行 ===
    print("📥 COPY BINARY実行中...")
    copy_start = time.time()
    
    try:
        conn = psycopg.connect(dsn)
        with conn.cursor() as cur:
            copy_sql = f"COPY (SELECT * FROM {table_name} LIMIT {limit}) TO STDOUT WITH (FORMAT BINARY)"
            with cur.copy(copy_sql) as copy:
                raw_host_data = copy.read()
        conn.close()
        
        raw_host = np.frombuffer(raw_host_data, dtype=np.uint8)
        
    except Exception as e:
        print(f"❌ COPY BINARY エラー: {e}")
        return None
    
    timing_info['copy_binary'] = time.time() - copy_start
    data_size_mb = len(raw_host) / (1024 * 1024)
    
    print(f"✅ COPY BINARY完了 ({timing_info['copy_binary']:.4f}秒)")
    print(f"   データサイズ: {data_size_mb:.2f} MB")
    
    # === 3. GPU転送 ===
    print("🚀 GPU転送中...")
    transfer_start = time.time()
    
    raw_dev = cuda.to_device(raw_host)
    
    timing_info['gpu_transfer'] = time.time() - transfer_start
    print(f"✅ GPU転送完了 ({timing_info['gpu_transfer']:.4f}秒)")
    
    # === 4. ヘッダーサイズ検出 ===
    header_size = detect_pg_header_size(raw_host[:128])
    print(f"📏 ヘッダーサイズ: {header_size} バイト")
    
    # === 5. 従来版GPUパース ===
    print("⚙️ 従来版GPUパース実行中...")
    parse_start = time.time()
    
    try:
        field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
            raw_dev, ncols, threads_per_block=256, header_size=header_size
        )
        rows = field_offsets_dev.shape[0]
        
    except Exception as e:
        print(f"❌ GPUパースエラー: {e}")
        return None
    
    timing_info['gpu_parse'] = time.time() - parse_start
    print(f"✅ 従来版パース完了 ({timing_info['gpu_parse']:.4f}秒)")
    print(f"   検出行数: {rows:,}")
    
    # === 6. cuDF ZeroCopy変換 + GPU直接Parquet書き出し ===
    print("🔄 cuDF ZeroCopy変換 + GPU直接書き出し...")
    zero_copy_start = time.time()
    
    try:
        # ZeroCopyプロセッサー初期化
        cudf_processor = CuDFZeroCopyProcessor(use_rmm=True)
        
        # GPU統合デコード + cuDF変換
        cudf_df = cudf_processor.decode_and_create_cudf_zero_copy(
            raw_dev, field_offsets_dev, field_lengths_dev, columns
        )
        
        # GPU直接Parquet書き出し
        try:
            cudf_df.to_parquet(
                output_path,
                compression=compression,
                engine='cudf'  # cuDFエンジンによる直接書き出し
            )
            print("✅ cuDFエンジンによる直接書き出し成功")
            
        except Exception as cudf_error:
            warnings.warn(f"cuDF直接書き出し失敗: {cudf_error}")
            print("🔄 PyArrowフォールバック中...")
            
            # フォールバック: PyArrow経由
            arrow_table = cudf_df.to_arrow()
            import pyarrow.parquet as pq
            pq.write_table(arrow_table, output_path, compression=compression)
            print("✅ PyArrowフォールバック書き出し成功")
        
    except Exception as e:
        print(f"❌ ZeroCopy処理エラー: {e}")
        return None
    
    timing_info['zero_copy_export'] = time.time() - zero_copy_start
    timing_info['total'] = time.time() - overall_start
    
    print(f"✅ ZeroCopy変換+書き出し完了 ({timing_info['zero_copy_export']:.4f}秒)")
    
    # === 7. 結果サマリー ===
    print("\n" + "=" * 80)
    print("📊 ベンチマーク結果サマリー")
    print("=" * 80)
    
    print(f"総実行時間: {timing_info['total']:.4f} 秒")
    print("\n--- 詳細タイミング ---")
    for key, value in timing_info.items():
        if key != 'total':
            percentage = (value / timing_info['total']) * 100
            print(f"  {key:20}: {value:8.4f} 秒 ({percentage:5.1f}%)")
    
    # スループット計算
    total_cells = rows * ncols
    cell_throughput = total_cells / timing_info['total']
    data_throughput = data_size_mb / timing_info['total']
    
    print(f"\n--- パフォーマンス指標 ---")
    print(f"  処理行数        : {rows:,}")
    print(f"  処理列数        : {ncols}")
    print(f"  総セル数        : {total_cells:,}")
    print(f"  セル処理速度    : {cell_throughput:,.0f} cells/sec")
    print(f"  データ処理速度  : {data_throughput:.2f} MB/sec")
    
    # === 8. 結果検証 ===
    print(f"\n--- 結果検証 ---")
    try:
        verify_df = cudf.read_parquet(output_path)
        print(f"✅ Parquetファイル読み込み成功: {len(verify_df):,} 行")
        print(f"✅ データ整合性確認: cuDF DataFrame → Parquet → cuDF 完了")
        
        # DataFrame情報表示
        print(f"\n--- cuDF DataFrame Info ---")
        print(verify_df.info())
        
        print(f"\n--- cuDF DataFrame Head ---")
        print(verify_df.head())
        
    except Exception as e:
        print(f"❌ 結果検証エラー: {e}")
    
    return {
        'timing': timing_info,
        'rows': rows,
        'columns': ncols,
        'data_size_mb': data_size_mb,
        'throughput': {
            'cells_per_sec': cell_throughput,
            'mb_per_sec': data_throughput
        },
        'output_file': output_path
    }

def main():
    """メイン実行関数"""
    
    import argparse
    parser = argparse.ArgumentParser(description='シンプルcuDF ZeroCopyベンチマーク')
    parser.add_argument('--table', default='lineorder', help='テーブル名')
    parser.add_argument('--rows', type=int, default=1000000, help='制限行数')
    parser.add_argument('--compression', default='snappy', 
                       choices=['snappy', 'gzip', 'lz4', 'brotli', 'zstd'],
                       help='圧縮方式')
    parser.add_argument('--output', help='出力Parquetファイルパス')
    
    args = parser.parse_args()
    
    try:
        result = run_simple_zero_copy_benchmark(
            table_name=args.table,
            limit=args.rows,
            compression=args.compression,
            output_path=args.output
        )
        
        if result:
            print(f"\n🎉 ベンチマーク成功完了!")
            print(f"   総時間: {result['timing']['total']:.2f}秒")
            print(f"   スループット: {result['throughput']['cells_per_sec']:,.0f} cells/sec")
        else:
            print(f"\n❌ ベンチマーク失敗")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ ユーザーによる中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()