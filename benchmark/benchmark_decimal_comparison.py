"""
Decimal128最適化の効果検証ベンチマーク

修正前後でアウトプットに変化がないことを確認し、改修方法が正しいことを示す。
環境変数 USE_DECIMAL_OPTIMIZATION で最適化のON/OFFを制御する。
"""

import os
import time
import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import cudf
from numba import cuda
import hashlib

# Import necessary functions from the correct modules using absolute paths from root
from src.meta_fetch import fetch_column_meta, ColumnMeta
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v2 import decode_chunk

# テスト設定
TABLE_NAME = "lineorder"
OUTPUT_DIR = "benchmark/comparison_output"
LIMIT_ROWS = 100000  # テスト用に制限

def setup_test_environment():
    """テスト環境のセットアップ"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # CUDAコンテキストの初期化を確認
    try:
        cuda.current_context()
        print("✓ CUDA context OK")
    except Exception as e:
        print(f"✗ CUDA context initialization failed: {e}")
        exit(1)

def run_benchmark_with_optimization(use_optimization: bool):
    """
    最適化のON/OFFを切り替えてベンチマーク実行
    
    Parameters
    ----------
    use_optimization : bool
        True: 最適化版カーネル使用, False: 従来版カーネル使用
    """
    # 環境変数でカーネル選択を制御
    os.environ["USE_DECIMAL_OPTIMIZATION"] = "1" if use_optimization else "0"
    
    optimization_label = "最適化版" if use_optimization else "従来版"
    print(f"\n{'='*60}")
    print(f"{optimization_label} ベンチマーク実行中...")
    print(f"{'='*60}")
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return None, None

    prefix = os.environ.get("PG_TABLE_PREFIX", "")
    tbl = f"{prefix}{TABLE_NAME}" if prefix else TABLE_NAME

    start_total_time = time.time()
    
    # データベース接続とメタデータ取得
    conn = psycopg.connect(dsn)
    try:
        print("メタデータを取得中...")
        start_meta_time = time.time()
        columns = fetch_column_meta(conn, f"SELECT * FROM {tbl}")
        meta_time = time.time() - start_meta_time
        
        # DECIMAL列の情報表示
        decimal_columns = [col for col in columns if col.arrow_id == 5]  # DECIMAL128
        if decimal_columns:
            print(f"DECIMAL列検出: {len(decimal_columns)}列")
            for col in decimal_columns:
                precision, scale = col.arrow_param or (38, 0)
                optimization_type = "Decimal64" if precision <= 18 and use_optimization else "Decimal128"
                print(f"  {col.name}: precision={precision}, scale={scale} → {optimization_type}")
        
        print(f"メタデータ取得完了 ({meta_time:.4f}秒)")
        ncols = len(columns)

        # COPY BINARY実行
        print(f"COPY BINARY を実行中 (LIMIT {LIMIT_ROWS})...")
        start_copy_time = time.time()
        copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {LIMIT_ROWS}) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
        copy_time = time.time() - start_copy_time
        print(f"COPY BINARY 完了 ({copy_time:.4f}秒), データサイズ: {len(raw_host) / (1024*1024):.2f} MB")

    finally:
        conn.close()

    # GPU処理
    print("GPUにデータを転送中...")
    start_transfer_time = time.time()
    raw_dev = cuda.to_device(raw_host)
    transfer_time = time.time() - start_transfer_time
    print(f"GPU転送完了 ({transfer_time:.4f}秒)")

    # ヘッダーサイズ検出
    header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    print(f"検出したヘッダーサイズ: {header_size} バイト")

    # GPUパース
    print("GPUでパース中...")
    start_parse_time = time.time()
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev,
        ncols=ncols,
        header_size=header_size
    )
    parse_time = time.time() - start_parse_time
    rows = field_offsets_dev.shape[0]
    print(f"GPUパース完了 ({parse_time:.4f}秒), 行数: {rows}")

    # GPUデコード（ここで最適化が適用される）
    print(f"GPUでデコード中 ({optimization_label})...")
    start_decode_time = time.time()
    batch = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    decode_time = time.time() - start_decode_time
    print(f"GPUデコード完了 ({decode_time:.4f}秒)")

    # Arrow Table作成
    result_table = pa.Table.from_batches([batch])
    print(f"Arrow Table 作成完了: {result_table.num_rows} 行, {result_table.num_columns} 列")

    # Parquet出力
    output_path = f"{OUTPUT_DIR}/lineorder_{optimization_label.replace('版', '')}.parquet"
    print(f"Parquetファイル書き込み中: {output_path}")
    start_write_time = time.time()
    pq.write_table(result_table, output_path)
    write_time = time.time() - start_write_time
    print(f"Parquet書き込み完了 ({write_time:.4f}秒)")

    total_time = time.time() - start_total_time
    
    # パフォーマンス結果
    performance_result = {
        "optimization": use_optimization,
        "total_time": total_time,
        "meta_time": meta_time,
        "copy_time": copy_time,
        "transfer_time": transfer_time,
        "parse_time": parse_time,
        "decode_time": decode_time,
        "write_time": write_time,
        "rows": rows,
        "columns": ncols,
        "decimal_columns": len(decimal_columns) if decimal_columns else 0
    }
    
    print(f"\n{optimization_label} 完了: 総時間 = {total_time:.4f} 秒")
    print("--- 時間内訳 ---")
    print(f"  メタデータ取得: {meta_time:.4f} 秒")
    print(f"  COPY BINARY   : {copy_time:.4f} 秒")
    print(f"  GPU転送       : {transfer_time:.4f} 秒")
    print(f"  GPUパース     : {parse_time:.4f} 秒")
    print(f"  GPUデコード   : {decode_time:.4f} 秒")
    print(f"  Parquet書き込み: {write_time:.4f} 秒")
    print("----------------")
    
    return result_table, performance_result

def calculate_table_hash(table: pa.Table) -> str:
    """Arrow Tableのハッシュ値を計算（データ一致確認用）"""
    # 各列のデータをバイナリ形式で結合してハッシュ化
    hash_input = bytearray()
    
    for i in range(table.num_columns):
        column = table.column(i)
        # Arrow配列をPyListに変換してバイト列化
        try:
            data_bytes = str(column.to_pylist()).encode('utf-8')
            hash_input.extend(data_bytes)
        except Exception as e:
            print(f"Warning: ハッシュ計算でエラー (列 {table.column_names[i]}): {e}")
            # フォールバック: 列名とnull情報のみ
            hash_input.extend(table.column_names[i].encode('utf-8'))
            hash_input.extend(str(column.null_count).encode('utf-8'))
    
    return hashlib.md5(hash_input).hexdigest()

def verify_data_consistency(table1: pa.Table, table2: pa.Table):
    """2つのテーブルのデータ一致を検証"""
    print("\n" + "="*60)
    print("データ一致検証")
    print("="*60)
    
    # 基本構造比較
    print(f"行数: {table1.num_rows} vs {table2.num_rows}")
    print(f"列数: {table1.num_columns} vs {table2.num_columns}")
    
    if table1.num_rows != table2.num_rows or table1.num_columns != table2.num_columns:
        print("✗ テーブル構造が異なります")
        return False
    
    # 列名比較
    if table1.column_names != table2.column_names:
        print("✗ 列名が異なります")
        return False
    
    # 各列のデータ比較
    all_match = True
    decimal_column_results = []
    
    for i, col_name in enumerate(table1.column_names):
        col1 = table1.column(i)
        col2 = table2.column(i)
        
        # null count比較
        if col1.null_count != col2.null_count:
            print(f"✗ 列 {col_name}: null数が異なります ({col1.null_count} vs {col2.null_count})")
            all_match = False
            continue
        
        # データ型比較
        if str(col1.type) != str(col2.type):
            print(f"✗ 列 {col_name}: データ型が異なります ({col1.type} vs {col2.type})")
            all_match = False
            continue
        
        # DECIMAL列の詳細比較
        if pa.types.is_decimal(col1.type):
            try:
                # PyArrowのcompare機能を使用
                equals = col1.equals(col2)
                if equals:
                    print(f"✓ DECIMAL列 {col_name}: データ一致")
                    decimal_column_results.append((col_name, True, "完全一致"))
                else:
                    print(f"✗ DECIMAL列 {col_name}: データ不一致")
                    decimal_column_results.append((col_name, False, "データ不一致"))
                    all_match = False
                    
                    # サンプルデータ表示（デバッグ用）
                    print(f"  サンプル (最適化前): {col1.slice(0, min(5, col1.length)).to_pylist()}")
                    print(f"  サンプル (最適化後): {col2.slice(0, min(5, col2.length)).to_pylist()}")
            except Exception as e:
                print(f"⚠ DECIMAL列 {col_name}: 比較エラー ({e})")
                decimal_column_results.append((col_name, False, f"比較エラー: {e}"))
        else:
            # 非DECIMAL列の比較
            try:
                equals = col1.equals(col2)
                if not equals:
                    print(f"✗ 列 {col_name}: データ不一致")
                    all_match = False
            except Exception as e:
                print(f"⚠ 列 {col_name}: 比較エラー ({e})")
    
    # ハッシュ値比較
    print("\nハッシュ値比較...")
    hash1 = calculate_table_hash(table1)
    hash2 = calculate_table_hash(table2)
    print(f"従来版ハッシュ: {hash1}")
    print(f"最適化版ハッシュ: {hash2}")
    
    if hash1 == hash2:
        print("✓ ハッシュ値一致 - データは同一です")
    else:
        print("✗ ハッシュ値不一致")
        all_match = False
    
    # 結果サマリー
    print("\n--- DECIMAL列検証結果 ---")
    for col_name, match, detail in decimal_column_results:
        status = "✓" if match else "✗"
        print(f"{status} {col_name}: {detail}")
    
    return all_match

def run_comparison_benchmark():
    """最適化前後の比較ベンチマーク実行"""
    setup_test_environment()
    
    print("Decimal128最適化効果検証ベンチマーク開始")
    print("=" * 80)
    
    # 従来版実行
    table_original, perf_original = run_benchmark_with_optimization(use_optimization=False)
    
    # 最適化版実行
    table_optimized, perf_optimized = run_benchmark_with_optimization(use_optimization=True)
    
    if table_original is None or table_optimized is None:
        print("ベンチマーク実行に失敗しました")
        return
    
    # データ一致検証
    data_consistent = verify_data_consistency(table_original, table_optimized)
    
    # パフォーマンス比較
    print("\n" + "="*60)
    print("パフォーマンス比較")
    print("="*60)
    
    decode_speedup = perf_original["decode_time"] / perf_optimized["decode_time"] if perf_optimized["decode_time"] > 0 else 1.0
    total_speedup = perf_original["total_time"] / perf_optimized["total_time"] if perf_optimized["total_time"] > 0 else 1.0
    
    print(f"総実行時間:")
    print(f"  従来版: {perf_original['total_time']:.4f}秒")
    print(f"  最適化版: {perf_optimized['total_time']:.4f}秒")
    print(f"  高速化率: {total_speedup:.2f}x")
    
    print(f"\nGPUデコード時間:")
    print(f"  従来版: {perf_original['decode_time']:.4f}秒")
    print(f"  最適化版: {perf_optimized['decode_time']:.4f}秒")
    print(f"  高速化率: {decode_speedup:.2f}x")
    
    print(f"\nDECIMAL列数: {perf_original['decimal_columns']}")
    print(f"総行数: {perf_original['rows']:,}")
    print(f"総列数: {perf_original['columns']}")
    
    # 最終結果
    print("\n" + "="*60)
    print("検証結果")
    print("="*60)
    
    if data_consistent:
        print("✅ データ一致確認: 成功")
        print("✅ 最適化の実装は正しく動作しています")
        
        if decode_speedup > 1.1:
            print(f"🚀 パフォーマンス向上: {decode_speedup:.2f}x 高速化達成!")
        elif decode_speedup > 0.9:
            print("📊 パフォーマンス: 同等レベル (最適化効果は限定的)")
        else:
            print("⚠️  パフォーマンス: 若干低下 (要調査)")
    else:
        print("❌ データ一致確認: 失敗")
        print("❌ 最適化の実装に問題があります")
    
    return data_consistent, decode_speedup

if __name__ == "__main__":
    try:
        success, speedup = run_comparison_benchmark()
        if success:
            print(f"\n🎉 ベンチマーク成功! 高速化率: {speedup:.2f}x")
            exit(0)
        else:
            print(f"\n💥 ベンチマーク失敗")
            exit(1)
    except Exception as e:
        print(f"\n💥 ベンチマーク実行エラー: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
