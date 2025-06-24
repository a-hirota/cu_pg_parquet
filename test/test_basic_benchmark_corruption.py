"""基本ベンチマークの文字列破損チェック

既存のチャンクファイルを使用して破損をチェック
"""

import os
import cudf

print("=== 基本ベンチマークの文字列破損チェック ===")

# /dev/shmにある既存のチャンクファイルを探す
chunk_files = []
for i in range(8):
    chunk_file = f"/dev/shm/chunk_{i}.bin"
    if os.path.exists(chunk_file):
        chunk_files.append((i, chunk_file))

print(f"発見されたチャンク: {len(chunk_files)}個")

if not chunk_files:
    print("チャンクファイルが見つかりません。Rustでチャンクを生成してください。")
    exit(1)

# 基本ベンチマークのParquetファイルを探す
basic_parquet_files = []
for i in range(8):
    # 通常の出力パス
    parquet_file = f"/dev/shm/chunk_{i}.parquet"
    if os.path.exists(parquet_file):
        basic_parquet_files.append((i, parquet_file))
        continue
    
    # ベンチマークディレクトリ
    parquet_file = f"benchmark/chunk_{i}_basic.parquet"
    if os.path.exists(parquet_file):
        basic_parquet_files.append((i, parquet_file))

print(f"\n基本ベンチマークParquetファイル: {len(basic_parquet_files)}個")

if not basic_parquet_files:
    print("\n基本ベンチマークを実行してParquetファイルを生成中...")
    
    # 基本ベンチマークのメイン処理を直接呼び出し
    from src.main_postgres_to_parquet import ZeroCopyProcessor
    from src.metadata import fetch_column_meta
    from src.cuda_kernels.postgresql_binary_parser import detect_pg_header_size
    from numba import cuda
    import numpy as np
    import psycopg
    
    # メタデータ取得
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
    conn.close()
    
    # 最初のチャンクを処理
    chunk_id, chunk_file = chunk_files[0]
    print(f"\nチャンク{chunk_id}を処理中...")
    
    with open(chunk_file, 'rb') as f:
        data = f.read()
    
    raw_host = np.frombuffer(data, dtype=np.uint8)
    raw_dev = cuda.to_device(raw_host)
    
    # ヘッダーサイズ検出
    header_sample = raw_dev[:128].copy_to_host()
    header_size = detect_pg_header_size(header_sample)
    
    # 基本ベンチマーク方式で処理
    processor = ZeroCopyProcessor(use_rmm=True, optimize_gpu=True)
    output_path = f"benchmark/chunk_{chunk_id}_basic_test.parquet"
    
    try:
        cudf_df, timing = processor.process_postgresql_to_parquet(
            raw_dev, columns, len(columns), header_size, output_path
        )
        print(f"✅ 基本ベンチマーク処理完了: {len(cudf_df)}行")
        basic_parquet_files = [(chunk_id, output_path)]
        
    except Exception as e:
        print(f"❌ 基本ベンチマーク処理エラー: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

# 文字列破損をチェック
print("\n=== 文字列破損チェック ===")

total_rows_checked = 0
even_errors = 0
odd_errors = 0
error_samples = []

for chunk_id, parquet_file in basic_parquet_files[:1]:  # 最初のチャンクのみ
    print(f"\nチャンク{chunk_id}: {parquet_file}")
    
    try:
        df = cudf.read_parquet(parquet_file)
        rows = len(df)
        print(f"  行数: {rows:,}")
        
        if 'lo_orderpriority' in df.columns:
            # 最初の10000行をチェック
            check_rows = min(10000, rows)
            
            for i in range(check_rows):
                try:
                    value = df['lo_orderpriority'].iloc[i]
                    expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                    is_valid = any(value.startswith(p) for p in expected_patterns)
                    
                    if not is_valid:
                        if i % 2 == 0:
                            even_errors += 1
                        else:
                            odd_errors += 1
                        
                        if len(error_samples) < 10:
                            error_samples.append((i, value))
                        
                except Exception as e:
                    pass
            
            total_rows_checked += check_rows
            
            # 中間部分もチェック
            if rows > 1000000:
                middle_start = rows // 2
                for i in range(middle_start, middle_start + 5000):
                    if i >= rows:
                        break
                    try:
                        value = df['lo_orderpriority'].iloc[i]
                        expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                        is_valid = any(value.startswith(p) for p in expected_patterns)
                        
                        if not is_valid:
                            if i % 2 == 0:
                                even_errors += 1
                            else:
                                odd_errors += 1
                    except:
                        pass
                    
                total_rows_checked += min(5000, rows - middle_start)
        
        del df
        
    except Exception as e:
        print(f"  エラー: {e}")

# 結果表示
print(f"\n=== 結果 ===")
print(f"チェック行数: {total_rows_checked:,}")
print(f"偶数行エラー: {even_errors}")
print(f"奇数行エラー: {odd_errors}")
print(f"総エラー数: {even_errors + odd_errors}")

if error_samples:
    print(f"\nエラーサンプル（最初の{len(error_samples)}個）:")
    for row, value in error_samples:
        print(f"  行{row}({'偶数' if row % 2 == 0 else '奇数'}): {repr(value)}")

# 診断
print(f"\n=== 診断 ===")
if odd_errors > even_errors * 2:
    print("❌ 基本ベンチマークでも奇数行に顕著な破損パターン!")
    print("→ 共通のコンポーネント（GPUパーサーまたは文字列処理）に問題がある")
elif even_errors + odd_errors == 0:
    print("✅ 基本ベンチマークでは破損なし")
    print("→ 直接抽出版固有の問題の可能性が高い")
else:
    print("⚠️ 基本ベンチマークでも破損あり（パターン不明）")
    
# クリーンアップオプション
if 'benchmark/chunk_' in parquet_file:
    print(f"\nテストファイル削除: {parquet_file}")
    os.remove(parquet_file)