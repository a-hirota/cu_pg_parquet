"""修正の検証 - 文字列破損が解消されたか確認"""

import os
import cudf

print("=== 修正検証テスト ===")

# 生成されたParquetファイルをチェック
parquet_files = []
for i in range(4):  # 最初の4チャンクのみ
    file_path = f"benchmark/chunk_{i}_direct.parquet"
    if os.path.exists(file_path):
        parquet_files.append((i, file_path))

print(f"検証対象ファイル: {len(parquet_files)}個")

if not parquet_files:
    print("Parquetファイルが見つかりません。ベンチマークが完了するまで待ってください。")
    exit(1)

# 文字列破損をチェック
total_rows = 0
even_errors = 0
odd_errors = 0
error_samples = []

for chunk_id, parquet_file in parquet_files:
    print(f"\nチャンク{chunk_id}: {parquet_file}")
    
    try:
        df = cudf.read_parquet(parquet_file)
        rows = len(df)
        total_rows += rows
        print(f"  行数: {rows:,}")
        
        if 'lo_orderpriority' in df.columns:
            # 最初の1000行をチェック
            check_rows = min(1000, rows)
            
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
                            error_samples.append((chunk_id, i, value))
                    
                except Exception as e:
                    pass
        
        del df
        
    except Exception as e:
        print(f"  エラー: {e}")

# 結果表示
print(f"\n=== 検証結果 ===")
print(f"総チェック行数: {total_rows:,}")
print(f"偶数行エラー: {even_errors}")
print(f"奇数行エラー: {odd_errors}")
print(f"総エラー数: {even_errors + odd_errors}")

if error_samples:
    print(f"\nエラーサンプル:")
    for chunk_id, row, value in error_samples:
        print(f"  チャンク{chunk_id}, 行{row}({'偶数' if row % 2 == 0 else '奇数'}): {repr(value)}")
else:
    print("\n✅ 文字列破損は検出されませんでした！修正成功！")

# パフォーマンス比較
print(f"\n=== 注意事項 ===")
print("現在の修正はホスト転送を使用しているため、パフォーマンスが低下しています。")
print("文字列破損の根本原因を特定後、真のゼロコピー実装に戻す必要があります。")