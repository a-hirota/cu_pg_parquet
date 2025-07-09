#!/bin/bash
# 256MB境界での欠落を再現するテスト

echo "=== 256MB境界欠落再現テスト ==="
echo

# 環境変数設定
export GPUPASER_PG_DSN="host=localhost dbname=postgres user=postgres"
export TABLE_NAME="customer"
export GPUPGPARSER_TEST_MODE="1"

# 1チャンクでテスト実行
echo "1. customerテーブルを1チャンクでエクスポート..."
python docs/benchmark/benchmark_rust_gpu_direct.py --table customer --chunks 1 --parallel 16 2>&1 | grep -E "(検出行数|欠落|PostgreSQL行数)"

echo
echo "2. 生成されたParquetファイルの確認..."
ls -la output/customer_chunk_0_queue.parquet

echo
echo "3. Parquetファイルから行数を確認..."
python -c "
import pandas as pd
df = pd.read_parquet('output/customer_chunk_0_queue.parquet')
print(f'Parquet行数: {len(df):,}')
print(f'期待行数: 12,030,000')
print(f'欠落: {12030000 - len(df):,} 行')
"

echo
echo "4. 256MB境界付近のデータを確認..."
python -c "
import pandas as pd
df = pd.read_parquet('output/customer_chunk_0_queue.parquet')

# _row_positionから256MB境界付近のデータを探す
MB_256 = 256 * 1024 * 1024
boundary_data = df[(df['_row_position'] > MB_256 - 1000) & (df['_row_position'] < MB_256 + 1000)]

print(f'256MB境界（{MB_256:,}）付近のデータ: {len(boundary_data)}行')
if len(boundary_data) > 0:
    print(boundary_data[['c_custkey', '_row_position']].head(10))
"