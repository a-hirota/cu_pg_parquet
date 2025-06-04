# Parquet出力機能の使い方

PostgreSQLデータをGPUで処理し、Parquetフォーマットで保存するための機能について説明します。この機能を使用することで、PostgreSQLからのデータを高速に処理し、分析ツールで利用可能な形式で永続化できます。

## Parquet出力の主な特徴

- **高速な処理**: GPUでPostgreSQLバイナリデータを直接処理し、Parquetに変換
- **効率的なストレージ**: 列指向フォーマットによるデータ圧縮と効率的なストレージ使用
- **互換性**: Apache Arrow、PySpark、Dask、pandas等の分析ツールとの互換性
- **チャンク処理**: 大規模データセットを複数チャンクに分割して効率的に処理

## 基本的な使い方

### Pythonからの利用

```python
from gpupaser.main import load_table_optimized

# テーブル全体をParquetファイルに出力
load_table_optimized('customer', parquet_output='customer_full.parquet')

# 行数を制限して出力
load_table_optimized('customer', limit=100000, parquet_output='customer_100k.parquet')
```

### コマンドラインからの利用

```bash
# 基本的な使い方
python examples/process_large_dataset.py --table=customer --parquet=customer_output.parquet

# 行数を制限して出力
python examples/process_large_dataset.py --table=customer --rows=100000 --parquet=customer_100k.parquet

# チャンクサイズを指定して処理（デフォルトは65535行）
python examples/process_large_dataset.py --table=customer --rows=200000 --chunk-size=50000 --parquet=customer_200k.parquet
```

## 高度な使い方

### テーブル分析

処理前にテーブルを分析して、最適なチャンクサイズや推定処理時間を確認できます：

```bash
python examples/process_large_dataset.py --table=customer --analyze
```

出力例：
```
=== テーブル customer の分析 ===
カラム情報:
  c_custkey: numeric
  c_name: character varying (長さ: 25)
  c_address: character varying (長さ: 25)
  ...

===== 分析結果 =====
テーブル行数: 6,000,000行
1行あたりサイズ: 120バイト
推定テーブル全体サイズ: 686.65MB
最適チャンクサイズ: 65,535行
推奨チャンク数: 92
推定処理時間: 10分 0.0秒
```

### マルチGPU処理

複数GPUを使用して処理を並列化することも可能です：

```bash
# マルチGPUでの処理（シェルスクリプトを使用）
./examples/run_multigpu_parquet.sh customer 1000000 customer_multigpu.parquet
```

## Parquetファイルの検証と利用

出力されたParquetファイルは、以下のツールで検証・利用できます：

### cuDFでの読み込み

```python
import cudf

# Parquetファイルを読み込む
df = cudf.read_parquet('customer_output.parquet')

# 内容を確認
print(df.head())
print(f"行数: {len(df)}")
```

### pandasでの読み込み

```python
import pandas as pd

# Parquetファイルを読み込む
df = pd.read_parquet('customer_output.parquet')

# 内容を確認
print(df.describe())
```

### PySparkでの利用

```python
from pyspark.sql import SparkSession

# SparkSessionの初期化
spark = SparkSession.builder.appName("ParquetReader").getOrCreate()

# Parquetファイルを読み込む
df = spark.read.parquet('customer_output.parquet')

# 内容を確認
df.printSchema()
df.show(5)
```

## パフォーマンス最適化のヒント

1. **適切なチャンクサイズの選択**:
   - 大規模テーブルでは、`--analyze`オプションで推奨チャンクサイズを確認する
   - GPUメモリ不足エラーが発生する場合は、チャンクサイズを小さくする

2. **文字列カラムの最適化**:
   - 文字列カラムは固定長バッファを使用するため、必要なカラムのみを選択する
   - 可能であれば、長い文字列カラムは別クエリで抽出する

3. **マルチGPU処理の活用**:
   - 複数GPUが利用可能な環境では、マルチGPU処理を使用して高速化
   - GPU間の負荷分散を適切に設定する

## トラブルシューティング

### エラー: "CUDA_ERROR_ILLEGAL_ADDRESS"

GPUメモリ不足が原因の可能性が高いです。以下の対策を試してください：

1. チャンクサイズを小さくする: `--chunk-size=30000`
2. 処理行数を減らす: `--rows=50000`
3. 他のGPUプロセスを終了させる

### エラー: "too many values to unpack (expected 3)"

最新バージョンでは修正されています。それでも発生する場合は、以下を確認してください：

1. ソースコードが最新であることを確認
2. バージョン互換性の問題：Numba 0.56以降が必要

### 出力ファイルが破損している場合

1. CPUフォールバック処理が発生していないか確認
2. ディスク容量が十分か確認
3. 小さなサンプルで動作確認後、行数を徐々に増やす
