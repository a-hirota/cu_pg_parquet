# gpuPaser 使用ガイド

## 概要

gpuPaserは、PostgreSQLからバイナリデータを直接取得し、GPUで高速に処理してcuDFデータフレームに変換するツールです。従来のCPUでのデータ変換処理ボトルネックを大幅に削減し、特に大規模データセットで高いパフォーマンスを発揮します。

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/username/gpuPaser.git
cd gpuPaser

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 基本的な使い方

### Pythonからの利用

```python
from gpupaser.main import load_table_optimized

# テーブルをGPUで処理
results = load_table_optimized(
    table_name="your_table",  # テーブル名（必須）
    limit=100000,             # 取得する最大行数（オプション）
    parquet_output="output.parquet"  # Parquet出力ファイルパス（オプション）
)

# 結果はPythonの辞書で返される
# カラム名をキー、データ配列を値として保持
print(results["column_name"])
```

### コマンドラインからの利用

```bash
# 基本的な使用法
python -m gpupaser.main --table your_table

# 行数を制限
python -m gpupaser.main --table your_table --limit 10000

# Parquetファイルとして出力
python -m gpupaser.main --table your_table --parquet output.parquet
```

## Parquet出力機能

65,535行を超えるデータセットを処理する場合など、結果をParquetファイルとして保存することができます。各チャンクは個別のレコードバッチとして処理され、1つの統合されたParquetファイルとして保存されます。

### Pythonからの利用

```python
from gpupaser.main import load_table_optimized

# 100,000行のデータをParquetファイルとして保存
results = load_table_optimized(
    table_name="large_table",
    parquet_output="large_table_output.parquet"
)
```

### 専用テストスクリプトの利用

大規模データセットのテスト用に専用スクリプトが用意されています：

```bash
python test_postgres_binary_cuda_parquet.py --table large_table --output large_table_output.parquet
```

このスクリプトは：
1. 指定されたテーブルからデータを取得
2. チャンク単位でGPUを使って処理
3. 各チャンクをParquetのレコードバッチとして出力
4. cuDFを使って結果のParquetファイルを読み込み、最初と最後の5行を表示して検証

### cuDFでの読み込み

生成されたParquetファイルは、cuDFで簡単に読み込むことができます：

```python
import cudf

# Parquetファイルを読み込む
df = cudf.read_parquet("output.parquet")

# 最初の5行を表示
print(df.head())

# 最後の5行を表示
print(df.tail())
```

## 高度な使用法

### PgGpuProcessorクラスの直接利用

より詳細な制御が必要な場合は、`PgGpuProcessor`クラスを直接利用できます：

```python
from gpupaser.main import PgGpuProcessor

# プロセッサの初期化
processor = PgGpuProcessor(
    dbname="your_db",
    user="postgres",
    password="postgres",
    host="localhost",
    parquet_output="output.parquet"  # Parquet出力を有効化
)

try:
    # テーブル処理
    results = processor.process_table("your_table", limit=10000)
finally:
    # 必ずリソースを解放
    processor.close()
```

## パフォーマンスのヒント

- **チャンクサイズ**: デフォルトでは最大65,535行/チャンクで処理されますが、GPUメモリに基づいて自動調整されます
- **GPUメモリ**: 大規模データセットを処理する場合は、十分なGPUメモリを確保してください
- **Parquet出力**: データサイズが大きい場合、メモリ内で全結果を保持する代わりにParquet出力を利用すると効率的です
