# Ray MultiGPU Parquetパーサーガイド

このドキュメントでは、複数のGPUを使用してPostgreSQLのデータを高速にParquetファイルに変換する方法について説明します。Rayフレームワークを使って処理をGPUに分散することで、処理速度を大幅に向上させることができます。

## 概要

gpuParserのマルチGPU機能は以下の特徴を持っています：

- **Rayフレームワーク**を使用した複数GPUへの処理分散
- チャンク単位でのデータ分割と並列処理
- 各GPUでのバイナリデータのGPUデコードとParquet出力
- シンプルなコマンドラインインターフェース

## 前提条件

- 複数のGPUを搭載したマシン (単一GPUでも動作可)
- Ray, CUDA, cuDF がインストールされていること
- PostgreSQLのデータベースに接続できること

## インストール

Rayのインストール:

```bash
pip install ray[default]
```

## 使用方法

### コマンドライン実行

シェルスクリプト `run_ray_multi_gpu.sh` を使用すると、簡単にマルチGPU処理を実行できます:

```bash
./examples/run_ray_multi_gpu.sh -t <テーブル名> -r <処理行数> -g <GPU数> -o <出力ディレクトリ>
```

#### 引数

- `-t, --table`: 処理するPostgreSQLテーブル名（必須）
- `-r, --rows`: 処理する総行数（デフォルト: 10000）
- `-g, --gpus`: 使用するGPU数（デフォルト: システムで利用可能な最大数）
- `-o, --output`: 出力ディレクトリパス（デフォルト: `./ray_output`）
- `-d, --db`: データベース名（デフォルト: `postgres`）
- `-u, --user`: データベースユーザー名（デフォルト: `postgres`）
- `-p, --password`: データベースパスワード（デフォルト: `postgres`）
- `-h, --host`: データベースホスト（デフォルト: `localhost`）

### 例

100,000行のcustomerテーブルデータを2つのGPUで処理し、`./ray_output`ディレクトリに出力:

```bash
./examples/run_ray_multi_gpu.sh -t customer -r 100000 -g 2 -o ./ray_output
```

## 内部動作

1. **チャンク分割**: 指定された総行数をGPU数で分割し、均等なチャンクを作成
2. **Rayタスク**: 各チャンクを処理するためのRayタスクを作成し、各GPUに割り当て
3. **並列処理**: 各GPUで並行して処理を実行
   - PostgreSQLからのデータ取得（バイナリ形式）
   - GPUバッファへのデータ転送
   - GPUでのバイナリデータパース
   - cudFデータフレームへの変換
   - Parquetファイルへの保存
4. **結果統合**: 処理が完了すると、チャンクごとのParquetファイルが生成

## パフォーマンス

処理速度は以下の要因に依存します：

- **GPUの数と性能**: より多くのGPUを使用することでスループットが向上
- **チャンクサイズ**: 大きすぎるとGPUメモリ不足、小さすぎるとオーバーヘッドが増加
- **PostgreSQLの応答速度**: データベースの読み取り速度がボトルネックになる可能性
- **GPUメモリ**: 利用可能なGPUメモリがチャンクサイズの上限を決定

### 実測値例

| 行数    | GPU数 | 処理時間 | スループット   |
|--------|------|--------|------------|
| 10,000  | 2    | 3.01秒  | 3,321行/秒  |
| 100,000 | 2    | 6.01秒  | 16,642行/秒 |

大きなデータセットほど効率が向上し、スループットが増加する傾向があります。

## 制限事項と注意点

- **GPU数の上限**: システムに搭載されているGPU数以上を指定することはできません
- **メモリ制限**: 各GPUで処理できるチャンクサイズはGPUメモリに依存します
- **Parquetファイル分割**: 現時点では各チャンクが個別のParquetファイルとして出力されます
- **エラー処理**: 一部のGPU処理が失敗した場合、他のGPUの結果は保存されます

## トラブルシューティング

1. **メモリ不足エラー**: チャンクサイズを小さくするか、より多くのGPUに分散してください
2. **接続エラー**: PostgreSQLの接続パラメータを確認してください
3. **GPU検出失敗**: CUDAドライバーが正しくインストールされているか確認してください

## 高度な使用例

### 特定GPUの使用

環境変数 `CUDA_VISIBLE_DEVICES` を使用して特定のGPUを指定できます：

```bash
CUDA_VISIBLE_DEVICES=0,2 ./examples/run_ray_multi_gpu.sh -t customer -r 100000 -g 2
```

これにより、システム上のGPU 0と2のみを使用して処理が実行されます。

### 複数の出力ファイルの結合

複数のParquetファイルを結合するには、以下のPythonスクリプトを使用できます：

```python
import pandas as pd
import glob

# 出力ディレクトリ内のすべてのParquetファイルのパスを取得
files = glob.glob('./ray_output/customer_chunk_*.parquet')

# 各ファイルを読み込んでリストに格納
dfs = [pd.read_parquet(f) for f in files]

# すべてのDataFrameを連結
combined_df = pd.concat(dfs)

# 結合したデータを1つのParquetファイルとして保存
combined_df.to_parquet('./ray_output/customer_combined.parquet')
