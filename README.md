# pgCuparquet

PostgreSQLのテーブルデータを高速にParquet形式に変換するGPU加速ツールです。

## 特徴

- **超高速変換**: GPUを活用してPostgreSQLのバイナリデータを直接Parquetに変換
- **並列処理**: Producer-Consumerパターンでデータ転送とGPU処理を並列実行
- **大規模データ対応**: チャンク分割により数百GBのテーブルも処理可能
- **ゼロコピー最適化**: CPU経由のデータコピーを排除し、最大限の性能を発揮

## 必要条件

### ハードウェア
- NVIDIA GPU（計算能力6.0以上）
- 16GB以上のGPUメモリ推奨
- 十分なシステムメモリ（データサイズの2倍以上推奨）

### ソフトウェア
- CUDA 12.0以上
- Python 3.8以上
- PostgreSQL 12以上
- RAPIDS/cuDF環境

## インストール

### 1. RAPIDS環境のセットアップ

```bash
# Minicondaのインストール（既にある場合はスキップ）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# RAPIDS環境の作成
conda create -n rapids-24.12 -c rapidsai -c conda-forge -c nvidia \
    rapids=24.12 python=3.11 cuda-version=12.0

conda activate rapids-24.12
```

### 2. 追加パッケージのインストール

```bash
pip install psycopg kvikio
```

### 3. Rust拡張のビルド

```bash
cd rust
pip install maturin
maturin develop --release
cd ..
```

## 使い方

### 基本的な使用方法

```bash
# 環境変数の設定
export GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"

# デフォルト設定で実行
python pgCuparquet.py

# カスタム設定で実行
python pgCuparquet.py --table mytable --parallel 8 --chunks 4
```

### コマンドラインオプション

- `--table TABLE`: 変換対象のテーブル名（デフォルト: lineorder）
- `--parallel N`: PostgreSQL並列接続数（デフォルト: 16）
- `--chunks N`: データ分割チャンク数（デフォルト: 8）

### 環境変数

- `GPUPASER_PG_DSN`: PostgreSQL接続文字列（必須）
- `RUST_PARALLEL_CONNECTIONS`: Rust側の並列接続数
- `KVIKIO_COMPAT_MODE`: GPUDirectが使えない環境で`on`に設定

## パフォーマンスチューニング

### チャンク数の調整

GPUメモリサイズに応じてチャンク数を調整してください：

- 16GB GPU: 8-16チャンク
- 24GB GPU: 4-8チャンク
- 40GB以上のGPU: 2-4チャンク

### 並列数の調整

CPUコア数とネットワーク帯域に応じて調整：

- 一般的な設定: CPUコア数の1/2から同数
- I/O待機が多い場合: CPUコア数の2倍まで増やす

## トラブルシューティング

### CUDA out of memory エラー

チャンク数を増やしてメモリ使用量を削減：

```bash
python pgCuparquet.py --chunks 16
```

### kvikio関連のエラー

互換モードを有効化：

```bash
export KVIKIO_COMPAT_MODE=on
python pgCuparquet.py
```

### PostgreSQL接続エラー

接続文字列を確認：

```bash
export GPUPASER_PG_DSN="dbname=mydb user=myuser host=localhost port=5432 password=mypass"
```

## 性能例

lineorderテーブル（52.86GB、1.19億行）の変換：

- 総実行時間: 46.92秒
- スループット: 1.13 GB/秒
- PostgreSQL読み取り: 1.24 GB/秒
- GPU処理: 2.09 GB/秒

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能リクエストは[GitHubのIssue](https://github.com/yourusername/pgCuparquet)でお願いします。