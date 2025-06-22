# GPUPGParser Rust実装

PostgreSQLのCOPY BINARYデータをGPUに直接転送するRust実装です。

## 機能

- PostgreSQL COPY BINARYデータの高速取得
- rust-cudaによるGPUメモリ直接転送
- Arrow形式バッファのGPU上での構築
- PyO3によるPythonバインディング
- 既存のpylibcudfとのゼロコピー統合

## ビルド手順

### 前提条件

1. Rust (1.70以上)
2. CUDA Toolkit (11.0以上)
3. Python環境 (cudf_dev conda環境)

### ビルド

```bash
# conda環境のアクティベート
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cudf_dev

# maturinのインストール（未インストールの場合）
pip install maturin

# Rustディレクトリに移動
cd rust

# 開発ビルド
maturin develop

# リリースビルド
maturin develop --release
```

### テスト実行

```bash
# プロジェクトルートに戻る
cd ..

# 環境変数設定
export GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"

# テスト実行
python test_rust_integration.py
```

## 使用方法

### 1. PostgreSQLから直接GPU転送

```python
from src.rust_integration import PostgresGPUReader

# リーダー初期化
reader = PostgresGPUReader()

# 文字列カラムをGPUに転送
query = "COPY (SELECT name FROM mytable) TO STDOUT WITH BINARY"
gpu_buffers = reader.fetch_string_column_to_gpu(query, column_index=0)

# cuDF Series作成（ゼロコピー）
series = reader.create_cudf_series_from_gpu_buffers(gpu_buffers)
```

### 2. Rust文字列ビルダー

```python
from src.rust_integration import RustStringBuilder

# ビルダー作成
builder = RustStringBuilder()

# 文字列追加
builder.add_string(b"Hello")
builder.add_string(b"World")

# cuDF Series構築
series = builder.build_cudf_series()
```

### 3. 既存Numba実装との統合

```python
# バイナリデータをGPUに転送
gpu_info = reader.transfer_binary_to_gpu(binary_data)

# 既存のNumbaカーネルで処理
# device_ptr = gpu_info['device_ptr']
```

## アーキテクチャ

```
PostgreSQL
    ↓ (COPY BINARY)
Rust (tokio-postgres)
    ↓ (バイナリ解析)
Arrow形式バッファ構築
    ↓ (rust-cuda)
GPU Memory
    ↓ (PyO3 FFI)
Python (pylibcudf)
    ↓ (ゼロコピー)
cuDF DataFrame
```

## トラブルシューティング

### ImportError: gpupgparser_rust

```bash
cd rust && maturin develop
```

### CUDA関連エラー

```bash
# CUDAパス確認
echo $CUDA_PATH
# 未設定の場合
export CUDA_PATH=/usr/local/cuda
```

### リンクエラー

```bash
# ライブラリパス追加
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```