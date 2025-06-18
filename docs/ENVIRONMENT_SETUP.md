# GPUPGParser 環境設定ガイド

## 必須環境設定

GPUPGParserプロジェクトを正常に動作させるために、以下の設定が必要です。

### 1. Conda環境のアクティベーション

**最重要**: 必ずCUDF開発環境をアクティベートしてください：

```bash
conda activate cudf_dev
```

⚠️ **警告**: この手順を忘れると `ModuleNotFoundError: No module named 'cudf'` エラーが発生します。

### 2. PYTHONPATH設定

プロジェクトのソースコードを正しくインポートするため：

```bash
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
```

### PostgreSQL接続設定

PostgreSQLデータベースへの接続用設定：

```bash
export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'
```

## 設定手順

### 1. 一時的な設定（現在のセッションのみ）

```bash
# 1. Conda環境をアクティベート
conda activate cudf_dev

# 2. 環境変数を設定
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'

# 3. 設定確認
echo "Conda環境: $CONDA_DEFAULT_ENV"
echo "PYTHONPATH: $PYTHONPATH"
echo "GPUPASER_PG_DSN: $GPUPASER_PG_DSN"
```

### 2. 永続的な設定

#### Bashの場合（推奨）

```bash
# ~/.bashrcに追加
echo 'export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser' >> ~/.bashrc
echo 'export GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"' >> ~/.bashrc

# 設定を読み込み
source ~/.bashrc
```

#### 一時設定ファイルの使用

プロジェクトルートに設定ファイルを作成：

```bash
# setup_env.sh を作成
cat > setup_env.sh << 'EOF'
#!/bin/bash
# Conda環境のアクティベーション
conda activate cudf_dev

# 環境変数の設定
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'

echo "✓ 環境設定が完了しました"
echo "Conda環境: $CONDA_DEFAULT_ENV"
echo "PYTHONPATH: $PYTHONPATH"
echo "GPUPASER_PG_DSN: $GPUPASER_PG_DSN"
EOF

# 実行権限を付与
chmod +x setup_env.sh

# 使用方法
source ./setup_env.sh
```

## 設定確認

環境変数が正しく設定されているか確認：

```bash
# Pythonモジュールのインポートテスト
python -c "
try:
    from src.cuda_kernels.heap_page_parser import parse_heap_pages_to_tuples
    print('✓ ヒープページパーサーのインポートに成功')
except ImportError as e:
    print(f'✗ インポートエラー: {e}')
"
```

## トラブルシューティング

### よくある問題

1. **ModuleNotFoundError: No module named 'cudf'**
   - Conda環境 `cudf_dev` がアクティベートされていない
   - 必要なパッケージがインストールされていない

2. **ModuleNotFoundError: No module named 'src'**
   - `PYTHONPATH`が正しく設定されていない
   - プロジェクトディレクトリからPythonを実行していない

3. **PostgreSQL接続エラー**
   - `GPUPASER_PG_DSN`の設定確認
   - PostgreSQLサービスの動作確認
   - 認証情報の確認

### 解決方法

```bash
# 1. 現在のディレクトリ確認
pwd
# 出力例: /home/ubuntu/gpupgparser

# 2. 環境変数確認
env | grep PYTHONPATH
env | grep GPUPASER

# 3. Pythonパス確認
python -c "import sys; print('\n'.join(sys.path))"
```

## 上級設定

### 開発環境用追加設定

```bash
# デバッグ用環境変数
export CUDA_VISIBLE_DEVICES=0
export NUMBA_ENABLE_CUDASIM=0
export NUMBA_CUDA_DEBUGINFO=1

# パフォーマンス調整
export MALLOC_TRIM_THRESHOLD_=0
export OMP_NUM_THREADS=1
```

### テスト実行

```bash
# 基本テスト
python test/test_heap_page_parser.py

# 全テスト実行
python -m pytest test/

# 特定テストのみ
python -m pytest test/test_heap_page_parser.py -v
```

## 注意事項

⚠ **重要な警告**

1. **Conda環境 `cudf_dev` のアクティベーションは最重要**です。これを忘れると cudf モジュールエラーが発生します。
2. `PYTHONPATH`設定は必須です。設定なしではプロジェクトのモジュールをインポートできません。
3. PostgreSQL接続設定は実際の環境に合わせて調整してください。
4. **本設定を上官および関係者に共有し、チーム全体で統一した環境を構築してください。**

## 設定確認チェックリスト

- [ ] **Conda環境 `cudf_dev` がアクティベートされている**
- [ ] `PYTHONPATH`が正しく設定されている
- [ ] `GPUPASER_PG_DSN`が設定されている
- [ ] プロジェクトのモジュールがインポートできる
- [ ] CUDAが利用可能である
- [ ] PostgreSQLに接続できる
- [ ] テストが正常に実行できる

---

**更新履歴**
- 2025/06/14: 初版作成 - 必須環境変数設定を文書化
- 2025/06/14: Conda環境設定を追加 - cudf_dev環境のアクティベーション手順を明記