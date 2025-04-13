# PostgreSQLデータのGPU直接デシリアライズとcuDF格納パイプライン

PostgreSQLから大量のデータを取得し、GPUメモリを最大限に活用してGPU上のデータフレーム（RAPIDS cuDF）に直接格納するパイプラインです。

## 特徴

- **GPUメモリに基づく動的チャンクサイズ計算**: 利用可能なGPUメモリ量から自動的に最適なチャンクサイズを計算
- **PostgreSQLバイナリ形式の直接パース**: テキスト変換を介さずに効率的なバイナリデータ処理
- **CUDAを活用した並列デコード**: GPUの並列処理能力を最大限に活用
- **メモリ効率の最適化**: リソース管理とクリーンアップの強化

## セットアップ

### 必要条件

- Python 3.8以上
- CUDA Toolkit 11.0以上
- PostgreSQL 12以上
- Numba 0.53.0以上
- CuPy 9.0.0以上
- NumPy 1.20.0以上
- psycopg2 2.9.0以上

### インストール

```bash
git clone https://github.com/username/gpuPaser.git
cd gpuPaser
pip install -e .
```

## 基本的な使い方

```python
from gpupaser.main import load_table_optimized

# テーブル全体を処理
results = load_table_optimized('mytable')

# 行数を制限して処理
results = load_table_optimized('mytable', limit=50000)

# 結果の利用（例：NumPy配列として取得）
first_column_data = results['column_name']
```

## GPUメモリを最大限活用する方法

このパイプラインは、利用可能なGPUメモリを自動的に検出し、最適なチャンクサイズを計算します。基本的には追加の設定は不要ですが、より詳細な調整が必要な場合は以下のパラメータを調整できます。

### チャンクサイズの手動調整

`gpupaser/utils.py`の`ChunkConfig`クラスを修正することで、チャンクサイズを手動で調整できます：

```python
# 例：チャンクサイズを30,000行に固定する場合
chunk_config = ChunkConfig(total_rows, 30000)
```

### メモリ安全マージンの調整

`gpupaser/memory_manager.py`の`calculate_optimal_chunk_size`メソッド内の安全マージンを調整できます：

```python
# デフォルトでは利用可能なメモリの80%を使用（0.8）
usable_memory = free_memory * 0.8  # 安全マージンを調整（0.7など）
```

## テストの実行

基本的なテスト実行：

```bash
python test_postgres_binary_cuda_parser_module.py
```

増分テスト（行数を徐々に増やして最大処理可能行数を探す）：

```bash
python test_postgres_binary_cuda_parser_module.py --incremental
```

## テストコードの修正点

現在のパイプラインでは、1チャンクの最大行数は65,535行に制限されています（CUDAのブロック数制限による）。複数チャンク処理を改善するには、以下の修正が必要です：

1. **バイナリパーサーの改善**: 
   `gpupaser/binary_parser.py`の`parse_chunk`メソッドで、行スキップ処理のロジックを修正する必要があります。

2. **テストコードで処理行数を調整**:
   `test_postgres_binary_cuda_parser_module.py`では、以下のように修正するとより詳細なテストが可能です：

```python
# 例：60,000行（単一チャンク）のテスト
test_rows = 60000  # 65,535未満で設定

# 例：増分テスト - 徐々に行数を増やす
def test_increasing_rows():
    start_rows = 10000    # より小さい値から開始
    increment = 10000     # 増分を小さく
    max_attempt = 200000  # 必要に応じて調整
```

## パフォーマンス最適化のヒント

1. **カラム数の制限**: 処理するカラム数が多い場合、特に文字列カラムが多いとGPUメモリ使用量が増加します。必要なカラムだけを選択するSQLクエリを使用することで、メモリ使用量を削減できます。

2. **文字列長の最適化**: 文字列カラムは固定長バッファを使用するため、実際のデータよりも大きなメモリを消費する可能性があります。文字列カラムの長さをデータに合わせて適切に設定することで、メモリ使用効率が向上します。

3. **チャンク処理の並列化**: 大規模データセットをより効率的に処理するためには、マルチGPUシステムでの並列処理を検討できます。

## トラブルシューティング

### "CUDA_ERROR_ILLEGAL_ADDRESS" エラーが発生する場合

このエラーは主にGPUメモリ不足または無効なメモリアクセスが原因です。以下の対策を試してください：

1. 処理行数を減らす
2. GPUメモリ使用率を確認（`nvidia-smi`コマンド）
3. システム上で他のGPUプロセスを終了させる

### "incompatible shape" エラーが発生する場合

このエラーは、パース処理と出力バッファのサイズに不一致がある場合に発生します。通常は以下の対策で解決します：

1. バイナリパーサーの行数制限を確認
2. 複数チャンク処理ロジックの見直し
