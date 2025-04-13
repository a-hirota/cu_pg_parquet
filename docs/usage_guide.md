# gpuPaser 詳細使用ガイド

このドキュメントでは、gpuPaserの詳細な使用方法と、特にGPUメモリ使用量を最大化するための設定について説明します。

## 目次

1. [アーキテクチャ概要](#アーキテクチャ概要)
2. [GPUメモリ使用量の最適化](#gpuメモリ使用量の最適化)
3. [テストコードの修正方法](#テストコードの修正方法)
4. [高度な設定](#高度な設定)
5. [実装の詳細](#実装の詳細)

## アーキテクチャ概要

gpuPaserは以下の主要モジュールで構成されています：

1. **pg_connector.py**: PostgreSQLデータベースとの接続とバイナリデータの取得
2. **binary_parser.py**: PostgreSQLバイナリ形式データのパース処理
3. **memory_manager.py**: GPUメモリ割り当てと管理
4. **gpu_decoder.py**: GPUでのデータデコード処理
5. **output_handler.py**: デコード結果の整形と出力
6. **utils.py**: ユーティリティ関数とデータ構造
7. **main.py**: 全体の処理フローとインターフェース

データフローは以下の通りです：

```
PostgreSQL → バイナリデータ取得 → パース処理 → GPUメモリ割り当て → 
GPU並列デコード → 結果整形 → Pythonオブジェクト
```

各ステップで、GPUメモリ使用量に基づいて適切なチャンクサイズを計算し、メモリオーバーフローを防ぎながら最大のパフォーマンスを実現します。

## GPUメモリ使用量の最適化

### 1. メモリ使用量計算の仕組み

`memory_manager.py`の`calculate_optimal_chunk_size`メソッドでは、以下の手順でメモリ使用量を計算しています：

```python
# 1. 利用可能なGPUメモリを取得
free_memory = self.get_available_gpu_memory()

# 2. 安全マージンを設定（デフォルト80%）
usable_memory = free_memory * 0.8

# 3. 1行あたりのメモリ使用量を計算
mem_per_row = 0
for col in columns:
    if get_column_type(col.type) <= 1:  # 数値型
        mem_per_row += 8  # 8バイト
    else:  # 文字列型
        mem_per_row += get_column_length(col.type, col.length)

# 4. オーバーヘッドを考慮
mem_per_row *= 1.2

# 5. 最大行数を計算
max_rows = int(usable_memory / mem_per_row)

# 6. ハードウェア制限を適用
max_rows = min(65535, max_rows, total_rows)
```

### 2. メモリ使用量を最大化するための調整

#### 安全マージンの調整

システムの安定性に問題がなければ、安全マージンを小さくすることでより多くのデータを処理できます：

```python
# memory_manager.py の calculate_optimal_chunk_size メソッド内
usable_memory = free_memory * 0.9  # 90%に増加
```

#### オーバーヘッド係数の調整

メモリオーバーヘッドがシステムによって異なる場合、この値を調整できます：

```python
# memory_manager.py の calculate_optimal_chunk_size メソッド内
mem_per_row *= 1.1  # 10%に減少
```

#### 文字列バッファサイズの最適化

文字列カラムのバッファサイズをより正確に設定することで、メモリ使用効率が向上します：

```python
# utils.py の get_column_length メソッド内
def get_column_length(type_name: str, length: Optional[int]) -> int:
    if type_name.startswith('character'):
        # 平均的な文字列長に基づいて調整
        return int(length) if length else 64  # デフォルト値を小さく
```

## テストコードの修正方法

### テストスクリプトの構造

`test_postgres_binary_cuda_parser_module.py`は以下の主要部分で構成されています：

1. 基本テスト関数 `run_tests()`
2. 増分テスト関数 `test_increasing_rows()`
3. メイン関数 `if __name__ == "__main__"`

### 70,000行を処理するための修正

現在のパイプラインでは65,535行までは正常に処理できますが、70,000行を完全に処理するには以下の修正が必要です：

#### 1. テスト行数の調整

まず、テスト行数を調整して処理可能範囲内に収めます：

```python
# テスト行数を65,000に設定（安全範囲内）
test_rows = 65000
```

または、増分テスト関数で段階的に増やしていきます：

```python
def test_increasing_rows():
    start_rows = 50000    # 開始行数
    increment = 5000      # 増分（小さくすることで詳細なテスト）
    max_attempt = 100000  # 最大試行行数
```

#### 2. 複数チャンク処理の有効化

70,000行を処理するには、複数回のチャンク処理が必要です。修正点は以下の通りです：

```python
# 複数チャンク処理のテスト
# 例: test_postgres_binary_cuda_parser_module.py に以下のようなテスト関数を追加

def test_multi_chunk():
    print("\n=== 複数チャンク処理テスト ===")
    chunk_sizes = [30000, 40000]  # 合計70,000行
    
    all_results = {}
    
    for i, size in enumerate(chunk_sizes):
        print(f"\n--- チャンク {i+1}: {size}行 ---")
        start_time = time.time()
        
        # 開始行を計算（前のチャンクの累積サイズ）
        start_row = sum(chunk_sizes[:i])
        end_row = start_row + size
        
        # 1チャンクずつ処理
        results = load_table_optimized('customer', 
                                      start_row=start_row, 
                                      limit=size)
        
        # 結果を集約
        if not all_results:
            all_results = results
        else:
            for col_name, data in results.items():
                all_results[col_name] = np.concatenate([all_results[col_name], data])
        
        print(f"チャンク {i+1} 処理時間: {time.time() - start_time:.3f}秒")
    
    # 最終結果表示
    print(f"\n全 {sum(chunk_sizes)} 行の処理が完了")
    return all_results
```

ただし、この関数が動作するためには `load_table_optimized` 関数に `start_row` パラメータを追加する必要があります：

```python
# main.py の load_table_optimized 関数を修正
def load_table_optimized(table_name: str, limit: Optional[int] = None, start_row: int = 0):
    processor = PgGpuProcessor()
    try:
        return processor.process_table(table_name, limit, start_row)
    finally:
        processor.close()
```

また、`process_table` メソッドも同様に修正が必要です：

```python
def process_table(self, table_name: str, limit: Optional[int] = None, start_row: int = 0):
    # 既存コード...
    
    # processed_rows の初期値を start_row に設定
    processed_rows = start_row
    
    # 残りは同じ...
```

#### 3. バイナリパーサーの改善

複数チャンク間での正確なデータ解析には、バイナリパーサーの修正も必要です。主な改善点：

- より正確な行スキップ処理
- チャンク境界の適切な処理
- バイナリデータオフセットの再計算

## 高度な設定

### データ型ごとのメモリ計算調整

データ型ごとにより正確なメモリ使用量を計算できます：

```python
# memory_manager.py に以下のような関数を追加
def get_datatype_memory_size(col_type: str) -> int:
    if col_type == 'integer':
        return 4
    elif col_type == 'bigint':
        return 8
    elif col_type == 'numeric':
        return 16  # 数値精度によって異なる
    elif col_type.startswith('character'):
        # 実際の平均長さに基づいて調整
        return int(col_type.split('(')[1].split(')')[0]) if '(' in col_type else 256
    # その他のデータ型
    return 8  # デフォルト
```

### バッファリング戦略の調整

ダブルバッファリングや非同期転送を最適化するため、バッファサイズとタイミングを調整できます：

```python
# 既存のバッファ初期化コードの拡張
# memory_manager.py の initialize_device_buffers メソッド内

# バッファサイズの計算（オプション: 予備バッファ確保）
buffer_multiplier = 1.1  # 10%余分に確保
int_buffer_size = int(chunk_size * num_int_cols * buffer_multiplier)
```

## 実装の詳細

### 主要クラスとその機能

- **PgGpuProcessor**: 全体の処理フローを制御
- **GPUMemoryManager**: GPUメモリの割り当てと管理
- **BinaryDataParser**: バイナリデータのパース
- **GPUDecoder**: GPU上でのデコード処理
- **OutputHandler**: 結果の整形と出力

### データ構造

- **ColumnInfo**: カラム情報（名前、型、長さ）
- **ChunkConfig**: チャンク処理設定（行数、スレッド数など）

### エラー処理

異常が発生した場合のフォールバックと回復処理：

1. GPUメモリ確保失敗 → 行数削減でリトライ
2. パース処理エラー → 部分的な結果を返す
3. デコードエラー → リカバリモードで再実行

### 拡張ポイント

システムを拡張するための主要ポイント：

1. **コネクタ拡張**: 他のデータソース（MySQL, CSV等）への対応
2. **パーサー拡張**: 異なるバイナリフォーマットの追加
3. **データ型拡張**: 新しいデータ型のサポート
4. **出力拡張**: 異なる出力形式（Arrow, Parquet等）の追加
