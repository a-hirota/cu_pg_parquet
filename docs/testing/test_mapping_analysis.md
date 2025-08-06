# 本番処理フローとテストのマッピング分析

## 1. 本番処理フロー（関数粒度）

### メインフロー
```
cu_pg_parquet.py::main()
└── processors/gpu_pipeline_processor.py::main()
    ├── setup_rmm_pool() - RMMメモリプール初期化
    ├── rust_producer() - Producerスレッド
    │   └── pg_chunk_extractor (Rustバイナリ)
    ├── gpu_warmup() - GPUウォーミングアップ
    └── gpu_consumer() - Consumerスレッド
        └── convert_postgres_to_parquet_format()
            └── DirectProcessor (内部で使用)
```

### 詳細な関数フロー

#### 1. データ抽出フェーズ（Rust）
```python
# processors/gpu_pipeline_processor.py
rust_producer()
  └── subprocess.run(pg_chunk_extractor)  # Rustバイナリ実行
      └── PostgreSQL COPY BINARY protocol でデータ抽出
```

#### 2. GPU処理フェーズ
```python
# processors/gpu_pipeline_processor.py
gpu_consumer()
  ├── kvikio.CuFile.read() - GPU Direct Storage転送
  ├── detect_pg_header_size() - ヘッダーサイズ検出
  └── convert_postgres_to_parquet_format()
      └── DirectProcessor() インスタンス作成
          └── transform_postgres_to_parquet_format()
              ├── parse_postgres_raw_binary_to_column_arrows()
              │   ├── calculate_gpu_grid_dimensions() - グリッド計算
              │   ├── parse_rows_and_fields_lite() - CUDAカーネル
              │   └── DirectColumnExtractor.extract_columns_direct()
              │       ├── create_string_buffers() - 文字列バッファ作成
              │       ├── extract_fixed_column_direct() - 固定長抽出
              │       └── _create_string_series_from_buffer() - 文字列列作成
              └── write_cudf_to_parquet_with_options() - Parquet書き込み
```

## 2. テストと本番関数のマッピング

### test_rust_extraction.py - PostgreSQLバイナリ抽出テスト

**テスト内容**: PostgreSQLのCOPY BINARYフォーマットの手動パースとpsycopg2との比較

**カバーしている本番機能**:
- ❌ `pg_chunk_extractor` (Rustバイナリ) - **テストされていない**
- ✅ PostgreSQL COPY BINARYフォーマット仕様の理解
- ✅ メタデータ（列名、型情報）の検証

**実際のテスト内容**:
```python
# バイナリフォーマットを手動でパース
binary_value = struct.unpack(">i", binary_field)[0]
assert binary_value == pg_field  # psycopg2の結果と比較
```

### test_gpu_processing.py - GPU処理テスト

**テスト内容**: GPU上でのバイナリデータ解析とArrow配列生成

**カバーしている本番機能**:
- ✅ `detect_pg_header_size()` - ヘッダーサイズ検出
- ✅ `parse_postgres_raw_binary_to_column_arrows()` - メイン解析関数
- ✅ `calculate_gpu_grid_dimensions()` - グリッド計算（内部で呼ばれる）
- ✅ `parse_rows_and_fields_lite()` - CUDAカーネル（内部で呼ばれる）
- ❌ `kvikio.CuFile.read()` - GPU転送（テストではnp.frombufferを使用）
- ❌ `DirectColumnExtractor` - カラム抽出（parse_postgres_raw_binary_to_column_arrowsに含まれる）

**実際のテスト内容**:
```python
result = parse_postgres_raw_binary_to_column_arrows(
    raw_dev, columns, header_size=header_size, debug=True, test_mode=True
)
# Arrow配列の内容を検証
```

### test_arrow_to_parquet.py - Arrow→cuDF→Parquet変換テスト

**テスト内容**: Arrow配列からcuDF DataFrameを作成し、Parquetファイルに書き込み

**カバーしている本番機能**:
- ✅ `cudf.DataFrame.from_arrow()` - Arrow→cuDF変換
- ✅ `cudf.DataFrame.to_parquet()` - Parquet書き込み（基本機能）
- ❌ `write_cudf_to_parquet_with_options()` - 本番の最適化版Parquet書き込み

**実際のテスト内容**:
```python
gdf = cudf.DataFrame.from_arrow(table)
gdf.to_parquet(parquet_path, compression='snappy')
```

### test_all_types.py - 統合テスト（全データ型）

**テスト内容**: DirectProcessorを使用した全体フローのテスト

**カバーしている本番機能**:
- ✅ `DirectProcessor.transform_postgres_to_parquet_format()` - メイン処理
- ✅ 内部で呼ばれるすべての関数（parse、extract、write）
- ❌ `rust_producer()` - Producerスレッド
- ❌ `gpu_consumer()` - Consumerスレッド
- ❌ Producer-Consumerパターンのキュー処理

### test_full_pipeline.py - 完全パイプラインテスト

**テスト内容**: PostgreSQLからParquetまでの完全なフロー

**カバーしている本番機能**:
- ✅ `DirectProcessor` - 完全な処理フロー
- ❌ マルチスレッド処理（Producer-Consumer）
- ❌ Rustバイナリとの連携

## 3. テストされていない重要な機能

### 1. Rust抽出処理 (`pg_chunk_extractor`)
- 本番では16並列接続でPostgreSQLからデータ抽出
- チャンク分割とワーカー管理
- /dev/shmへの高速書き込み

### 2. Producer-Consumerパターン
- `rust_producer()` スレッド
- `gpu_consumer()` スレッド
- キューによるバックプレッシャー制御
- スレッド間の同期とエラー処理

### 3. GPU Direct Storage (`kvikio`)
- ファイルからGPUメモリへの直接転送
- ゼロコピー最適化

### 4. 最適化されたParquet書き込み
- `write_cudf_to_parquet_with_options()`
- 圧縮オプション（zstd等）
- Spark互換性メタデータ

### 5. メモリ管理
- RMMプール管理
- GPU/CPUメモリの解放処理
- エラー時のクリーンアップ

## 4. 推奨される追加テスト

1. **Rust抽出テスト**
   - `pg_chunk_extractor`の単体テスト
   - PostgreSQL接続エラーのハンドリング
   - チャンク境界の正確性

2. **マルチスレッドテスト**
   - Producer-Consumerの協調動作
   - キューのオーバーフロー/アンダーフロー
   - スレッド間のエラー伝播

3. **パフォーマンステスト**
   - 大規模データでの処理時間測定
   - メモリ使用量の監視
   - GPUリソースの利用効率

4. **エラーハンドリングテスト**
   - 不正なバイナリデータ
   - GPUメモリ不足
   - ファイルI/Oエラー
