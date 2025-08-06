# GPU PostgreSQL Parser テスト計画

## 1. 概要

本ドキュメントは、GPU PostgreSQL Parserの包括的なテスト計画を定義します。3つの主要機能と全PostgreSQLデータ型について、End-to-End（E2E）テストを実装します。

## 2. システム機能概要

### 2.1 処理フロー
現在の処理は大きく3つの機能に分かれています：

1. **機能1**: PostgreSQLバイナリ抽出とメタデータ生成
   - PostgreSQLからCOPY BINARYプロトコルでデータ取得
   - Rust/Pythonでバイナリデータ解析とメタデータ生成
   - PostgreSQL型からArrow型へのマッピング（src/types.py）

2. **機能2**: GPU処理（転送・解析・Arrow生成）
   - バイナリデータをGPUメモリに転送（kvikio使用）
   - GPUカーネルでPostgreSQLバイナリ形式を解析（src/cuda_kernels/）
   - GPU上でArrow配列を生成
   - row offsetとfield indicesを作成
   - indexを利用してcolumn_arrowsを作成（固定長/可変長）

3. **機能3**: Arrow→cuDF→Parquet変換
   - Arrow配列をcuDF DataFrameに変換
   - 各種圧縮オプションでParquetファイル出力

### 2.2 データ型マッピング

| PostgreSQL型 | Arrow型 | cuDF型 | 備考 |
|-------------|---------|---------|------|
| **数値型** |
| SMALLINT | int16 | int16 | 16ビット整数 |
| INTEGER | int32 | int32 | 32ビット整数 |
| BIGINT | int64 | int64 | 64ビット整数 |
| REAL | float32 | float32 | 単精度浮動小数点 |
| DOUBLE PRECISION | float64 | float64 | 倍精度浮動小数点 |
| NUMERIC/DECIMAL | string | string | 文字列として読み取り（精度の損失を防ぐため） |
| **文字列型** |
| TEXT | string | object/string | UTF-8文字列 |
| VARCHAR(n) | string | object/string | 可変長文字列 |
| CHAR(n) | string | object/string | 固定長文字列 |
| **バイナリ型** |
| BYTEA | binary | ListDtype(uint8) | バイナリデータをuint8の配列として表現 |
| **日付・時刻型** |
| DATE | date32 | datetime64[ms] | 日付（cuDFでは日単位精度はサポートされず、ミリ秒精度で扱う） |
| TIME | time64[us] | (部分サポート) | 時刻（cuDFは主にdatetimeを使用） |
| TIMESTAMP | timestamp[us] | datetime64[us] | タイムスタンプ（マイクロ秒精度） |
| TIMESTAMP WITH TIME ZONE | timestamp[us, tz=UTC] | datetime64[us] | cuDFはタイムゾーンを内部的に扱わない |
| INTERVAL | duration | timedelta64 | 期間 |
| **論理型** |
| BOOLEAN | bool | bool8 | 真偽値（cuDFは1バイト/値、Arrowはビットマップ） |
| **配列型** |
| ARRAY | list | ListDtype | 配列（1次元配列のみ完全サポート） |
| **複合型** |
| COMPOSITE TYPE | struct | StructDtype | ユーザー定義複合型 |
| **ネットワーク型** |
| INET | string または binary | object/string | IPアドレス |
| MACADDR | string または binary | object/string | MACアドレス |
| **その他の型** |
| UUID | fixed_size_binary[16] | ListDtype(uint8) | UUID（16バイトの固定長バイナリとして） |
| JSON/JSONB | string | object/string | JSON（文字列として） |
| XML | string | object/string | XML（文字列として） |

## 3. テスト実装計画

### 3.1 フェーズ1: テスト環境構築（2日）

- PostgreSQLテストデータベースのセットアップ
- テーブル作成スクリプトの実装
- テストデータ生成ユーティリティの作成

### 3.2 フェーズ2: 機能別テスト実装（10日）
各機能について以下のテストを実装：

#### 機能1テスト: PostgreSQLバイナリ抽出とメタデータ生成（3日）

- **Rust抽出とpsycopg2の比較テスト**
  - COPY BINARYで取得したデータをRust実装（rust/src/lib.rs）で解析
  - 同じデータをpsycopg2で取得し、値の完全一致を確認
  - サポートされている全データ型で検証

- **メタデータ生成テスト**
  - PostgreSQL型情報からArrow型への正しいマッピング
  - src/types.pyのPG_OID_TO_ARROWに基づく変換確認
  - 精度・スケール情報の保持（NUMERIC、VARCHARなど）

- **未実装型のエラーテスト**
  - TIME、UUID、JSON等の未実装型で適切なエラーが発生することを確認

#### 機能2テスト: GPU処理（転送・解析・Arrow生成）（3日）

- **実際のGPUカーネルテスト**
  - src/cuda_kernels/postgres_binary_parser.pyの実行
  - kvikio（または直接転送）でGPUメモリにデータ転送
  - GPU上でのバイナリ解析処理

- **Arrow配列生成と検証**
  - 固定長データ型の解析結果確認
  - 可変長データ型の解析結果確認
  - NULL値の正しい処理
  - CPUに戻して値・型・桁の一致確認

#### 機能3テスト: Arrow→cuDF→Parquet変換（2日）

- **cuDF変換テスト**
  - Arrow配列からcuDF DataFrameへの実際の変換
  - データ型の保持確認

- **Parquet出力テスト**
  - 各種圧縮オプション（snappy、zstd）の動作確認
  - 出力ファイルの読み込みと内容検証

### 3.3 フェーズ3: データ型別テスト実装（10日）
全データ型について以下のテストケースを作成：

#### テストデータ仕様

- 通常値: 100行
- NULL値: 各型で10%含む
- 境界値: 最小値、最大値を1行ずつ

#### 実装順序

1. **数値型テスト**（2日）
   - SMALLINT, INTEGER, BIGINT
   - REAL, DOUBLE PRECISION
   - NUMERIC/DECIMAL

2. **文字列型テスト**（2日）
   - TEXT（UTF-8、日本語含む）
   - VARCHAR(n)
   - CHAR(n)

3. **日付時刻型テスト**（2日）
   - DATE
   - TIME
   - TIMESTAMP（WITH/WITHOUT TIME ZONE）

4. **その他の基本型テスト**（2日）
   - BOOLEAN
   - BYTEA
   - UUID

5. **複雑な型テスト**（2日）
   - ARRAY
   - JSON/JSONB
   - COMPOSITE TYPE（必要に応じて）

### 3.4 フェーズ4: E2E統合テストとツール作成（3日）

- **E2E統合テスト**
  - 機能1→2→3を連結した完全なパイプラインテスト
  - PostgreSQL → COPY BINARY → Rust抽出 → GPU処理 → Arrow → cuDF → Parquet
  - 実際のコードのみ使用（モックなし）

- **Parquet検証ツール**
  - 出力されたParquetファイルの内容検証
  - PostgreSQL元データとの値の完全一致確認
  - データ型、NULL値、精度の保持確認

- **CI/CDパイプラインへの組み込み**
  - 実装済み機能のみテスト成功
  - 未実装機能は適切にスキップまたはエラー

## 4. テストコード構成

```
tests/
├── setup/
│   ├── create_test_db.py      # テストDB作成
│   └── generate_test_data.py  # テストデータ生成
├── e2e/
│   ├── test_rust_extraction.py      # 機能1: Rust抽出とメタデータ生成テスト
│   ├── test_gpu_processing.py      # 機能2: GPU処理（転送・解析・Arrow生成）テスト
│   └── test_arrow_to_parquet.py      # 機能3: Arrow→cuDF→Parquetテスト
├── datatypes/
│   ├── test_numeric_types.py  # 数値型テスト（実際のパイプライン使用）
│   ├── test_string_types.py   # 文字列型テスト（実際のパイプライン使用）
│   ├── test_datetime_types.py # 日付時刻型テスト（実際のパイプライン使用）
│   └── test_other_types.py    # その他の型テスト（実際のパイプライン使用）
├── integration/
│   └── test_full_pipeline.py  # E2E統合テスト（全機能結合）
└── utils/
    ├── verification.py        # Parquet検証ユーティリティ
    └── comparison.py         # Rust/psycopg2比較ユーティリティ
```

## 5. テスト実装戦略

### 5.1 段階的テスト実装アプローチ

実装の現状を考慮し、以下の段階的アプローチでテストを実装します：

#### フェーズ1: INTEGER型のみでの基本動作確認
1. **最初にINTEGER型（int32）のみでテストを作成**
   - 最も基本的で実装が完了している可能性が高い
   - バイナリ形式が単純（4バイト固定長）
   - エンディアン変換のみで値を取得可能

2. **各機能でINTEGER型テストを実装**
   - 機能1: INTEGER型のバイナリ抽出とメタデータ
   - 機能2: INTEGER型のGPU処理とArrow配列生成
   - 機能3: INTEGER型のArrow→cuDF→Parquet

3. **E2E統合テストもINTEGER型で実装**
   - 全パイプラインが動作することを確認

#### フェーズ2: 他の固定長数値型への拡張
- SMALLINT（int16）、BIGINT（int64）を追加
- REAL（float32）、DOUBLE PRECISION（float64）を追加
- これらも固定長で実装が比較的容易

#### フェーズ3: 可変長型への拡張
- TEXT、VARCHAR型を追加（可変長の処理が必要）
- NUMERIC/DECIMAL型を追加（特殊な形式）

#### フェーズ4: その他の型への拡張
- BOOLEAN、DATE、TIMESTAMP等
- 実装済みの型から順次追加

### 5.2 テストコードの構造化

```python
# 各テストクラスに型ごとのテストメソッドを追加
class TestFunction2GPUProcessing:
    def test_integer_type_only(self):
        """INTEGER型のみでGPU処理をテスト"""
        # まずこれを動作させる

    def test_numeric_types(self):
        """数値型全般のテスト"""
        # INTEGER型が動作したら追加

    def test_string_types(self):
        """文字列型のテスト"""
        # 可変長処理が実装されたら追加
```

## 6. 検証項目

### 6.1 機能検証

- [ ] 各機能が個別に正しく動作すること
- [ ] 全機能を通した処理が正常に完了すること
- [ ] エラー時に適切なメッセージが出力されること

### 5.2 データ整合性検証

- [ ] 全データ型で値が完全一致すること
- [ ] NULL値が正しく処理されること
- [ ] 境界値が正しく処理されること

### 5.3 性能検証（参考値）

- [ ] 1GBデータを10分以内に処理できること
- [ ] メモリ使用量が適切な範囲内であること

## 6. スケジュール

総期間: 約23日（4-5週間）

- **第1週**: テスト環境構築 + 機能1-2のE2Eテスト
- **第2週**: 機能3-4のE2Eテスト + 数値型・文字列型テスト
- **第3週**: 日付時刻型・その他の基本型テスト
- **第4週**: 複雑な型テスト + 統合テスト
- **第5週**: バグ修正とドキュメント整備

## 7. 成功基準

1. **必須**: 全機能のE2Eテストが合格
2. **必須**: 基本データ型（数値、文字列、日付、論理）のテストが合格
3. **推奨**: 全データ型のテストが合格
4. **推奨**: 統合テストでの値の完全一致確認

## 8. 現行のテスト実装とマッピング

現在のテスト実装と本番機能のマッピングについては、[test_mapping_analysis.md](./test_mapping_analysis.md)を参照してください。このドキュメントには以下の情報が含まれています：

- 本番処理フローの詳細（関数粒度）
- 各テストファイルと本番機能のマッピング
- テストされていない重要な機能のリスト
- 推奨される追加テストの提案

## 8. テスト実行環境

### 8.1 pytest設定

プロジェクトルートに`pytest.ini`を配置し、以下の設定を使用：

- テストパス: `tests/`
- マーカー: gpu, integration, slow, datatypes, e2e
- 出力オプション: 詳細表示、短いトレースバック

### 8.2 pre-commit統合

`.pre-commit-config.yaml`でテストを自動実行：

```yaml
- repo: local
  hooks:
    - id: pytest-check
      name: pytest
      entry: bash -c 'pytest tests/ -v --tb=short --maxfail=1 || exit 1'
      language: system
      types: [python]
      pass_filenames: false
      always_run: true
      stages: [commit]
```

これにより、コミット時に自動的にテストが実行され、Red/Greenの状態が確認できます。

### 8.3 テスト実行コマンド

```bash
# 全テスト実行
pytest tests/

# 特定の機能テストのみ
pytest tests/e2e/test_rust_extraction.py

# GPUテストを除外
pytest tests/ -m "not gpu"

# 高速テストのみ（slowを除外）
pytest tests/ -m "not slow"

# カバレッジ付き実行
pytest tests/ --cov=src --cov-report=html
```

### 8.4 継続的インテグレーション

GitHub ActionsやGitLab CIでの自動テスト実行設定も可能です。
