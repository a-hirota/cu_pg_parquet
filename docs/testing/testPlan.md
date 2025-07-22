# GPU PostgreSQL Parser テスト計画

## 1. 概要

本ドキュメントは、GPU PostgreSQL Parserの包括的なテスト計画を定義します。4つの主要機能と全PostgreSQLデータ型について、End-to-End（E2E）テストを実装します。

## 2. システム機能概要

### 2.1 処理フロー
現在の処理は大きく4つの機能に分かれています：

1. **機能1**: PostgreSQLからpostgres_raw_binaryを/dev/shmにあるキューに蓄積
2. **機能2**: キューから取り出し、kvikioを利用してGPUメモリに転送
3. **機能3**: GPUメモリ上でバイナリ解析とArrow配列作成
   - row offsetとfield indicesを作成
   - indexを利用してcolumn_arrowsを作成（固定長/可変長）
4. **機能4**: column_arrowをcuDF DataFrameに変換し、Parquetファイルとして出力

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

### 3.2 フェーズ2: 機能別E2Eテスト実装（8日）
各機能について以下のテストを実装：

#### 機能1テスト: PostgreSQL → /dev/shmキュー（2日）
- 基本動作確認（10行のシンプルデータ）
- 大量データ処理（100万行）
- エラーハンドリング（接続エラー、キュー満杯）

#### 機能2テスト: /dev/shmキュー → GPU転送（2日）
- kvikio転送の動作確認
- メモリ不足時の挙動
- 並行転送の安定性

#### 機能3テスト: GPUバイナリ解析 → Arrow配列（2日）
- 固定長データ型の解析
- 可変長データ型の解析
- NULL値の正しい処理

#### 機能4テスト: Arrow → cuDF → Parquet（2日）
- 基本的な変換動作
- 圧縮オプション（ZSTD）の動作
- 大規模データの出力

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

### 3.4 フェーズ4: 統合テストとツール作成（3日）
- 全機能（1-4）を通した統合テストスクリプト
- テスト結果検証ツール（値の完全一致確認）
- CI/CDパイプラインへの組み込み

## 4. テストコード構成

```
tests/
├── setup/
│   ├── create_test_db.py      # テストDB作成
│   └── generate_test_data.py  # テストデータ生成
├── e2e/
│   ├── test_function1.py      # 機能1のE2Eテスト
│   ├── test_function2.py      # 機能2のE2Eテスト
│   ├── test_function3.py      # 機能3のE2Eテスト
│   └── test_function4.py      # 機能4のE2Eテスト
├── datatypes/
│   ├── test_numeric_types.py  # 数値型テスト
│   ├── test_string_types.py   # 文字列型テスト
│   ├── test_datetime_types.py # 日付時刻型テスト
│   └── test_other_types.py    # その他の型テスト
├── integration/
│   └── test_full_pipeline.py  # 統合テスト
└── utils/
    ├── verification.py        # 検証ユーティリティ
    └── performance.py         # 性能測定ツール
```

## 5. 検証項目

### 5.1 機能検証
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
