# GPU PostgreSQL Parser - Documentation

## 概要

GPU PostgreSQL Parserは、PostgreSQLのCOPY BINARYデータをGPUで高速処理し、ArrowやParquet形式に変換するライブラリです。

## ドキュメント構成

### 📁 guides/ - ユーザーガイド
実際の使用方法やベストプラクティスに関するドキュメント

- `usage_guide.md` - 基本的な使用方法
- `parquet_usage.md` - Parquet出力の詳細ガイド
- `parquet_flow.md` - Parquetワークフローの解説
- `ray_multigpu_guide.md` - Ray分散処理ガイド

### 📁 implementation/ - 実装詳細
内部実装やアルゴリズムに関する技術文書

- `DECIMAL_OPTIMIZATION_README.md` - Decimal型最適化の実装詳細
- `DECIMAL_PASS1_OPTIMIZATION_GUIDE.md` - Pass1最適化ガイド

### 📁 archive/ - アーカイブ文書
過去の実装やバージョン固有の文書

- `V7_COLUMN_WISE_INTEGRATION_GUIDE.md` - V7列処理統合ガイド（旧版）
- `DECIMAL_COLUMN_WISE_IMPLEMENTATION.md` - 列単位Decimal実装（旧版）
- `test_code_modifications.md` - テストコード変更履歴
- `graph.md` - パフォーマンスグラフ

### 📁 ppt/ - プレゼンテーション資料
プロジェクト概要や技術説明のスライド

## クイックスタート

1. **基本的な使用方法**: `guides/usage_guide.md`
2. **Parquet出力**: `guides/parquet_usage.md`
3. **分散処理**: `guides/ray_multigpu_guide.md`

## 技術詳細

実装の詳細について知りたい場合は `implementation/` ディレクトリの文書を参照してください。

## バージョン履歴

過去のバージョンや実装に関する文書は `archive/` ディレクトリにあります。
