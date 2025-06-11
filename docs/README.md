# GPU PostgreSQL Parser - Documentation

## 概要

GPU PostgreSQL Parserは、PostgreSQLのCOPY BINARYデータをGPUで高速処理し、ArrowやParquet形式に変換するライブラリです。

## 🎉 最新の成果（統合最適化）

### パフォーマンス向上
- **メモリアクセス**: 50%削減（218MB × 2回 → 218MB × 1回）
- **実行時間**: 26.3%短縮
- **スループット**: 32,201,354 cells/sec

## 主要ドキュメント

### 🔥 最新実装
- **[INTEGRATED_PARSER_OPTIMIZATION.md](INTEGRATED_PARSER_OPTIMIZATION.md)** - **統合最適化実装ガイド（推奨）**
- `SUCCESS_ANALYSIS.md` - 最適化成功の分析
- `VERIFICATION_GUIDE.md` - テストと検証のガイド

### 📁 guides/ - ユーザーガイド
実際の使用方法やベストプラクティスに関するドキュメント

- `usage_guide.md` - 基本的な使用方法
- `parquet_usage.md` - Parquet出力の詳細ガイド
- `parquet_flow.md` - Parquetワークフローの解説
- `ray_multigpu_guide.md` - Ray分散処理ガイド

### 📁 implementation/ - 実装詳細
内部実装やアルゴリズムに関する技術文書

- `DECIMAL_OPTIMIZATION_README.md` - Decimal型最適化の実装詳細

### 📁 archive/ - アーカイブ文書
過去の実装やバージョン固有の文書

- `V7_COLUMN_WISE_INTEGRATION_GUIDE.md` - V7列処理統合ガイド（旧版）
- `DECIMAL_COLUMN_WISE_IMPLEMENTATION.md` - 列単位Decimal実装（旧版）
- `test_code_modifications.md` - テストコード変更履歴
- `graph.md` - パフォーマンスグラフ

### 📁 ppt/ - プレゼンテーション資料
プロジェクト概要や技術説明のスライド

## クイックスタート

1. **統合最適化版**: `INTEGRATED_PARSER_OPTIMIZATION.md` **（推奨）**
2. **基本的な使用方法**: `guides/usage_guide.md`
3. **Parquet出力**: `guides/parquet_usage.md`
4. **分散処理**: `guides/ray_multigpu_guide.md`

## 技術詳細

最新の統合最適化実装については `INTEGRATED_PARSER_OPTIMIZATION.md` を参照してください。

---
*最終更新: 2025年6月11日 - 統合最適化実装完了*
