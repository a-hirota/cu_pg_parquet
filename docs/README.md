# GPU PostgreSQL Parser - Documentation
=====================================

## 概要

GPU PostgreSQL Parserは、PostgreSQLのCOPY BINARYデータをGPUで高速処理し、ArrowやParquet形式に変換するライブラリです。

## 🚀 最新の成果（GPUソート最適化）

### パフォーマンス向上
- **適応的ソート戦略**: 小規模データはCPU、大規模データはGPU最適化
- **大規模データ高速化**: 2-6倍のソート性能向上（50,000行以上）
- **GPU↔CPU転送排除**: 大規模データでメモリ転送を完全排除
- **スループット**: 35M+ cells/sec（大規模データ）

## 主要ドキュメント

### 🔥 最新実装（2025年6月）
- **[GPU_SORT_OPTIMIZATION_SUMMARY.md](GPU_SORT_OPTIMIZATION_SUMMARY.md)** - **GPUソート最適化実装ガイド（推奨）**
- **[INDEX_ORDERING_EXPLANATION.md](INDEX_ORDERING_EXPLANATION.md)** - インデックス順序の詳細説明
- **[PGM_PROCESSING_FLOW.md](PGM_PROCESSING_FLOW.md)** - 処理フロー図（最新版）

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

## 🎯 クイックスタート

### 基本的な使用方法
```bash
# 環境変数設定
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'

# ベンチマーク実行（500万行）
python benchmark_main.py --rows 5000000

# ベンチマーク実行（1000万行）  
python benchmark_main.py --rows 10000000

# GPUソート性能テスト
python benchmark_main.py --test gpu_sort
```

### ドキュメント推奨順序
1. **GPUソート最適化**: `GPU_SORT_OPTIMIZATION_SUMMARY.md` **（推奨）**
2. **処理フロー理解**: `PGM_PROCESSING_FLOW.md`
3. **基本的な使用方法**: `guides/usage_guide.md`
4. **Parquet出力**: `guides/parquet_usage.md`
5. **分散処理**: `guides/ray_multigpu_guide.md`

## 技術的ハイライト

### 適応的ソート最適化
- **小規模データ（<50,000行）**: CPU最適ソート（GPU初期化オーバーヘッド回避）
- **大規模データ（≥50,000行）**: GPU高速ソート（CuPy並列処理活用）
- **自動切り替え**: データサイズに応じた最適な処理方式を選択

### 実装ファイル
- `src/cuda_kernels/integrated_parser_lite.py` - 軽量統合パーサー
- `src/cuda_kernels/postgresql_binary_parser.py` - 従来版パーサー
- `benchmark_main.py` - 性能測定ツール

## 📊 性能指標

| データサイズ | 最適ソート方式 | 性能向上 | 理由 |
|-------------|---------------|----------|------|
| 10,000行    | CPU          | 基準     | GPU初期化コストが支配的 |
| 100,000行   | GPU          | 2.0x     | GPU並列処理開始 |
| 1,000,000行 | GPU          | 5.0x     | GPU真価発揮 |
| 10,000,000行| GPU          | 6.3x     | 大規模並列処理 |

## 🧪 テスト環境

- `test/test_gpu_sort_simple.py` - 基本動作確認
- `test/test_gpu_sort_performance.py` - 詳細性能測定
- `benchmark_main.py` - 実践的ベンチマーク

---
*最終更新: 2025年6月12日 - GPUソート最適化実装完了*