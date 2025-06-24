# GPUPGParser - Claude Developer Guide

## Project Overview

**GPUPGParser** is a high-performance GPU-accelerated PostgreSQL binary data parser that converts PostgreSQL COPY BINARY data directly to GPU-native formats (cuDF DataFrames and Parquet files). The project leverages RAPIDS cuDF, CUDA kernels, and zero-copy optimizations to achieve significant performance improvements over traditional CPU-based parsing.

### Core Purpose
- **Primary Goal**: High-speed conversion of PostgreSQL binary data to columnar formats (Arrow/Parquet)
- **Key Innovation**: GPU-accelerated binary parsing with zero-copy optimizations
- **Target Use Case**: Large-scale PostgreSQL data analytics and ETL pipelines

## GPGPU開発哲学

### 開発スタンス
このプロジェクトは**妥協なきGPUによる最速処理**を追求します。どんなに実装難易度が高くても決してCPU速度に甘んじません：

**1. 言語対応**
- 日本語での開発・ドキュメント作成

**2. GPGPU革新への妥協なき姿勢**
- CPUはメタデータ管理やPostgres接続のみに限定使用です。

**3. メモリ最適化への深い理解**
- メモリコアレッシング（連続アクセス）を常に意識
- メモリバンクコンフリクトを回避する実装パターン
- GPU並列性を最大化するメモリアクセスパターン

**4. 技術スタック専門知識**
- **cuDF/pylibcudf/libcudf**: GPU DataFrameとApache Arrow統合
- **rmm/librmm**: unified memoryとmanaged memory.
- **Numba-cuda**: CUDAカーネル開発とGPU並列処理
- **Rust-cudf**: Rustベースの高性能GPU処理（必要に応じて）

**5. 特別な専門領域**
- **デシリアライゼーションGPGPU実装**: バイナリフォーマット→GPU直接変換
- **Apache Arrow変換GPGPU**: 任意フォーマット→Arrow形式のGPU加速
- **ゼロコピー最適化**: メモリ転送を排除した超高速処理

この哲学により、従来のCPU中心の処理を完全に刷新し、GPU本来の性能を100%引き出す革新的な実装を実現しています。


## 環境設定

### 重要: 実行環境の設定
このプロジェクトを実行する際は、必ず正しいconda環境を使用してください：

```bash
# conda環境のアクティベート（必須）
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cudf_dev

# 環境変数の設定
export GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"
```

### 利用可能なconda環境
- `cudf_dev` - **推奨**: cuDF/RAPIDS開発環境（メイン開発環境）
- `cudf_dev2412` - cuDF 24.12バージョン
- `numba-cuda` - Numba CUDA環境（cuDFなし）

### よくある問題
- `ModuleNotFoundError: No module named 'cudf'` → conda環境が間違っています。`conda activate cudf_dev`を実行してください
- `CondaError: Run 'conda init' before 'conda activate'` → `source /home/ubuntu/miniconda3/etc/profile.d/conda.sh`を先に実行してください

## 開発体制とガバナンス

### コミュニケーション原則
- 技術的な判断に迷った場合は、独断せずに必ず相談
- エラーや問題は隠さず、透明性を持って報告
- 実装の進捗を定期的に共有

## ベンチマーク実行の原則

### 重要な開発指針
1. **必ず全テーブルを対象に実施すること**
   - LIMITやサンプリングは使用しない
   - ctid範囲分割による並列処理で全データを処理
   - 真の性能測定のため、常に完全なデータセットを使用

2. **ストリーミング処理は極力避けること**
   GPUのクロック数はCPUよりも低い。しかし並列度をCPUより上げられるので速くなる。
   よって基本的にStream処理よりもBatch処理が速い。

3. **標準ベンチマーク設定**
   - **並列数**: 16 (`--parallel 16`)
   - **チャンク数**: 4 (`--chunks 4`)
   - **総タスク数**: 64（16×4）
   - この設定により真の高並列処理性能を測定。
   - この設定を変更する場合はユーザの許可をとること。勝手に変更することは許されない。

### 質問への回答方法
- 必ずウェブ検索しURL根拠をつけてください。
- 並列で考えられるところは、並列で考えること。
- 既存PGMを参考にする際は参照しているPGMも再帰的に調査し、深く理解すること。
- PGM構造に関する質問の場合、処理フローを記載してください。
- PGM構造に関する質問の場合、処理フロー全体が最速になるか確認すること。局所最適ではなく全体最適です。
- 回答を答える際に回答が問題を解決するか確認すること。
- 常にultrathinkすること。


#### ウェブ検索先
各サイトに検索ボックスがあれば有用。
該当ページから再帰的に調べること。
- **cuDF**:
  - https://docs.rapids.ai/api/cudf/stable/developer_guide/index.html
  - https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/
- **pylibcudf**:
  - https://docs.rapids.ai/api/cudf/stable/pylibcudf/developer_docs/
  - https://docs.rapids.ai/api/cudf/stable/pylibcudf/api_docs/  
- **libcudf**:
  - https://github.com/rapidsai/cudf/blob/branch-25.08/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md
  - https://docs.rapids.ai/api/libcudf/stable/modules.html
  - https://github.com/rapidsai/cudf/blob/branch-25.08/cpp/doxygen/developer_guide/TESTING.md
- **rmm**:
  - https://docs.rapids.ai/api/rmm/stable/python/
- **rmm**:
  - https://docs.rapids.ai/api/librmm/25.06/group__memory__resources
- **Numba-cuda**:
  -  [CUDAカーネル開発とGPU並列処理](https://numba.readthedocs.io/en/stable/)
- **Rust-cudf**: 
  - https://github.com/Rust-GPU/Rust-CUDA/blob/main/README.md
  - https://github.com/Rust-GPU/Rust-CUDA/blob/main/guide/src/guide/getting_started.md
  - https://github.com/Rust-GPU/Rust-CUDA/blob/main/guide/src/features.md
##### 参考
アルゴリズムの参考になるサイトは以下。
- **arrow-adbc**:CPUによるPostgresからArrow変換。Postgresのメタデータ転送が参考になる。
  - https://github.com/apache/arrow-adbc


## Memories
- **Memory**: Added memory section to track development insights and key project memories
- **Environment Setup**: 毎回conda環境の設定で問題が発生。必ず`cudf_dev`環境を使用すること
- **Development Governance**: 方針変更には上司の許可が必要。実装困難時は即座に状況報告と代替案の提示を行う。勝手に方針変更しない。
- **Benchmark Principles**: 必ず全テーブルを対象に実施。ストリーミング処理は使用禁止
- **Problem Solving**: 解決策を提案する際は、必ずその解決策が問題を解決することを確認してから提案すること
- **GPU Philosophy Compliance**: CPU転送は絶対に避ける。GPU→CPU→GPU転送は開発哲学違反。必ずGPU専用解決策を追求すること
- **Memory Coalescing**: 奇数/偶数行処理ではワープダイバージェンスとメモリコアレッシング問題に注意。ワープ最適化カーネルで解決
- **Current Development**: benchmark/benchmark_rust_gpu_direct.pyを使用してチューニング中。これは直接抽出版（統合バッファ削除）で、文字列破損修正済み
- **Work Output Organization**: 作業実施後は必ずアウトプットを整理すること。実行したプログラム、設定、結果を明確に文書化
- **File Organization After Development**: 開発完了後は必ずファイル整理を実施。旧版は`archive/`へ移動し、CLAUDE.mdを更新すること
- **16並列×8チャンク実装**: Rust側は`pg_fast_copy_single_chunk`を使用、環境変数`RUST_PARALLEL_CONNECTIONS=16`で16並列実行。各チャンク約8GB
- **kvikio+RMM最適化**: ファイル読み込みをkvikio+RMMで5.1倍高速化（3.6秒→0.71秒）。CPU経由を完全排除

## プロジェクト構造

### 現在のディレクトリ構成

```
gpupgparser/
├── CLAUDE.md               # このファイル - プロジェクト開発ガイド
├── check_lineorder_size.py # テーブルサイズ確認スクリプト
├── measure_postgres_speed.py # PostgreSQL読み取り速度測定
├── rust_fast_copy.py       # Rust実装ベンチマーク
├── simple_rust_benchmark.py # Rust簡易ベンチマーク
├── src/                    # メインソースコード
│   ├── __init__.py
│   ├── build_buf_from_postgres.py      # Postgres接続とバイナリデータ取得
│   ├── build_cudf_from_buf.py          # バイナリ→cuDF変換
│   ├── direct_column_extractor.py      # 直接カラム抽出（統合バッファなし）
│   ├── main_postgres_to_parquet.py     # メインエントリポイント（旧版）
│   ├── main_postgres_to_parquet_direct.py  # メインエントリポイント（直接抽出版）
│   ├── write_parquet_from_cudf.py     # cuDF→Parquet出力
│   ├── metadata.py         # メタデータ管理
│   ├── memory_manager.py   # メモリ管理
│   ├── types.py            # 型定義
│   ├── heap_file_reader.py # HEAPファイル読み込み
│   ├── cuda_kernels/       # CUDAカーネル実装
│   │   ├── __init__.py
│   │   ├── data_decoder.py             # GPUデータデコーダ
│   │   ├── integrated_parser_lite.py   # 統合パーサー軽量版
│   │   ├── postgres_binary_parser.py   # PostgreSQLバイナリパーサー
│   │   ├── postgresql_binary_parser.py # PostgreSQLバイナリパーサー（別実装）
│   │   ├── heap_page_parser.py         # HEAPページパーサー
│   │   ├── decimal_tables.py           # Decimal型変換テーブル
│   │   ├── gpu_config_utils.py         # GPU設定ユーティリティ
│   │   ├── math_utils.py               # 数学関数ユーティリティ
│   │   └── memory_utils.py             # メモリユーティリティ
│   ├── rust_integration/   # Rust連携モジュール
│   │   ├── __init__.py
│   │   ├── postgres_gpu_reader.py      # Rust経由のGPU読み込み
│   │   └── string_builder.py           # 文字列構築ヘルパー
│   └── old/                # 旧版実装（アーカイブ）
├── rust/                   # RustによるPostgres接続とGPU転送
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── README.md
│   ├── build.rs
│   ├── pyproject.toml
│   ├── src/
│   │   ├── lib.rs          # Python FFIインターフェース
│   │   ├── postgres.rs     # PostgreSQL接続・COPY実装
│   │   ├── arrow_builder.rs # Arrow形式構築
│   │   ├── cuda.rs         # CUDA/GPU転送実装
│   │   └── ffi.rs          # FFI定義
│   └── target/             # ビルド成果物
├── rust_bench/             # Rustベンチマーク（旧版）
├── rust_bench_optimized/   # Rust最適化ベンチマーク
│   └── src/
│       ├── main.rs         # メインベンチマーク
│       ├── main_single_chunk.rs  # 単一チャンク処理
│       ├── main_sequential_chunks.rs  # シーケンシャルチャンク処理
│       ├── main_sequential.rs      # シーケンシャル処理
│       └── main_env.rs             # 環境変数ベース処理
├── test/                   # テストコード
│   ├── debug/              # デバッグ用テスト
│   ├── expected_meta/      # テスト用期待値メタデータ
│   ├── fixtures/           # テスト用入力データ（旧input/から移動）
│   └── test_*.py           # 各種テストスクリプト
├── benchmark/              # ベンチマークスクリプト群
│   ├── benchmark_rust_gpu_direct.py    # 現在使用中のメインベンチマーク
│   ├── benchmark_*.py      # 各種ベンチマーク実装
│   └── *.parquet          # ベンチマーク出力（要整理）
├── docs/                   # ドキュメント
│   ├── guides/             # 使用ガイド
│   ├── implementation/     # 実装詳細
│   ├── ppt/                # プレゼンテーション資料
│   └── *.md                # 各種ドキュメント
├── examples/               # サンプルコード
│   ├── multigpu/           # マルチGPUサンプル
│   └── *.py                # 各種サンプル
└── archive/                # アーカイブされたコード（旧版・不要ファイル）
```

### 主要ファイルの役割

#### コアモジュール
- `src/build_buf_from_postgres.py`: PostgreSQLからバイナリデータを取得
- `src/build_cudf_from_buf.py`: GPUメモリ上のバイナリデータをcuDF DataFrameに変換
- `src/direct_column_extractor.py`: 統合バッファを使わない直接カラム抽出（最新版）
- `src/cuda_kernels/`: CUDA実装のコアロジック

#### Rust連携
- `rust/`: PostgreSQL接続とGPU転送の高速化実装
- `rust_fast_copy.py`: Python側のRust連携インターフェース

#### ベンチマーク
- `benchmark/benchmark_rust_gpu_direct.py`: 現在の主要ベンチマーク（16並列×8チャンク）

### 整理状況
1. **完了した整理作業**:
   - `old/` → `archive/` ✓
   - `input/` → `test/fixtures/` ✓
   - `output/` ディレクトリ削除 ✓
   - `logs/` ディレクトリ削除 ✓
   - `__pycache__/` ディレクトリ削除 ✓
   - `.pytest_cache/` ディレクトリ削除 ✓
   - 従来版benchmark_rust_gpu_direct.py → `archive/` ✓
   - kvikio版を新しいメインベンチマークに昇格 ✓

2. **今後の整理予定**:
   - benchmark/ディレクトリ内の多数の`.parquet`ファイルと`.bin`ファイルの削除
   - 不要なベンチマークスクリプトの整理

### 開発完了後のファイル整理ルール

開発が完了した際には、以下のルールに従ってファイルを整理すること：

1. **旧版・不要ファイルの移動**
   - 改良版に置き換えられた旧版は`archive/`ディレクトリへ移動
   - 一時的な実験コードや不要になったファイルも`archive/`へ
   - CLAUDE.mdに記載されているディレクトリ構造のみを使用（新規フォルダ作成禁止）

2. **ベンチマーク結果の整理**
   - 出力された`.parquet`や`.bin`ファイルは削除
   - 必要な結果のみドキュメント化してから削除

3. **ファイル名の正規化**
   - 新しい実装がメインになる場合は、分かりやすいファイル名に変更
   - 例: `benchmark_rust_gpu_direct_kvikio.py` → `benchmark_rust_gpu_direct.py`

4. **ドキュメントの更新**
   - CLAUDE.mdのプロジェクト構造を最新状態に更新
   - 整理作業の内容を記録

## 開発哲学準拠チェックリスト
解決策を提案する前に必ず確認：
- [ ] CPU転送を使用していないか？
- [ ] ゼロコピーを維持しているか？
- [ ] GPU並列性を最大化しているか？
- [ ] メモリコアレッシングを考慮しているか？
- [ ] 局所最適ではなく全体最適か？

## 開発計画と課題解決ロードマップ

### 現状の課題（2025年1月）
1. **GPUパース時間の不安定性**
   - チャンク毎に4.01秒〜17.24秒と大きくバラつく
   - 原因: GPU並列性が最適化されていない

2. **逐次処理によるスループット低下**
   - 8チャンクを順番に処理（並列化なし）
   - 全体スループット: 0.27 GB/秒（目標: 1GB/秒以上）

3. **16並列の恩恵を受けていない**
   - Rust側は16並列で高速（1.34 GB/秒）
   - GPU側がボトルネック

### 開発ロードマップ

#### Phase 1: 現状分析と最適化（即時対応）
- [ ] GPUパース処理のプロファイリング
  - Grid/Block sizeの最適化
  - メモリアクセスパターンの分析
- [ ] チャンク並列処理の実装
  - 複数チャンクの同時GPU処理
  - パイプライン化（転送と処理の並列化）

#### Phase 2: アーキテクチャ改善（1週間）
- [ ] 全データ一括処理の実装
  - 8チャンクを1つのGPU処理で実行
  - メモリ効率の向上
- [ ] カーネル最適化
  - ワープ効率の改善
  - 共有メモリの活用

#### Phase 3: 次世代実装（2週間）
- [ ] CUDA Graphsの導入
  - カーネル起動オーバーヘッドの削減
- [ ] マルチGPU対応
  - データ並列処理の実装

### 開発ループ防止策
1. **明確な目標設定**
   - 全体スループット: 1GB/秒以上
   - GPUパース時間: 安定して5秒以内

2. **段階的実装**
   - 各Phaseで測定可能な改善を確認
   - 改善が見られない場合は方針転換

3. **定期的な性能測定**
   - 各実装後にベンチマーク実行
   - 結果を文書化

### 次のアクション
1. GPUパース処理のGrid/Block size最適化
2. チャンク並列処理の実装
3. 全データ一括処理への移行
