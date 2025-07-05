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
- **Producer-Consumer並列処理実装（2025/01）**: キューベースの並列処理で1.56倍高速化達成（74.77秒→47.95秒）。Rust転送とGPU処理を真の並列実行
- **ログリファクタリング完了**: 詳細ログを削減し、最後に構造化された統計表示のみ出力。validate_parquet_output関数のサンプル表示は維持
- **データ圧縮パース**: Apache Arrowベースの新規データ圧縮パース方式を研究開始（2025/03予定）
- **16チャンク900万行制限問題（2025/01/30）**: 第2段階のCUDAカーネル（_extract_fixed_direct、_extract_decimal_direct、copy_string_data_direct）が1スレッド1行処理に固定されており、総スレッド数（9,328,896）が処理可能な最大行数となっている。第1段階は理論上2,388,197,376行まで検出可能だが、第2段階がボトルネック。
- **to memorize**