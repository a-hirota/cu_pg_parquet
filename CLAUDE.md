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


## Memories
- **Memory**: Added memory section to track development insights and key project memories
- **Environment Setup**: 毎回conda環境の設定で問題が発生。必ず`cudf_dev`環境を使用すること
- **Development Governance**: 方針変更には上司の許可が必要。実装困難時は即座に状況報告と代替案の提示を行う。勝手に方針変更しない。
- **Benchmark Principles**: 必ず全テーブルを対象に実施。ストリーミング処理は使用禁止
- **Problem Solving**: 解決策を提案する際は、必ずその解決策が問題を解決することを確認してから提案すること
