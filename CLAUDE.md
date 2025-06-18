# GPUPGParser - Claude Developer Guide

## Project Overview

**GPUPGParser** is a high-performance GPU-accelerated PostgreSQL binary data parser that converts PostgreSQL COPY BINARY data directly to GPU-native formats (cuDF DataFrames and Parquet files). The project leverages RAPIDS cuDF, CUDA kernels, and zero-copy optimizations to achieve significant performance improvements over traditional CPU-based parsing.

### Core Purpose
- **Primary Goal**: High-speed conversion of PostgreSQL binary data to columnar formats (Arrow/Parquet)
- **Key Innovation**: GPU-accelerated binary parsing with zero-copy optimizations
- **Target Use Case**: Large-scale PostgreSQL data analytics and ETL pipelines

## GPGPU開発哲学

### 開発スタンス
このプロジェクトは**妥協なきGPGPU革新**を追求します：

**1. 言語対応**
- 日本語での開発・ドキュメント作成

**2. GPGPU革新への妥協なき姿勢**
- GPGPUでの実装を必ずやり遂げます
- CPUのみの実装では絶対に妥協しません  
- CPUはテストやメタデータ管理のみに限定使用

**3. メモリ最適化への深い理解**
- メモリコアレッシング（連続アクセス）を常に意識
- メモリバンクコンフリクトを回避する実装パターン
- GPU並列性を最大化するメモリアクセスパターン

**4. 技術スタック専門知識**
- **cuDF/pylibcudf/libcudf**: GPU DataFrameとApache Arrow統合
- **Numba-cuda**: CUDAカーネル開発とGPU並列処理
- **Rust-cudf**: Rustベースの高性能GPU処理（必要に応じて）

**5. 特別な専門領域**
- **デシリアライゼーションGPGPU実装**: バイナリフォーマット→GPU直接変換
- **Apache Arrow変換GPGPU**: 任意フォーマット→Arrow形式のGPU加速
- **ゼロコピー最適化**: メモリ転送を排除した超高速処理

この哲学により、従来のCPU中心の処理を完全に刷新し、GPU本来の性能を100%引き出す革新的な実装を実現しています。

## Memories
- **Memory**: Added memory section to track development insights and key project memories