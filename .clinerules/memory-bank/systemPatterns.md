# システムパターン: GpuPgParser

## 1. アーキテクチャ概要

GpuPgParserは、PostgreSQLからデータを抽出し、GPUで処理してParquetファイルとして出力するパイプラインアーキテクチャを採用しています。主要なコンポーネントは以下の通りです。（フォルダ構成は `src/`, `test/` などに変更済み）

```mermaid
graph LR
    A[PostgreSQL DB] -- Binary Data --> B(Python Process / src/main.py);
    B -- Control & Data Chunk --> C{GPU Memory};
    C -- Decoded Data --> B;
    B -- Parquet Data --> D[Output Parquet File];

    subgraph GPU Processing Unit
        C -- Binary Chunk --> E[CUDA Kernels (Numba/CuPy)];
        E -- Decoded Chunk --> C;
    end

    F[run_multigpu_parquet_optimized.sh] -- Manages --> B;
```

-   **実行スクリプト (`run_multigpu_parquet_optimized.sh`):**
    -   複数のPythonプロセス（`src/main.py`）を起動し、それぞれに異なるGPUとデータ範囲を割り当てて並列処理を管理します。
    -   環境変数やコマンドライン引数を通じて、各プロセスに必要な設定（SQLクエリ、出力パス、GPU IDなど）を渡します。
-   **メイン処理プロセス (`src/main.py`):**
    -   `PgGpuProcessor` クラス（`src/main.py` 内と想定）が中心となり、データ取得、GPU処理、出力のパイプラインを制御します。
    -   **データ取得 (`src/pg_connector.py`):** PostgreSQLに接続し、`COPY TO STDOUT (FORMAT BINARY)` を実行してバイナリデータを取得します。
    -   **メタデータ取得 (`src/meta_fetch.py`):** クエリからスキーマ情報を取得し、`ColumnMeta` オブジェクトを生成します。
    -   **GPUパース (Pass 0) (`src/gpu_parse_wrapper.py`, `src/cuda_kernels/pg_parser_kernels.py`):** GPUカーネルを呼び出し、バイナリデータからフィールドオフセットと長さを計算します。
    -   **メモリ管理 (`src/gpu_memory_manager_v2.py`):** `ColumnMeta` に基づき、GPU上にArrow形式出力用のバッファ（データ、NULL、可変長用オフセット）を初期確保します。**可変長データバッファは後で実際のサイズに再確保されます。**
    -   **GPUデコード (Pass 1 & 2) & Arrow組立 (`src/gpu_decoder_v2.py`, `src/cuda_kernels/arrow_*.py`):**
        *   Pass 1: GPUカーネルでNULL/長さ収集。
        *   Prefix Sum: CuPyで可変長列のオフセットと合計サイズを計算。
        *   メモリ再確保: 計算された合計サイズに基づき、`gpu_memory_manager_v2.py` を介して可変長データバッファを再確保。
        *   Pass 2: GPUカーネルでデータコピー/変換 (Scatter Copy)。固定長、可変長、Decimal128用に個別のカーネルを使用。
        *   Arrow組立: **`pyarrow.from_buffers` を使用し、GPUバッファから（`pyarrow.cuda` 利用可能時はゼロコピーで）`pyarrow.RecordBatch` を直接組み立てる。**
    -   **出力処理 (`src/output_handler.py`):** 生成された `RecordBatch` を受け取り、`pyarrow` を使用してParquetファイルとして書き込みます。

## 2. 主要なデザインパターンと技術的決定

-   **マルチパスGPU処理パイプライン:** データ取得 → メタデータ取得 → GPU転送 → GPUパース(Pass 0) → GPUメタ収集(Pass 1) → Prefix Sum & メモリ再確保 → GPUデータ変換(Pass 2) → Arrow組立 → 出力 のステップを順次実行します。**Arrow組立ステップは `from_buffers` を使用して最適化されています。** 現在は各ステップが同期的に実行されていますが、将来的にはCUDA Streams等を用いた非同期化によるオーバーラップを目指します (`techContext.md` 参照)。
-   **二段階メモリ確保 (可変長):** 可変長列のデータバッファは、まず最大長で見積もって確保し、Prefix Sumで実際の合計サイズが判明した後に再確保する方式を採用しています (`gpu_memory_manager_v2.py`)。
-   **チャンキング:** 大規模データをGPUメモリに収まるサイズのチャンクに分割して処理します。
-   **プロセスベースの並列化 (マルチGPU):** `run_multigpu_parquet_optimized.sh` が複数の独立したPythonプロセス (`src/main.py`) を起動し、それぞれが特定のGPUを担当します。
-   **GPUカーネルの直接実行 (Numba):** Numbaを使用してPython (`.py`) ファイル内でCUDAカーネルを定義し、JITコンパイルして実行します。
-   **バイナリ形式の直接処理:** PostgreSQLのバイナリ形式を直接GPUで処理します。
-   **Apache Arrowによる中間表現:** GPUでデコードされたデータはArrow形式で表現されます。

## 3. データフロー

    1.  シェルスクリプト (`run_multigpu_parquet_optimized.sh`) が `src/main.py` を複数起動 (GPUごとに1プロセス)。
    2.  各 `src/main.py` プロセスが担当範囲のデータをPostgreSQLからバイナリ形式で取得 (`src/pg_connector.py`)。
    3.  スキーマ情報を取得 (`src/meta_fetch.py`)。
    4.  バイナリデータをチャンクに分割。
    5.  チャンクデータをGPUメモリに転送。
    6.  GPUカーネルでオフセット/長さを計算 (Pass 0) (`src/gpu_parse_wrapper.py` -> `src/cuda_kernels/pg_parser_kernels.py`)。
    7.  GPUカーネルでNULL/可変長メタデータを収集 (Pass 1) (`src/gpu_decoder_v2.py` -> `src/cuda_kernels/arrow_gpu_pass1.py`)。
    8.  GPUメモリを初期確保（オフセットバッファ含む） (`src/gpu_memory_manager_v2.py`)。
    9.  Prefix Sumでオフセット計算 & 可変長データバッファを再確保 (`src/gpu_decoder_v2.py`, `src/gpu_memory_manager_v2.py`)。
    10. GPUカーネルでデータをコピー/変換 (Pass 2) (`src/gpu_decoder_v2.py` -> `src/cuda_kernels/arrow_gpu_pass2_*.py`)。
    11. デコード済みGPUバッファから `from_buffers` を使用してArrow RecordBatchを組み立てる (`src/gpu_decoder_v2.py`)。
    12. RecordBatchをParquet形式でファイルに書き込む (`src/output_handler.py`)。
    13. 全プロセスの完了を待つ。

## 4. 課題と考慮事項

-   **エラーハンドリング:** パイプライン各段階でのエラーハンドリングとリカバリ戦略。
-   **型システムの整合性:** PostgreSQL型、バイナリ表現、CUDAカーネル内型、Arrow/Parquet型間の整合性。
-   **マルチGPUの同期と結果統合:** 現状は独立ファイル出力。単一ファイル統合や依存関係処理は未実装。
-   **カーネルの最適化:** Numbaカーネルのパフォーマンス最適化（メモリアクセス、スレッド効率）。
-   **128ビット演算:** Numbaカーネル内の `uint64` ペアによる128ビット演算の正確性（特に乗除算）。
