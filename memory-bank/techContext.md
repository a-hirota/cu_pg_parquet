# 技術コンテキスト

## 開発環境

- Python環境: conda (numba-cuda)
  - 主要パッケージ:
    - cupy: CUDA対応のNumPy互換ライブラリ
    - numba: JITコンパイラ（CUDA対応）
    - psycopg2: PostgreSQLクライアント

## 技術スタック

- CUDA: GPUプログラミング基盤
- PostgreSQL: データベース
- Python: 開発言語

## 開発ツール

- VSCode: 開発IDE
- Git: バージョン管理

## 実行環境

- conda環境名: numba-cuda
  - CUDA対応のnumbaとその他必要なパッケージがインストール済み
  - 実行時は `conda activate numba-cuda` で環境を有効化する必要あり

## 依存関係

- CUDA Toolkit: GPUプログラミングに必要
- PostgreSQL: データベースサーバー
- Python 3.x: 実行環境
