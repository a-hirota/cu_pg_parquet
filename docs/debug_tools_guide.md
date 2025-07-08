# デバッグツールガイド

## 概要

GPUPGParserのデバッグツールは、PostgreSQLバイナリデータのパース時に発生する問題を調査するためのユーティリティです。

## ツール一覧

### 1. debug_by_page_and_row.py

PostgreSQLのページ番号またはParquetのrow_numberから、該当位置のバイナリデータを表示します。

#### 使用方法

```bash
# row_numberで検索（Parquetファイルの行番号）
python tools/debug_by_page_and_row.py <row_number>

# ページ番号とオフセットで検索
python tools/debug_by_page_and_row.py <page_number> <offset>
```

#### 使用例

```bash
# row_number 1234567の行を調査
python tools/debug_by_page_and_row.py 1234567

# ページ13023の先頭を調査
python tools/debug_by_page_and_row.py 13023 0
```

#### 出力内容

- スレッド情報（thread_id、開始位置、終了位置）
- フィールド値（c_custkey、c_name等）
- バイナリダンプ（16進数とASCII表示）
- スレッド境界での00 08パターン検索結果

### 2. debug_custkey_zero_dump.py

c_custkey=0となっている問題のある行を検出し、その周辺のバイナリデータを詳細に分析します。

#### 使用方法

```bash
python tools/debug_custkey_zero_dump.py
```

#### 機能

- Parquetファイルからc_custkey=0の行を自動検出
- 該当行のバイナリ位置を特定
- フィールド構造の解析
- 前後のバイナリデータをカラー表示
- 実際の行ヘッダ位置の推定

#### 出力内容

- 問題のある行の詳細情報
- フィールド毎のバイト位置と値
- カラーハイライト付きバイナリダンプ
- 前の行構造の探索結果

## デバッグのワークフロー

### 1. 問題の特定

```bash
# Parquetファイルの内容を確認
python analyze_boundary_issue.py
```

### 2. 詳細調査

```bash
# c_custkey=0の行を詳細分析
python tools/debug_custkey_zero_dump.py

# 特定のrow_numberを調査
python tools/debug_by_page_and_row.py <row_number>
```

### 3. バイナリ構造の確認

出力されたバイナリダンプから以下を確認：
- 行ヘッダ（00 08）の位置
- フィールド長の妥当性
- スレッド境界での異常パターン

## PostgreSQLバイナリフォーマット

### 行構造
```
[行ヘッダ(2bytes)] [フィールド1] [フィールド2] ... [フィールドN]
```

### フィールド構造
```
[長さ(4bytes)] [データ(可変長)]
```

- 長さが-1（0xFFFFFFFF）の場合はNULL
- customerテーブルの場合、8フィールド（行ヘッダは00 08）

### 固定長フィールド

customerテーブルの固定長フィールド：
- c_city: 10バイト
- c_nation: 15バイト
- c_region: 12バイト
- c_phone: 15バイト
- c_mktsegment: 10バイト

## トラブルシューティング

### c_custkey=0の問題

スレッド境界で前の行の途中にある`00 08`パターンを誤認識する問題：

1. スレッド開始位置で`00 08`を発見
2. しかし実際は前の行のデータの一部
3. 検証失敗後の処理で不正な位置が記録される

### 解決方法

- スレッド境界での追加検証
- 最初のフィールド長の妥当性チェック
- 固定長フィールドの検証強化