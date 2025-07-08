# バイナリファイル命名規則の提案

## 問題
- 現在のバイナリファイル名（`chunk_0.bin`）からテーブル名が分からない
- 異なるテーブルのデータを誤って処理する可能性がある

## 解決策

### 1. ファイル名にテーブル名を含める
```
# 現在
/dev/shm/chunk_0.bin
/dev/shm/chunk_1.bin

# 提案
/dev/shm/customer_chunk_0.bin
/dev/shm/customer_chunk_1.bin
/dev/shm/lineorder_chunk_0.bin
/dev/shm/lineorder_chunk_1.bin
```

### 2. メタデータファイルの追加
各チャンクファイルと一緒にメタデータJSONを作成：
```json
// /dev/shm/customer_chunk_0.meta.json
{
  "table_name": "customer",
  "chunk_id": 0,
  "total_chunks": 2,
  "columns": 8,
  "start_page": 0,
  "end_page": 104186,
  "timestamp": "2024-07-08T12:00:00Z",
  "expected_rows": 6015125,
  "file_size": 848104872,
  "checksum": "sha256:..."
}
```

### 3. カスタムヘッダーの追加
PostgreSQL COPY BINARYフォーマットの後に独自のヘッダーを追加：
```
[PGCOPY標準ヘッダー 19バイト]
[GPUPGParser拡張ヘッダー]
  - マジックナンバー: "GPUPG" (5バイト)
  - バージョン: 1 (1バイト)
  - テーブル名長: N (2バイト)
  - テーブル名: "customer" (Nバイト)
  - チャンクID: 0 (4バイト)
  - 総チャンク数: 2 (4バイト)
[実際のデータ]
```

## 実装の優先順位

### 簡単な実装（推奨）
**オプション1: ファイル名にテーブル名を含める**
- 実装が最も簡単
- 既存コードへの影響が最小限
- ファイル名を見るだけでテーブルが分かる

変更箇所：
1. Rust側: `chunk_path` を `{table_name}_chunk_{chunk_id}.bin` に変更
2. Python側: ファイル名パターンを更新

### 中程度の実装
**オプション2: メタデータファイル**
- チャンク処理前にメタデータを読み込み
- データ整合性チェックが可能
- ファイル管理が少し複雑になる

### 複雑な実装
**オプション3: カスタムヘッダー**
- 最も堅牢だが実装が複雑
- バイナリフォーマットの変更が必要
- GPU側のパーサーも修正が必要

## 推奨事項

短期的には**オプション1（ファイル名にテーブル名を含める）**を実装し、将来的にオプション2のメタデータファイルを追加することを推奨します。