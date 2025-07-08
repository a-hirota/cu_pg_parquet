# エレガントなファイル管理の提案

## 現在の課題
- 複数のジョブが同時実行される場合の衝突
- テーブル名の混同
- 中間ファイルの管理

## 解決策の提案

### 1. セッションベースのディレクトリ構造
```
/dev/shm/gpupgparser/
├── session_20240708_120000_pid12345_customer/
│   ├── metadata.json
│   ├── chunk_0.bin
│   ├── chunk_1.bin
│   └── .lock
└── session_20240708_120100_pid12346_lineorder/
    ├── metadata.json
    ├── chunk_0.bin
    └── chunk_1.bin
```

**利点:**
- 完全な分離
- 自動クリーンアップが容易
- メタデータ管理が明確

### 2. UUIDベースの命名
```python
import uuid
session_id = str(uuid.uuid4())[:8]
chunk_file = f"/dev/shm/{session_id}_{table_name}_chunk_{chunk_id}.bin"
```

**利点:**
- 衝突がほぼ不可能
- シンプルな実装

### 3. アトミックファイル操作
```python
# 一時ファイルを作成してから移動
temp_file = f"{chunk_file}.tmp.{os.getpid()}"
# データ書き込み
os.rename(temp_file, chunk_file)  # アトミック操作
```

### 4. 自動クリーンアップ機構
```python
class SessionManager:
    def __init__(self, table_name):
        self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}_{table_name}"
        self.session_dir = f"/dev/shm/gpupgparser/{self.session_id}"
        os.makedirs(self.session_dir, exist_ok=True)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # セッション終了時に自動削除
        shutil.rmtree(self.session_dir, ignore_errors=True)
```

### 5. メタデータファイルの活用
```json
// /dev/shm/gpupgparser/session_xxx/metadata.json
{
  "session_id": "20240708_120000_pid12345_customer",
  "table_name": "customer",
  "start_time": "2024-07-08T12:00:00Z",
  "pid": 12345,
  "chunks": [
    {
      "id": 0,
      "file": "chunk_0.bin",
      "size": 848104872,
      "rows": 6015118,
      "checksum": "sha256:..."
    }
  ],
  "status": "in_progress"
}
```

## 推奨実装

### 短期的解決（最小変更）
**現在の実装（テーブル名を含む）を維持**
- 実装済み
- 最小の変更で効果的
- 後方互換性あり

### 中期的改善
**セッションディレクトリ + メタデータ**
```python
# 例：benchmark_rust_gpu_direct.pyの修正
session_dir = f"/dev/shm/gpupgparser/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{TABLE_NAME}"
os.makedirs(session_dir, exist_ok=True)

# Rust側にセッションディレクトリを渡す
env['SESSION_DIR'] = session_dir
```

### 長期的理想
**完全なセッション管理システム**
- ジョブトラッキング
- 自動リトライ
- リソース管理
- 監視・ログ機能

## まとめ

現在実装したテーブル名を含むファイル名は、シンプルで効果的な解決策です。将来的には、より洗練されたセッション管理システムへの移行を検討できますが、現時点では十分な解決策と言えます。