# GPU PostgreSQL Parser 必須テスト計画（厳選版）

## 1. 概要

本ドキュメントは、すぐに実装を開始できる最小限かつ効果的なテストセットを定義します。
全体の80%の品質を20%の労力で達成することを目指します。

## 2. 最優先テスト（第1週で実装）

### 2.1 スモークテスト - 基本動作確認

```yaml
SMOKE-1: 最小構成での動作確認
  目的: システムが最低限動作することを確認
  手順:
    1. 10行のテストテーブル作成（INT, TEXT, TIMESTAMPの3カラム）
    2. PostgreSQL → Parquet変換実行
    3. 出力ファイルの存在確認
  期待結果: エラーなく完了、行数一致
  実装時間: 2時間
```

### 2.2 基本データ型テスト - よく使う型のみ

```yaml
TYPE-1: 必須データ型の変換確認
  対象型:
    - INTEGER（数値の代表）
    - TEXT（文字列の代表）
    - TIMESTAMP（日時の代表）
    - BOOLEAN（真偽値）
    - NULL値
  
  テストデータ:
    - 通常値: 100行
    - NULL値: 各型で10%含む
    - 境界値: 最小値、最大値を1行ずつ
  
  検証: 値の完全一致
  実装時間: 4時間
```

### 2.3 基本性能テスト - ベースライン確立

```yaml
PERF-1: 1GBデータの処理時間
  目的: 性能のベースライン確立
  データ: 1GB、1000万行、10カラム
  測定:
    - 全体処理時間
    - メモリ使用量ピーク
    - GPU使用率
  合格基準: 10分以内に完了
  実装時間: 3時間
```

## 3. 第2優先テスト（第2週で実装）

### 3.1 エラーハンドリング基本テスト

```yaml
ERROR-1: 接続エラー処理
  - PostgreSQL停止中の実行
  - 処理中の接続切断
  期待: エラーメッセージ表示、クリーンな終了

ERROR-2: リソース不足処理  
  - GPUメモリ不足（大きなテーブル処理）
  - ディスク容量不足
  期待: 適切なエラー、部分的な成功なし
```

### 3.2 可変長データ型テスト

```yaml
TYPE-2: 可変長型の基本動作
  対象:
    - VARCHAR(100) - 短い文字列
    - TEXT - 長い文字列（最大1MB）
    - 日本語文字列
  検証: エンコーディング保持、長さ一致
```

### 3.3 実用的な統合テスト

```yaml
E2E-1: 実際のユースケース
  シナリオ: 売上データの月次集計
  - 100万行の売上テーブル
  - 日付でパーティション分割
  - 圧縮有効（ZSTD）
  検証: ビジネスロジックでの利用可能性
```

## 4. 第3優先テスト（必要に応じて実装）

### 4.1 特殊データ型（使用頻度による）
- DECIMAL（金額計算が必要な場合）
- ARRAY（配列データを扱う場合）
- JSON/JSONB（JSONデータを扱う場合）

### 4.2 スケーラビリティ（大規模運用時）
- 10GB以上のテーブル
- マルチGPU処理
- 並行処理

### 4.3 互換性（環境が増えた時）
- PostgreSQL バージョン違い
- 異なるGPUモデル
- 各種Parquetリーダー

## 5. テスト実装ガイド

### 5.1 テストデータ準備スクリプト

```python
# tests/prepare_test_data.py
import psycopg2
import random
from datetime import datetime, timedelta

def create_basic_test_table(conn, table_name, row_count=1000):
    """基本テストテーブル作成"""
    cur = conn.cursor()
    
    # テーブル作成
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value INTEGER,
            created_at TIMESTAMP,
            is_active BOOLEAN
        )
    """)
    
    # データ投入
    for i in range(row_count):
        cur.execute(f"""
            INSERT INTO {table_name} VALUES (
                %s, %s, %s, %s, %s
            )
        """, (
            i,
            f'test_name_{i}' if i % 10 != 0 else None,  # 10%をNULL
            random.randint(0, 1000000),
            datetime.now() - timedelta(days=random.randint(0, 365)),
            random.choice([True, False, None])
        ))
    
    conn.commit()
    print(f"Created test table {table_name} with {row_count} rows")
```

### 5.2 基本テスト実行スクリプト

```python
# tests/test_basic.py
import subprocess
import time
import psutil
import cupy as cp

def test_smoke():
    """スモークテスト実行"""
    start_time = time.time()
    
    # テスト実行
    result = subprocess.run([
        'python', 'cu_pg_parquet.py',
        '--table', 'test_basic',
        '--output', 'output/test_basic.parquet'
    ], capture_output=True, text=True)
    
    # 基本検証
    assert result.returncode == 0, f"Failed: {result.stderr}"
    assert os.path.exists('output/test_basic.parquet')
    
    # 性能記録
    duration = time.time() - start_time
    print(f"Smoke test passed in {duration:.2f} seconds")
    
    # データ検証
    df = pd.read_parquet('output/test_basic.parquet')
    assert len(df) == 1000, f"Row count mismatch: {len(df)}"
    
    return True

def test_performance_baseline():
    """性能ベースラインテスト"""
    # メモリ使用量監視開始
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # GPU使用率監視
    gpu_memory_before = cp.cuda.MemoryPool().used_bytes() / 1024 / 1024
    
    # 1GBテーブル処理
    start_time = time.time()
    result = subprocess.run([
        'python', 'cu_pg_parquet.py',
        '--table', 'test_1gb',
        '--output', 'output/test_1gb.parquet'
    ])
    
    # 測定結果
    duration = time.time() - start_time
    peak_memory = process.memory_info().rss / 1024 / 1024
    memory_used = peak_memory - initial_memory
    
    print(f"""
    Performance Baseline Results:
    - Duration: {duration:.2f} seconds
    - Memory Used: {memory_used:.2f} MB
    - Throughput: {1024 / duration:.2f} MB/s
    """)
    
    # 合格基準
    assert duration < 600, f"Too slow: {duration} seconds"
    assert memory_used < 4096, f"Too much memory: {memory_used} MB"
```

### 5.3 継続的インテグレーション設定

```yaml
# .github/workflows/essential-tests.yml
name: Essential Tests
on: [push, pull_request]

jobs:
  essential-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup PostgreSQL
        run: |
          sudo apt-get install postgresql
          sudo -u postgres createdb testdb
          
      - name: Prepare test data
        run: python tests/prepare_test_data.py
        
      - name: Run smoke tests
        run: pytest tests/test_basic.py::test_smoke -v
        
      - name: Run type tests
        run: pytest tests/test_basic.py::test_types -v
        
      - name: Check performance
        if: github.ref == 'refs/heads/main'
        run: pytest tests/test_basic.py::test_performance_baseline -v
```

## 6. テスト結果判定基準

### 6.1 必須合格基準（リリースブロッカー）
- [ ] スモークテスト全項目合格
- [ ] 基本データ型テスト合格
- [ ] 性能ベースラインを満たす

### 6.2 推奨合格基準
- [ ] エラーハンドリングテスト合格
- [ ] 可変長データ型テスト合格
- [ ] 実用的統合テスト合格

### 6.3 メトリクス記録
```
テストID | 実行時間 | メモリ使用 | 備考
---------|----------|-----------|-----
SMOKE-1  | 0.5秒    | 100MB     | OK
TYPE-1   | 2.3秒    | 250MB     | OK
PERF-1   | 85秒     | 2.1GB     | OK
```

## 7. トラブルシューティング

### よくある問題と対処

1. **GPUメモリ不足エラー**
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # 特定GPUのみ使用
   export GPUPGPARSER_CHUNK_SIZE=10000  # チャンクサイズ縮小
   ```

2. **PostgreSQL接続エラー**
   ```bash
   # 接続情報確認
   psql -h localhost -U postgres -l
   # 環境変数設定
   export GPUPASER_PG_DSN="host=localhost dbname=testdb"
   ```

3. **性能が目標に達しない**
   - GPUウォームアップを追加
   - データのプリロード
   - チャンクサイズ最適化

## 8. 次のステップ

1. **第1週**: 最優先テスト実装（約20時間）
   - 月曜: 環境構築とスモークテスト
   - 火-水: 基本データ型テスト
   - 木-金: 性能ベースライン確立

2. **第2週**: 第2優先テスト実装（約30時間）
   - エラーハンドリング追加
   - 可変長型対応
   - 実用シナリオ検証

3. **継続的改善**
   - テスト実行時間の短縮
   - カバレッジの段階的拡大
   - 自動化の強化