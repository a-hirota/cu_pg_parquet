# 64MB境界問題の解決策

## 問題の概要
Rust側で64MBバッファを使用してPostgreSQLデータを/dev/shmに書き込む際、バッファ境界で行が分断され、GPUが検出できない。

## 根本原因
- `rust_bench_optimized/src/main_single_chunk.rs`で`BUFFER_SIZE = 64MB`
- バッファがいっぱいになると強制的にフラッシュ
- PostgreSQL COPY BINARYの行が境界をまたぐと分断される

## 解決策

### 1. 短期的解決策（簡単）
バッファサイズを増やして問題の発生頻度を下げる：
```rust
const BUFFER_SIZE: usize = 256 * 1024 * 1024;  // 256MBに増加
```

### 2. 中期的解決策（推奨）
行境界を意識したバッファリング：
```rust
// 最大行サイズを定義
const MAX_ROW_SIZE: usize = 10 * 1024;  // 10KB

// バッファフラッシュ前に余裕を確認
if write_buffer.len() + MAX_ROW_SIZE >= BUFFER_SIZE {
    // 現在のバッファをフラッシュ
    flush_buffer(&mut write_buffer, &chunk_file, &worker_offset)?;
}
```

### 3. 長期的解決策（最善）
PostgreSQL COPY BINARYプロトコルを解析して行境界を正確に検出：

```rust
// 行ヘッダーを読んで行の完全性を保証
fn ensure_complete_row(buffer: &[u8], pos: usize) -> Option<usize> {
    if pos + 2 > buffer.len() { return None; }
    
    let num_fields = u16::from_be_bytes([buffer[pos], buffer[pos+1]]);
    let mut current_pos = pos + 2;
    
    for _ in 0..num_fields {
        if current_pos + 4 > buffer.len() { return None; }
        
        let field_len = i32::from_be_bytes([
            buffer[current_pos], buffer[current_pos+1],
            buffer[current_pos+2], buffer[current_pos+3]
        ]);
        
        current_pos += 4;
        if field_len > 0 {
            current_pos += field_len as usize;
        }
    }
    
    if current_pos <= buffer.len() {
        Some(current_pos)
    } else {
        None
    }
}
```

## GPU側の対応

### 1. データ整合性チェック
```python
# 64MB境界付近で特別な処理
if thread_start_pos % (64 * 1024 * 1024) < 1024:
    # 境界付近では慎重に行ヘッダーを検証
    validate_row_header_carefully()
```

### 2. オーバーラップ読み込み
```python
# スレッドが境界付近の場合、少し余分に読む
if is_near_64mb_boundary(thread_start_pos):
    overlap_size = 256  # 256バイトのオーバーラップ
    actual_start = max(0, thread_start_pos - overlap_size)
```

## テスト方法

1. 64MB境界をまたぐテストデータを作成
2. 欠落行が発生しないことを確認
3. パフォーマンスへの影響を測定

## 実装優先度

1. **即座に実装**: バッファサイズを256MBに増加（問題を緩和）
2. **次に実装**: 行境界を意識したバッファリング
3. **将来的に実装**: 完全なプロトコル解析による解決