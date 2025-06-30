# lineorderテーブル処理の行数不足問題の調査結果

## 問題の概要
- **期待される行数**: 246,012,324行（PostgreSQL実際の行数）
- **処理された行数**: 148,913,189行（約60.5%）
- **不足行数**: 97,099,135行（約39.5%）

## 根本原因
Rust側のチャンク分割処理で、複数のワーカーが同じファイル位置に書き込んでいることが原因です。

### 具体的な問題点

1. **ワーカーのオフセット管理の問題**
   - 16個のワーカーが全て`offset: 0`から書き込みを開始
   - 各ワーカーは独立して動作し、`write_all_at`で書き込むが、同じ位置に書き込んでいる
   - 結果として、データが上書きされる

2. **64MB境界での破損**
   - 各ワーカーは64MBのバッファを使用
   - 位置67,109,077（約64MB）でデータ破損が発生
   - これは最初のバッファフラッシュ時に他のワーカーのデータを上書きしている証拠

3. **実際のデータ量**
   - チャンクファイルサイズ: 3.34GB
   - 正常に読み取れたタプル: 292,378個（期待値の約1.9%）
   - これは1つのワーカーの最初のバッファ分のみが正しく保存されていることを示す

## 解決策

### 方法1: シーケンシャル書き込み（推奨）
各ワーカーが順番に書き込むように変更：

```rust
// 各ワーカーに固定オフセットを割り当てる
let worker_data_size = (end_page - start_page) as u64 * ESTIMATED_BYTES_PER_PAGE;
let worker_start_offset = worker_id as u64 * worker_data_size;

// または、先にデータサイズを計算してから書き込む
```

### 方法2: 別ファイルに書き込み後マージ
各ワーカーが別々のファイルに書き込み、最後にマージ：

```rust
let worker_file = format!("{}/chunk_{}_worker_{}.bin", OUTPUT_DIR, chunk_id, worker_id);
// 各ワーカーは自分のファイルに書き込む
// 最後に全ファイルをマージ
```

### 方法3: アトミックオフセット管理の修正（現在の方式を活かす）
`worker_offset`の使用方法を修正：

```rust
// process_range関数内で
let mut current_offset = 0u64;
let mut write_buffer = Vec::with_capacity(BUFFER_SIZE);

// バッファフラッシュ時
if write_buffer.len() >= BUFFER_SIZE {
    let write_offset = worker_offset.fetch_add(write_buffer.len() as u64, Ordering::SeqCst);
    chunk_file.write_all_at(&write_buffer, write_offset)?;
    current_offset = write_offset + write_buffer.len() as u64;
    write_buffer.clear();
}
```

## 推奨アクション

1. **即座の修正**: 方法3（アトミックオフセット管理の修正）を実装
2. **長期的な改善**: 方法1（シーケンシャル書き込み）への移行を検討
3. **テスト**: 修正後、全16チャンクで正しい行数が処理されることを確認

## 検証方法

修正後、以下を確認：
- 各チャンクの実際のタプル数が理論値と一致
- 全チャンクの合計行数が246,012,324行になる
- データ破損がないこと（64MB境界を超えても正常に読み取れる）