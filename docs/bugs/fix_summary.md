# c_custkey=0 問題の修正報告

## 問題の概要
customerテーブルのパース時に、c_custkey（最初のカラム）が時々0として解析され、全てのフィールドが1つずつシフトする問題が発生していました。

### 症状
- c_custkey: 0 (本来は 1220008, 2300008 など)
- c_name: アドレスデータが入る
- c_address: 都市データが入る
- 以降のフィールドも全て1つずつシフト

## 根本原因
`src/postgres_to_cudf.py`の`_extract_fixed_direct`および`_extract_decimal_direct`メソッドで、以下の問題のあるコードがありました：

```python
if is_null or src_offset == 0:
    # NULLとして扱い、ゼロで埋める
```

このコードは、`src_offset == 0`の場合も NULL として扱っていました。しかし、最初のフィールドの相対オフセットが0になることは正常であり、これを NULL として扱うのは誤りでした。

## 実施した修正

### 1. postgres_to_cudf.py の修正
以下の2つのメソッドで条件を修正：
- `_extract_fixed_direct`: 行418
- `_extract_decimal_direct`: 行494

修正内容：
```python
# 修正前
if is_null or src_offset == 0:

# 修正後
if is_null:
```

### 2. postgres_binary_parser.py の確認
NULLフィールドのオフセット処理が正しく実装されていることを確認：
```python
if flen == 0xFFFFFFFF:  # NULL
    # NULLでも正しいオフセットを記録（pos+4の位置）
    relative_offset = uint32((pos + 4) - row_start)
    field_offsets_out[field_idx] = relative_offset
    field_lengths_out[field_idx] = -1
```

## 修正結果
- ✅ c_custkeyが正しく解析されるようになりました
- ✅ フィールドのシフト問題が解決されました
- ✅ 全てのカラムデータが正しい位置に配置されるようになりました

## テスト結果
修正後の動作確認：
1. c_custkey=0のレコードが存在しないことを確認
2. PostgreSQLのデータと一致することを確認
3. 複数回実行しても安定して正しい結果が得られることを確認

## 今後の推奨事項
1. 単体テストの追加：最初のフィールドのオフセットが0の場合のテストケース
2. 境界値テスト：様々なオフセット値でのテスト
3. 回帰テストの実施：この修正が他のテーブルに影響しないことの確認