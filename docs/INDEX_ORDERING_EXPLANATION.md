# インデックス順序の違いとは？
==============================

## 基本概念

### field_offsets_dev[rows, cols]とfield_lengths_dev[rows, cols]

これらは2次元配列で、以下のような構造です：

```
field_offsets_dev[行インデックス, 列インデックス] = データ開始位置
field_lengths_dev[行インデックス, 列インデックス] = データ長
```

## 具体例で説明

### 元データ（PostgreSQLバイナリ）
```
物理位置100: 行A → [ID=3, 名前="鈴木"]
物理位置250: 行B → [ID=1, 名前="田中"] 
物理位置400: 行C → [ID=2, 名前="佐藤"]
```

### GPU並列検出結果（検出順序）
```
GPU検出順序: [行A, 行C, 行B]
物理位置順序: [100, 400, 250]
```

### 配列への格納

**ソート前の状態**:
```
field_offsets_dev[0, 0] = 104  // 行A のID開始位置
field_offsets_dev[0, 1] = 108  // 行A の名前開始位置
field_offsets_dev[1, 0] = 404  // 行C のID開始位置  
field_offsets_dev[1, 1] = 408  // 行C の名前開始位置
field_offsets_dev[2, 0] = 254  // 行B のID開始位置
field_offsets_dev[2, 1] = 258  // 行B の名前開始位置

field_lengths_dev[0, 0] = 4    // 行A のID長
field_lengths_dev[0, 1] = 6    // 行A の名前長
field_lengths_dev[1, 0] = 4    // 行C のID長
field_lengths_dev[1, 1] = 6    // 行C の名前長  
field_lengths_dev[2, 0] = 4    // 行B のID長
field_lengths_dev[2, 1] = 6    // 行B の名前長
```

**問題**: 行インデックス0,1,2が物理位置順序[100,400,250]に対応
→ 行インデックス1が実際には3番目の行データを指している

## ソート処理の詳細

### 1. 物理位置の並び替え
```python
row_positions = [100, 400, 250]  # GPU検出順序
sort_indices = np.argsort(row_positions)  # [0, 2, 1]
# 意味: インデックス0→0番目、インデックス2→1番目、インデックス1→2番目
```

### 2. インデックス順序の違いが発生するケース

**重複値がある場合**:
```python
# 例：同じ物理位置が複数検出された場合
row_positions = [100, 250, 100, 400]  # 重複あり

# CPUソート結果
cpu_sort_indices = [0, 2, 1, 3]  # NumPyの安定ソート

# GPUソート結果  
gpu_sort_indices = [2, 0, 1, 3]  # CuPyの異なる安定性

# どちらも物理的には正しい順序: [100, 100, 250, 400]
# しかし、同じ値の並び順（インデックス順序）が異なる
```

## field_offsets_dev, field_lengths_devへの影響

### ソート後の正しい状態
```
// 物理位置順: [100, 250, 400] = [行A, 行B, 行C]

field_offsets_dev[0, 0] = 104  // 行A のID開始位置
field_offsets_dev[0, 1] = 108  // 行A の名前開始位置  
field_offsets_dev[1, 0] = 254  // 行B のID開始位置
field_offsets_dev[1, 1] = 258  // 行B の名前開始位置
field_offsets_dev[2, 0] = 404  // 行C のID開始位置
field_offsets_dev[2, 1] = 408  // 行C の名前開始位置
```

### インデックス順序の違いの例

**CPUソート結果**:
```
sort_indices = [0, 2, 1]  // 行A→0, 行B→1, 行C→2
field_offsets_sorted[0] = field_offsets_original[0]  // 行A
field_offsets_sorted[1] = field_offsets_original[2]  // 行B  
field_offsets_sorted[2] = field_offsets_original[1]  // 行C
```

**GPUソート結果（重複時の異なる順序）**:
```
sort_indices = [0, 2, 1]  // 同じ最終結果だが、
// 内部的に重複値の処理順序が異なる可能性
```

## 実際の確認ケース

テストで発生した状況：
```python
# 両方とも同じソート結果
CPU sort indices: [1528, 4935, 9032, 9590, 8156, ...]
GPU sort indices: [1528, 4935, 9032, 9590, 8156, ...]

# ソート後の物理位置も一致
sorted_positions_cpu: [54886, 110268, 121958, 131932, ...]
sorted_positions_gpu: [54886, 110268, 121958, 131932, ...]
```

## 実際のfield_offsets_dev[rows, cols]操作例

### GPU並列検出の実際の流れ

```python
# 1. GPU並列検出で得られる結果（順序が乱れている）
row_positions_detected = [100, 400, 250]  # GPU検出順序
field_offsets_detected = [
    [104, 108],  # 行A: ID位置104, 名前位置108
    [404, 408],  # 行C: ID位置404, 名前位置408
    [254, 258]   # 行B: ID位置254, 名前位置258
]

# 2. ソート処理
sort_indices = np.argsort([100, 400, 250])  # [0, 2, 1]

# 3. field_offsets_dev[rows, cols]の並び替え
field_offsets_sorted = field_offsets_detected[sort_indices]
# 結果:
# field_offsets_sorted = [
#     [104, 108],  # インデックス0: 行A（物理位置100）
#     [254, 258],  # インデックス1: 行B（物理位置250）
#     [404, 408]   # インデックス2: 行C（物理位置400）
# ]
```

### 列指向バッファでの使用

```python
# ソート後のfield_offsets_dev[rows, cols]を使用
for row_idx in range(3):  # 行インデックス 0, 1, 2
    for col_idx in range(2):  # 列インデックス 0, 1 (ID, 名前)
        offset = field_offsets_dev[row_idx, col_idx]
        length = field_lengths_dev[row_idx, col_idx]
        
        # raw_devからデータを抽出
        data = raw_dev[offset:offset+length]
        
        # 列指向バッファに格納
        column_buffer[col_idx][row_idx] = data

# 正しい結果:
# column_buffer[0] = [行A_ID, 行B_ID, 行C_ID]  # ID列
# column_buffer[1] = [行A_名前, 行B_名前, 行C_名前]  # 名前列
```

### インデックス順序の違いが許容される理由

**重複値の例**:
```python
# 同じ物理位置に複数行が検出された場合
row_positions = [100, 250, 100, 400]

# CPUソート: 安定ソート（元の順序を保持）
cpu_indices = [0, 2, 1, 3]  # 100(1回目), 100(2回目), 250, 400

# GPUソート: 異なる安定性実装
gpu_indices = [2, 0, 1, 3]  # 100(2回目), 100(1回目), 250, 400

# どちらも物理的に正しい順序: [100, 100, 250, 400]
# field_offsets_dev[rows, cols]の最終内容も同じ
```

## 結論

「インデックス順序の違いは許容」とは：

1. **最終的な物理位置順序が一致**していれば正しい
2. **field_offsets_dev[rows, cols]の最終配列内容**が同じであれば問題なし
3. **重複値の内部処理順序**は実装により異なる可能性がある
4. **列指向バッファへの書き込み結果**が同一であることが重要

**重要ポイント**:
- `rows`: 行インデックス（0,1,2...）は配列の添字
- `cols`: 列インデックス（0,1,2...）は列の添字
- 最終的にfield_offsets_dev[row_idx, col_idx]で正しいデータ位置を参照できることが本質

**実用的な確認方法**:
1. ソート後のrow_positions順序が物理順序と一致
2. field_offsets_dev[0,0]が最初の行の最初の列を正しく指す
3. 列指向バッファの構築結果が期待通り