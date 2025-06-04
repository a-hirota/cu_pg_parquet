# Decimal Column-wise Pass1統合最適化 実装ガイド

## 概要

Numba CUDAの制約（`List[DeviceNDArray]`非対応）を回避し、実用的なDecimal Pass1統合最適化を実現。

## 実装方針

### 従来版の問題点
```python
# 各Decimal列を個別にPass2で処理（メモリ二重アクセス）
for decimal_column in decimal_columns:
    pass2_scatter_decimal128(single_buffer)  # rawデータを再読み込み
```

### Column-wise最適化版の解決策
```python
# Pass1段階で各Decimal列を個別に統合処理
for decimal_column in decimal_columns:
    pass1_len_null_decimal_column_wise(
        raw_data,                    # 一度だけアクセス
        column_offsets,              # この列のみ
        column_output_buffer,        # 個別バッファ
        null_flags,                  # NULL情報
        scale_info                   # メタデータ
    )
```

## 核心技術

### 1. 列ごと統合カーネル
**ファイル**: `src/cuda_kernels/arrow_gpu_pass1_decimal_column_wise.py`

```python
@cuda.jit
def pass1_len_null_decimal_column_wise(
    raw,                # rawデータ
    field_offsets,      # この列のオフセット
    field_lengths,      # この列の長さ
    decimal_dst_buf,    # この列の出力バッファ
    stride,             # 16バイト
    target_scale,       # スケール
    d_pow10_table_lo,   # 10^nテーブル
    d_pow10_table_hi,
    d_nulls_col,        # この列のNULLフラグ
    var_idx,            # 可変長インデックス
    d_var_lens          # 可変長長さ配列
):
    # 1行ごとに以下を統合実行:
    # 1. NULL判定
    # 2. 可変長長さ記録
    # 3. Decimalバイナリ解析
    # 4. 128ビット整数変換
    # 5. 出力バッファ書き込み
```

### 2. デコーダー統合
**ファイル**: `src/gpu_decoder_v2_decimal_column_wise.py`

```python
# Pass1: 非Decimal列は従来処理
pass1_len_null_non_decimal[blocks, threads](
    field_lengths_dev,
    var_indices_dev,
    d_var_lens,
    d_nulls_all,
    decimal_cols_mask  # Decimal列を除外
)

# Pass1: Decimal列は統合処理（列ごと）
for decimal_column in decimal_columns:
    pass1_len_null_decimal_column_wise[blocks, threads](
        raw_dev,
        field_offsets_dev[:, cidx],    # 1列分
        field_lengths_dev[:, cidx],    # 1列分
        decimal_output_buffer,         # 1列分
        # ...
    )

# Pass2: Decimal列はスキップ（既に処理済み）
```

## メモリアクセス最適化効果

### 従来版
```
Pass1: field_lengths読み取り（Decimal列も含む）
Pass2: rawデータ再読み取り（Decimal列のみ）
合計: Decimal列につき2回のメモリアクセス
```

### Column-wise最適化版
```
Pass1: rawデータ読み取り + 即座に変換（Decimal列）
Pass2: Decimal列はスキップ
合計: Decimal列につき1回のメモリアクセス
```

**理論効果**: Decimal列数 × 行数 分のメモリアクセス削減

## 実装の特徴

### ✅ **利点**
1. **Numba互換**: `List[DeviceNDArray]`を使わない
2. **構造保持**: 従来のGPUMemoryManagerV2との整合性
3. **段階的適用**: Decimal列のみ最適化、他は従来処理
4. **スケーラブル**: Decimal列数増加に比例して効果拡大

### ⚠️ **制約**
1. **カーネル呼び出し増加**: Decimal列数分のカーネル起動
2. **実装複雑度**: 統合カーネルの保守性
3. **バッファ管理**: 列ごとの個別管理が必要

## 使用方法

### 基本テスト
```bash
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'
python test_decimal_column_wise_optimization.py
```

### ベンチマーク比較
```bash
# Column-wise最適化版
export DECIMAL_OPTIMIZATION_MODE=column_wise
python benchmark/benchmark_lineorder_5m.py

# 従来版
export DECIMAL_OPTIMIZATION_MODE=traditional  
python benchmark/benchmark_lineorder_5m.py

# Integrated版（参考）
export DECIMAL_OPTIMIZATION_MODE=integrated
python benchmark/benchmark_lineorder_5m.py
```

## 期待される性能改善

### **対象ワークロード**
- Decimal列数: 5列以上
- データ行数: 10万行以上
- Decimal精度: 10桁以上

### **期待効果**
- **メモリ帯域効率**: 20-40%向上
- **全体処理時間**: 10-25%短縮
- **GPU利用率**: メモリバウンド解消

### **スケーリング特性**
```
高速化率 ≈ 1 + (decimal_columns / total_columns) × 0.3
```

例: 17列中10列がDecimal → 約1.18倍の高速化期待

## トラブルシューティング

### Numbaエラー
```python
# ❌ 問題のあるコード
decimal_buffers = [buf1, buf2, buf3]  # List[DeviceNDArray]
kernel(decimal_buffers)               # NumbaError

# ✅ 修正版
for buffer in [buf1, buf2, buf3]:     # 個別処理
    kernel(buffer)                    # OK
```

### スケール不一致
```python
# メタデータでスケール確認
precision, scale = column.arrow_param or (38, 0)
print(f"Column {column.name}: scale={scale}")
```

### NULL処理
```python
# NULL判定の統一
is_null = (field_length == -1)
d_nulls[row, col] = uint8(0 if is_null else 1)
```

## 今後の拡張

### 1. **他データ型への適用**
- Timestamp: タイムゾーン変換の統合
- JSON: パース処理の前倒し
- UUID: フォーマット変換の統合

### 2. **マルチGPU対応**
- 列分散処理
- GPUメモリ効率化

### 3. **動的最適化**
- ワークロード特性に応じた自動切り替え
- 列数・行数に基づく最適戦略選択

## 結論

Column-wise Pass1統合最適化は、Numbaの制約下で実用的なDecimal処理最適化を実現する効果的なアプローチです。特に、複数のDecimal列を持つ大規模データセットにおいて、メモリアクセス効率の大幅な改善が期待できます。