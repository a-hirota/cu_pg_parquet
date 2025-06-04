# Decimal Pass1統合最適化実装ガイド

## 概要

PostgreSQL COPY BINARY形式のDecimal（NUMERIC）データをGPUで処理する際の最適化手法として、**Pass1段階でのDecimal先行変換**を実装しました。この最適化により、メモリアクセス回数の削減とカーネル起動オーバーヘッドの削減を実現します。

## アーキテクチャ

### 従来の2パス処理

```
Pass1: 行並列でNULL/長さ収集のみ
├─ 各Decimalフィールドを読み飛ばし
└─ 長さ情報だけを記録

Pass2: 列並列でDecimal変換
├─ 同じDecimalフィールドを再度読み込み
├─ バイナリ解析→128bit整数変換
└─ Arrow形式で出力
```

**問題点:**
- Decimalフィールドの**二重読み込み**（Pass1で読み飛ばし→Pass2で再読み込み）
- **カーネル起動オーバーヘッド**（Decimal列数分のPass2カーネル起動）
- **メモリ帯域の非効率利用**

### 最適化版統合処理

```
Pass1統合: 行並列でNULL/長さ収集 + Decimal変換
├─ 各行のDecimalフィールドを一度だけ読み込み
├─ その場でバイナリ解析→128bit整数変換
├─ 結果を直接出力バッファに書き込み
└─ NULL/長さ情報も同時収集

Pass2軽量: Decimal以外の列のみ処理
├─ Decimal列はスキップ（Pass1で処理済み）
└─ 固定長・可変長列のみ処理
```

**効果:**
- **メモリアクセス50%削減**（Decimalフィールドの読み込み1回化）
- **カーネル起動削減**（Decimal列数分のPass2カーネルが不要）
- **GPU利用率向上**（早期の重処理実行によるパイプライン最適化）

## 実装ファイル

### 1. 統合カーネル実装
**ファイル:** `src/cuda_kernels/arrow_gpu_pass1_decimal_optimized.py`

#### 主要コンポーネント

```python
@cuda.jit
def pass1_len_null_decimal_integrated(
    field_lengths, field_offsets, raw,
    var_indices, decimal_indices, decimal_scales,
    decimal_buffers, d_var_lens, d_nulls,
    d_pow10_table_lo, d_pow10_table_hi
):
    """Pass1統合カーネル: NULL/長さ収集 + Decimal変換を同時実行"""
```

#### 特徴
- **共有メモリ最適化**: 10^nテーブルをワープ協調でロード
- **基数1e8最適化**: PostgreSQL基数10000の2桁を結合処理
- **インライン関数活用**: 128ビット演算の高速化
- **エラーハンドリング**: NaN・オーバーフロー・NULL適切な処理

### 2. 最適化デコーダー実装
**ファイル:** `src/gpu_decoder_v2_decimal_optimized.py`

#### 主要API

```python
def decode_chunk_decimal_optimized(
    raw_dev, field_offsets_dev, field_lengths_dev, columns,
    use_pass1_integration: bool = True
) -> pa.RecordBatch:
    """
    Decimal Pass1統合最適化版のデコーダー
    
    Parameters:
    -----------
    use_pass1_integration : bool
        True: 最適化版（Pass1統合）
        False: 従来版（互換性確認用）
    """
```

#### 処理フロー
1. **メタデータ分析**: Decimal列の検出とインデックス構築
2. **Pass1統合実行**: 統合カーネルでDecimal変換とNULL/長さ収集
3. **Prefix Sum**: 可変長列のオフセット計算（従来と同様）
4. **Pass2軽量**: Decimal以外の列のみ処理
5. **Arrow組立**: 従来と同じロジックでRecordBatch構築

### 3. ベンチマークテスト
**ファイル:** `test_decimal_pass1_optimization.py`

#### 機能
- **性能比較**: 従来版vs最適化版の実行時間測定
- **正確性検証**: 結果の一致性確認
- **スケーラビリティ評価**: 異なる行数・列数での効果測定
- **統計分析**: Decimal列数と効果の相関分析

## 使用方法

### 1. 基本的な使用

```python
from src.gpu_decoder_v2_decimal_optimized import decode_chunk_decimal_optimized

# 最適化版を使用
result = decode_chunk_decimal_optimized(
    raw_dev, field_offsets_dev, field_lengths_dev, columns,
    use_pass1_integration=True  # 最適化有効
)

# 従来版と比較（デバッグ用）
result_traditional = decode_chunk_decimal_optimized(
    raw_dev, field_offsets_dev, field_lengths_dev, columns,
    use_pass1_integration=False  # 最適化無効
)
```

### 2. 環境変数による制御

```bash
# Decimal最適化を有効化（デフォルト）
export USE_DECIMAL_PASS1_OPTIMIZATION=1

# 従来版を使用（比較・デバッグ用）
export USE_DECIMAL_PASS1_OPTIMIZATION=0
```

### 3. ベンチマーク実行

```bash
# 包括的ベンチマークの実行
python test_decimal_pass1_optimization.py

# 結果例
# Rows     DecCols  Traditional  Optimized    Speedup
# 1000     1        0.1234s      0.0987s      1.25x
# 10000    5        0.5678s      0.4321s      1.31x
# 50000    10       2.3456s      1.8901s      1.24x
```

## 技術的詳細

### 共有メモリ最適化

```cuda
__shared__ uint64_t shared_pow10_lo[39];
__shared__ uint64_t shared_pow10_hi[39];

// ワープ協調ロード
load_pow10_to_shared(d_pow10_table_lo, d_pow10_table_hi, 
                     shared_pow10_lo, shared_pow10_hi);
```

**効果:**
- グローバルメモリアクセス削減
- 10^n定数の高速アクセス
- ワープ内での効率的な定数共有

### 基数1e8最適化

```cuda
// 2桁の基数10000を結合して基数1e8として処理
if (i + 1 < nd) {
    digit_high = read_digit(src_pos);
    digit_low = read_digit(src_pos + 2);
    combined = digit_high * 10000 + digit_low;
    
    val = val * 1e8 + combined;  // 1回の乗算で2桁処理
    i += 2;
}
```

**効果:**
- ループ回数の削減（理論上50%削減）
- 乗算回数の削減
- レジスタ利用効率の向上

### 128ビット演算最適化

```cuda
@cuda.jit(device=True, inline=True)
def mul128_u64_optimized(a_hi, a_lo, b):
    """32ビット分割による高速128×64ビット乗算"""
    // 32ビット要素への分割
    // 部分積計算
    // 高速組み立て
    return res_hi, res_lo
```

**効果:**
- Decimal変換の高速化
- 分岐削減による並列効率向上
- インライン展開による呼び出しオーバーヘッド削減

## 性能特性

### 理論的効果

| 最適化項目 | 従来版 | 最適化版 | 改善率 |
|------------|--------|----------|---------|
| Decimalフィールド読み込み | 2回 | 1回 | 50%削減 |
| カーネル起動回数 | 2+N | 2 | N削減 |
| GPU演算並列度 | 列並列 | 行並列 | 向上 |

（N = Decimal列数）

### 実測効果（予想）

| シナリオ | Decimal列数 | 期待効果 |
|----------|-------------|----------|
| 軽負荷 | 1-2列 | 1.1-1.2x |
| 中負荷 | 3-5列 | 1.2-1.4x |
| 重負荷 | 5-10列 | 1.3-1.6x |

### 適用条件

**効果が高い場合:**
- Decimal列が複数存在
- 行数が多い（1万行以上）
- Decimalデータのprecisionが高い

**効果が限定的な場合:**
- Decimal列が少ない（1列以下）
- 行数が少ない（1千行以下）
- 他の処理がボトルネック

## トラブルシューティング

### 1. コンパイルエラー

```
numba.core.errors.TypingError: Failed in cuda mode pipeline
```

**対処法:**
- CUDAカーネル内でのPython標準ライブラリ使用を避ける
- 型アノテーションを明示的に指定
- デバイス関数の`@cuda.jit(device=True)`を確認

### 2. 実行時エラー

```
CUDA_ERROR_INVALID_VALUE
```

**対処法:**
- GPUメモリサイズの確認
- バッファサイズの適切性確認
- NULLポインタアクセスの回避

### 3. 性能が改善しない

**確認項目:**
- `use_pass1_integration=True`が設定されているか
- Decimal列が実際に存在するか
- GPU使用率の確認（`nvidia-smi`）
- メモリ帯域の使用状況確認

### 4. 結果の不一致

**デバッグ手順:**
1. 小規模データでの比較テスト
2. 個別Decimal値の手動検証
3. NULL処理の確認
4. スケール変換の検証

## 今後の拡張可能性

### 1. さらなる最適化案

- **Decimal64対応**: precision≤18の場合の64ビット処理
- **Warp Shuffle活用**: ワープ内でのデータ協調処理
- **テンソルコア活用**: 行列演算での高速化
- **マルチGPU対応**: 複数GPU間での並列処理

### 2. 他データ型への適用

- **Timestamp変換**: 日時データの先行処理
- **JSON解析**: 半構造化データの先行処理
- **配列データ**: PostgreSQL配列型の先行処理

### 3. 自動最適化

- **列特性分析**: 実行時の最適カーネル選択
- **性能予測**: コスト分析による自動切り替え
- **適応的チューニング**: ワークロード学習による最適化

## 結論

Pass1段階でのDecimal先行変換により、以下の効果が期待できます：

1. **メモリアクセス効率化**: 二重読み込み削減による帯域使用量の最適化
2. **カーネル起動削減**: Pass2 Decimalカーネルの完全省略
3. **GPU利用率向上**: 早期重処理実行によるパイプライン最適化
4. **スケーラビリティ向上**: Decimal列数増加時の線形性能改善

この最適化は特に、複数のDecimal列を持つ大規模データセットの処理において顕著な効果を発揮すると予想されます。