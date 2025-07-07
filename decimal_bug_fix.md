# Decimal パースバグの原因と修正方法

## バグの原因

`src/postgres_to_cudf.py`の`_extract_decimal_direct`カーネル（487行目）で、128ビット演算が不完全：

```python
# 現在のバグのあるコード（487-488行目）
val_lo = val_lo * 10000 + digit
# TODO: 完全な128ビット演算実装
```

上位64ビット（`val_hi`）が更新されないため、大きな値で桁あふれが発生。

## 症状
- 5350000 → 535（下位4桁が失われる）
- 8770000 → 877（同様）
- 690000 → 69（同様）

## 修正案

### 方法1: 最小限の修正（64ビットオーバーフロー検出）

```python
# 基数10000桁読み取り
for digit_idx in range(nd):
    if current_offset + 2 > raw_data.size:
        break
    
    digit = (raw_data[current_offset] << 8) | raw_data[current_offset + 1]
    current_offset += 2
    
    # 128ビット演算: val = val * 10000 + digit
    # まず10000倍を計算
    old_lo = val_lo
    val_lo = val_lo * 10000
    
    # オーバーフロー検出
    if val_lo < old_lo:
        # オーバーフローが発生した場合の繰り上がり計算
        val_hi = val_hi * 10000 + (old_lo >> 44)  # 概算値
    else:
        val_hi = val_hi * 10000
    
    # digitを加算
    old_lo = val_lo
    val_lo = val_lo + digit
    if val_lo < old_lo:
        val_hi += 1
```

### 方法2: 完全な128ビット演算実装

`data_decoder.py`から正しい実装をコピー：

```python
# デバイス関数として128ビット演算を追加
@cuda.jit(device=True, inline=True)
def mul128_u64(a_hi, a_lo, b):
    """128ビット × 64ビット乗算"""
    mask32 = 0xFFFFFFFF
    
    a0 = a_lo & mask32
    a1 = a_lo >> 32
    a2 = a_hi & mask32
    a3 = a_hi >> 32
    
    b0 = b & mask32
    b1 = b >> 32
    
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    p20 = a2 * b0
    p21 = a2 * b1
    p30 = a3 * b0
    
    c0 = p00 >> 32
    r0 = p00 & mask32
    
    temp1 = p01 + p10 + c0
    c1 = temp1 >> 32
    r1 = temp1 & mask32
    
    temp2 = p11 + p20 + c1
    c2 = temp2 >> 32
    r2 = temp2 & mask32
    
    temp3 = p21 + p30 + c2
    r3 = temp3 & mask32
    
    res_lo = (r1 << 32) | r0
    res_hi = (r3 << 32) | r2
    
    return res_hi, res_lo

@cuda.jit(device=True, inline=True)
def add128(a_hi, a_lo, b_hi, b_lo):
    """128ビット加算"""
    res_lo = a_lo + b_lo
    carry = 1 if res_lo < a_lo else 0
    res_hi = a_hi + b_hi + carry
    return res_hi, res_lo
```

そして`_extract_decimal_direct`内で使用：

```python
# 基数10000桁読み取り
for digit_idx in range(nd):
    if current_offset + 2 > raw_data.size:
        break
    
    digit = (raw_data[current_offset] << 8) | raw_data[current_offset + 1]
    current_offset += 2
    
    # 完全な128ビット演算
    val_hi, val_lo = mul128_u64(val_hi, val_lo, 10000)
    val_hi, val_lo = add128(val_hi, val_lo, 0, digit)
```

## 推奨修正

方法2（完全な128ビット演算）を推奨。理由：
1. 数学的に正確
2. `data_decoder.py`に既存の実装があり、動作実績あり
3. 将来的により大きな値にも対応可能

## テスト方法

修正後、以下の値で確認：
- c_custkey = 5350000 → 正しく5350000が取得されること
- c_custkey = 8770000 → 正しく8770000が取得されること
- c_custkey = 690000 → 正しく690000が取得されること