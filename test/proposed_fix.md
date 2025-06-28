# 偶数行破損問題の修正提案

## 問題の要約
- lineorderテーブルのすべての行ヘッダが奇数アドレスに存在（19, 251, 483...）
- 偶数位置から検索を開始するスレッドが行ヘッダを見逃している
- これにより偶数チャンクのParquetファイルが破損または空になる

## 修正案

### 1. 行ヘッダ検出の改善（推奨）
`read_uint16_simd16_lite`関数を修正して、偶数・奇数両方の位置を確実にチェックする：

```python
@cuda.jit(device=True, inline=True)
def read_uint16_simd16_lite(raw_data, pos, end_pos, ncols):
    """行ヘッダ検出（改善版）"""
    if pos + 1 > raw_data.size:
        return -2
    
    # 偶数位置から開始する場合、1バイト前（奇数位置）もチェック
    start_offset = -1 if pos % 2 == 0 and pos > 0 else 0
    max_offset = min(17, raw_data.size - pos + 1)  # 16→17に拡張
    
    for i in range(start_offset, max_offset):
        check_pos = pos + i
        if check_pos < 0 or check_pos + 1 > end_pos:
            continue
        
        num_fields = (raw_data[check_pos] << 8) | raw_data[check_pos + 1]
        if num_fields == ncols:
            return check_pos
    
    return -1
```

### 2. スレッド開始位置の調整
各スレッドの開始位置を行境界に合わせる：

```python
# parse_rows_and_fields_lite内で
# 担当範囲計算を調整
start_pos = header_size + tid * thread_stride

# 奇数位置に調整（lineorderの場合）
if start_pos % 2 == 0 and start_pos > header_size:
    start_pos -= 1
```

### 3. ワープ最適化アプローチ（高度な修正）
ワープ内でのメモリアクセスを最適化：

```python
@cuda.jit
def parse_rows_and_fields_lite_warp_optimized(...):
    """ワープ最適化版カーネル"""
    warp_id = cuda.threadIdx.x // 32
    lane_id = cuda.threadIdx.x % 32
    
    # ワープ内で偶数・奇数レーンを交互に割り当て
    # これにより、隣接するスレッドが異なるパリティの位置を処理
    adjusted_pos = start_pos + lane_id * 2
    if lane_id >= 16:
        adjusted_pos = start_pos + (lane_id - 16) * 2 + 1
```

## テスト方法

1. 修正前後で同じデータを処理
2. 各チャンクの行数を比較
3. 偶数チャンクが正常にデータを含むことを確認

## 期待される効果

- 偶数チャンクでも正常に行を検出
- 全チャンクで均等な行数分布
- Parquetファイルの破損が解消