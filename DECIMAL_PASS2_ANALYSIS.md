# Decimal変換のPass2性能分析

## 概要

このドキュメントは、decimal変換のpass2が遅い原因を分析するために追加された機能と、分析方法について説明します。

## 追加された分析機能

### 1. 詳細タイミング測定

`src/gpu_decoder_v2.py`に以下のタイミング測定機能を追加しました：

- **CPU タイミング**: `time.time()`を使用した壁時計時間
- **GPU タイミング**: CUDAイベントを使用した純粋なGPU実行時間
- **同期オーバーヘッド**: CPU-GPU同期にかかる時間

### 2. メモリ帯域幅分析

- **処理データ量**: 入力データ + 出力データのサイズ
- **帯域幅計算**: データ量 ÷ 実行時間
- **GPUメモリ使用量**: 実行時のGPUメモリ状況

### 3. 行単位性能指標

- **行あたり処理時間**: 1行のdecimal変換にかかる時間
- **スループット**: 1秒間に処理できる行数

## 分析方法

### ベンチマーク実行

```bash
python benchmark_decimal_pass2.py
```

### 手動テスト

```bash
# 最適化版を有効にして実行
export USE_DECIMAL_OPTIMIZATION=1
python benchmark/benchmark_lineorder_5m.py

# 元のカーネルと比較
export USE_DECIMAL_OPTIMIZATION=0
python benchmark/benchmark_lineorder_5m.py
```

## 出力例と分析ポイント

### 正常な性能の目安

```
DECIMAL PASS2 TIMING: column_name GPU timing: 0.001234 seconds
DECIMAL PASS2 TIMING: column_name processed 1000000 rows, 0.001 ms per row (GPU)
DECIMAL PASS2 TIMING: column_name GPU bandwidth: 150.00 GB/s
```

### ボトルネック判定基準

| 指標 | 正常値 | 問題値 | 考えられる原因 |
|------|--------|--------|----------------|
| GPU帯域幅 | >100 GB/s | <50 GB/s | メモリ帯域制限 |
| 同期オーバーヘッド | <10% | >50% | CPU-GPU同期問題 |
| 行あたり時間 | <0.01ms | >0.1ms | 計算ボトルネック |
| GPU使用率 | >80% | <50% | 並列度不足 |

## 想定される原因と対策

### 1. GPUメモリ帯域制限

**症状**: GPU帯域幅が理論値の50%以下

**原因**:
- 非連続メモリアクセス
- キャッシュミス
- メモリ競合

**対策**:
- メモリアクセスパターンの最適化
- データレイアウトの改善
- キャッシュフレンドリーなアルゴリズム

### 2. カーネル実行効率

**症状**: 行あたり処理時間が長い

**原因**:
- 分岐処理の多用
- 128bit演算の非効率な実装
- スケール変換処理の複雑さ

**対策**:
- ループ展開
- 分岐予測の改善
- SIMD最適化

### 3. CPU-GPU同期オーバーヘッド

**症状**: 同期オーバーヘッドが実行時間の大部分

**原因**:
- 頻繁な`cuda.synchronize()`呼び出し
- 小さなカーネル起動
- GPU使用率の低さ

**対策**:
- カーネル融合
- 非同期実行
- バッチ処理

## 性能改善の実装

### 実装した改修

分析結果に基づき、NUMERIC列を文字列として処理するオプションを追加しました：

**改修ファイル:**
- `src/meta_fetch.py`: NUMERIC列の型マッピングを文字列に変更するオプション
- `benchmark_decimal_pass2.py`: 新しい環境変数を設定
- `benchmark_decimal_comparison.py`: DECIMAL vs STRING の性能比較ツール

### 使用方法

**高速化モード（NUMERIC→STRING）:**
```bash
export NUMERIC_AS_STRING=1
python benchmark_decimal_pass2.py
```

**従来モード（NUMERIC→DECIMAL128）:**
```bash
export NUMERIC_AS_STRING=0
python benchmark_decimal_pass2.py
```

**性能比較:**
```bash
python benchmark_decimal_comparison.py
```

### 期待される改善

- **処理時間**: 26秒 → 1秒以下（20-30倍高速化）
- **メモリ帯域幅**: 0.01 GB/s → 100+ GB/s
- **GPU利用効率**: 大幅改善

## 次のステップ

1. **性能比較実行**: `benchmark_decimal_comparison.py`で改善効果を測定
2. **本格導入検討**: 用途に応じてデフォルト設定を決定
3. **追加最適化**: 必要に応じてDECIMAL128カーネルの最適化も実施

## 関連ファイル

- `src/gpu_decoder_v2.py`: メインの分析コード
- `src/meta_fetch.py`: **改修済み** - NUMERIC列の型マッピング制御
- `src/cuda_kernels/arrow_gpu_pass2_decimal128.py`: Decimalカーネル実装
- `benchmark_decimal_pass2.py`: **改修済み** - 高速化モード対応
- `benchmark_decimal_comparison.py`: **新規** - 性能比較ツール
- `benchmark/benchmark_lineorder_5m.py`: 既存のベンチマーク