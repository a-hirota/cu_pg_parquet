# GPU直接デシリアライズのマルチGPU機能改善

本ドキュメントでは、PostgreSQLデータのGPU直接デシリアライズとcuDF格納パイプラインのマルチGPU機能改善について説明します。

## 問題点

現在の実装では、Rayフレームワークを使用してマルチGPU処理を試みていますが、実際には**GPU 0のみが使用される**という問題がありました。この問題の原因は次の点にありました：

- `GPUDecoder`クラスの初期化時に`cuda.select_device(0)`がハードコードされており、全てのRayタスクが同じGPU 0を使用していた

## 修正内容

### 1. GPUDecoderクラスの修正

`GPUDecoder.__init__`メソッドからハードコードされたGPU選択を削除し、Rayによる自動GPU割り当てを利用するように変更しました。

**修正前:**
```python
def __init__(self):
    # CUDAコンテキストを初期化
    try:
        # デバイス0を選択（複数GPUの場合は適切なデバイスを選択）
        cuda.select_device(0)
        print("CUDA device initialized for GPUDecoder")
    except Exception as e:
        print(f"CUDA初期化エラー: {e}")
```

**修正後:**
```python
def __init__(self):
    # CUDAコンテキストを初期化
    try:
        # 明示的なデバイス選択を削除
        # Rayが設定した環境変数により、割り当てられたGPUが自動的に選択される
        print("CUDA device initialized for GPUDecoder (using Ray-assigned GPU)")
    except Exception as e:
        print(f"CUDA初期化エラー: {e}")
```

### 2. Rayチャンク割り当てロジックの改善

`ray_distributed_parquet.py`のチャンク割り当てロジックを改善し、前半チャンクと後半チャンクを異なるGPUセットに割り当てるようにしました。これにより、前半チャンクと後半チャンクを異なるGPUで並列処理できるようになります。

**修正前:**
```python
# GPUごとのチャンク割り当て計画を作成
gpu_assignments = []
for i in range(num_chunks):
    gpu_idx = i % len(gpu_ids)  # ラウンドロビンでGPUを割り当て
    gpu_assignments.append(gpu_ids[gpu_idx])
```

**修正後:**
```python
# GPUごとのチャンク割り当て計画を作成（前半・後半チャンクを分離）
gpu_assignments = []
num_chunks_per_half = (num_chunks + 1) // 2  # 半分に分割（切り上げ）
num_gpus_per_half = max(1, len(gpu_ids) // 2)  # 半分のGPU数（最低1）

for i in range(num_chunks):
    if i < num_chunks_per_half:
        # 前半部分 - 前半のGPUに割り当て
        gpu_idx = i % num_gpus_per_half
    else:
        # 後半部分 - 後半のGPUに割り当て
        remaining_gpus = len(gpu_ids) - num_gpus_per_half
        if remaining_gpus <= 0:
            # GPUが1つしかない場合は同じGPUを使用
            gpu_idx = 0
        else:
            # 後半のGPUセットからラウンドロビンで割り当て
            gpu_idx = num_gpus_per_half + ((i - num_chunks_per_half) % remaining_gpus)
    
    # GPU IDの範囲チェック
    gpu_idx = min(gpu_idx, len(gpu_ids) - 1)
    gpu_assignments.append(gpu_ids[gpu_idx])
    
print(f"チャンク割り当て計画: {gpu_assignments}")
```

## 動作と仕組み

改修後の動作の仕組みは次の通りです：

1. **Ray環境でのGPU割り当て**
   - Rayは各タスクに1つのGPUリソースを割り当て、環境変数`CUDA_VISIBLE_DEVICES`を自動設定します
   - この設定により、各プロセスからは割り当てられた特定のGPUのみが「GPU 0」として見えるようになります

2. **GPUDecoderの動作**
   - 明示的なGPU選択を行わないため、GPUDecoderはRayが設定した環境に従って自動的に割り当てられたGPUを使用します
   - 各タスクは異なるGPUリソースを使って並行処理されます

3. **前半・後半チャンクの並列処理**
   - データを前半と後半に分け、異なるGPUセットで処理することでリソースを最大活用します
   - 特にデータ量が多い場合に効果的です

## 期待される効果

この修正により、次のような効果が期待されます：

1. **複数GPU利用率の向上**
   - すべてのGPUが均等に活用されるようになり、処理能力が最大化されます

2. **処理時間の短縮**
   - 複数のGPUでの並列処理によりスループットが向上し、処理時間が短縮されます
   - 特に大規模データセットでの効果が顕著です

3. **スケーラビリティの向上**
   - GPUの数に応じて自動的にスケールするため、将来GPUを追加した場合も容易に活用できます

## テスト方法

テスト用の`examples/test_multigpu_fix.sh`スクリプトを用意しました。このスクリプトでは以下の検証を行います：

1. **修正前後での処理時間比較**
   - 修正前のコード（GPU 0のみ使用）での処理時間
   - 修正後のコード（複数GPU使用）での処理時間

2. **GPU使用率の監視**
   - nvidia-smiを使用して各GPUの使用率をモニタリング
   - 複数GPUが均等に活用されているかを確認

3. **出力データの一貫性確認**
   - 修正前後での出力ファイル数、サイズ、行数の比較
   - サンプルデータの内容確認による文字ずれチェック

## 使用方法

テストスクリプトを以下のように実行します：

```bash
./examples/test_multigpu_fix.sh
```

通常のマルチGPU処理は以下のコマンドで実行できます：

```bash
./examples/run_ray_multi_gpu.sh -t <テーブル名> -r <行数> -g <GPU数> -o <出力ディレクトリ>
```

## 注意点

1. Rayが正しくインストールされていることを確認してください
2. 十分なGPUメモリがあることを確認してください
3. チャンクサイズが大きすぎるとGPUメモリ不足になる可能性があります
