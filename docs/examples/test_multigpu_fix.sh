#!/bin/bash
# マルチGPU修正のテストスクリプト
# 修正前と修正後の処理時間とGPU使用率を比較

# テスト設定
TABLE="customer"  # テスト対象テーブル
TOTAL_ROWS=100000  # 処理行数（必要に応じて調整）
NUM_GPUS=2  # 使用GPU数
BEFORE_DIR="./test_before_fix"  # 修正前の出力ディレクトリ
AFTER_DIR="./test_after_fix"    # 修正後の出力ディレクトリ

# 必要なディレクトリを作成
mkdir -p "$BEFORE_DIR" "$AFTER_DIR"

# conda環境の確認
if ! command -v conda &> /dev/null; then
    echo "conda がインストールされていません。"
    exit 1
fi

# 実行前にconda環境をアクティブ化
echo "numba-cuda 環境をアクティブ化..."
eval "$(conda shell.bash hook)"
conda activate numba-cuda

# GPUモニタリング関数
start_gpu_monitoring() {
    # バックグラウンドでnvidia-smiを実行し、GPUの使用率をログに記録
    echo "GPUモニタリングを開始..."
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 1 > "$1/gpu_usage.log" &
    echo $! > nvidia_smi.pid
    sleep 2  # モニタリング開始を待つ
}

stop_gpu_monitoring() {
    # GPU使用率モニタリングを停止
    if [ -f nvidia_smi.pid ]; then
        echo "GPUモニタリングを停止..."
        kill $(cat nvidia_smi.pid)
        rm nvidia_smi.pid
    fi
}

# テスト実行前の確認
echo "======================================================"
echo "マルチGPU修正のテスト - $TABLE テーブル ($TOTAL_ROWS 行)"
echo "======================================================"
echo "利用可能なGPU:"
nvidia-smi --list-gpus
echo ""
echo "テスト1: 修正前のコード"
echo "出力ディレクトリ: $BEFORE_DIR"
echo "テスト2: 修正後のコード"
echo "出力ディレクトリ: $AFTER_DIR"
echo ""
read -p "テストを開始しますか？ (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "テストを中止します。"
    exit 0
fi

# テスト用にgpu_decoder.pyのバックアップを作成
cp gpupaser/gpu_decoder.py gpupaser/gpu_decoder.py.bak
# 修正前のバージョンに戻す (cuda.select_device(0)を追加)
sed -i 's/# 明示的なデバイス選択を削除/# デバイス0を選択（複数GPUの場合は適切なデバイスを選択）\n            cuda.select_device(0)/g' gpupaser/gpu_decoder.py
sed -i 's/using Ray-assigned GPU/initialized for GPUDecoder/g' gpupaser/gpu_decoder.py

echo "======================================================"
echo "テスト1: 修正前のコード実行"
echo "======================================================"

# GPU使用率のモニタリングを開始
start_gpu_monitoring "$BEFORE_DIR"

# 検証オプションを無効化（処理時間のみを測定）
export VERIFY_SAMPLE=0

# 時間計測開始
TIMEFORMAT="実行時間: %3Rs"
time {
    # 修正前のコード実行
    ./examples/run_ray_multi_gpu.sh -t "$TABLE" -r "$TOTAL_ROWS" -g "$NUM_GPUS" -o "$BEFORE_DIR"
}

# モニタリング停止
stop_gpu_monitoring

# 修正後のバージョンに復元
mv gpupaser/gpu_decoder.py.bak gpupaser/gpu_decoder.py

echo "======================================================"
echo "テスト2: 修正後のコード実行"
echo "======================================================"

# GPU使用率のモニタリングを開始
start_gpu_monitoring "$AFTER_DIR"

# 時間計測開始
TIMEFORMAT="実行時間: %3Rs"
time {
    # 修正後のコード実行（検証オプションは依然として無効）
    ./examples/run_ray_multi_gpu.sh -t "$TABLE" -r "$TOTAL_ROWS" -g "$NUM_GPUS" -o "$AFTER_DIR"
}

# モニタリング停止
stop_gpu_monitoring

# テスト結果の検証
echo "======================================================"
echo "テスト結果の検証"
echo "======================================================"

# 出力ファイルの一覧
echo "修正前の出力ファイル:"
find "$BEFORE_DIR" -name "*.parquet" | sort

echo "修正後の出力ファイル:"
find "$AFTER_DIR" -name "*.parquet" | sort

# ファイル数の確認
before_count=$(find "$BEFORE_DIR" -name "*.parquet" | wc -l)
after_count=$(find "$AFTER_DIR" -name "*.parquet" | wc -l)

echo "出力ファイル数: 修正前 $before_count 個, 修正後 $after_count 個"

# 各ファイルのサイズ確認
echo "ファイルサイズ比較:"
echo "修正前:"
find "$BEFORE_DIR" -name "*.parquet" -exec du -h {} \;

echo "修正後:"
find "$AFTER_DIR" -name "*.parquet" -exec du -h {} \;

# データの内容を確認（cuDFで最初の1つのファイルを読み込み）
echo "データ内容の確認（サンプル）:"
# PyScriptでParquetファイル検証
python -c "
import os
import glob
import cudf
import pandas as pd

# 修正前ファイル
before_files = sorted(glob.glob('$BEFORE_DIR/*.parquet'))
if before_files:
    before_df = cudf.read_parquet(before_files[0])
    before_rows = len(before_df)
    before_sample = before_df.head(3).to_pandas()
    print(f'修正前ファイル: {os.path.basename(before_files[0])}')
    print(f'行数: {before_rows}')
    print('サンプル行:')
    print(before_sample)
    print()

# 修正後ファイル
after_files = sorted(glob.glob('$AFTER_DIR/*.parquet'))
if after_files:
    after_df = cudf.read_parquet(after_files[0])
    after_rows = len(after_df)
    after_sample = after_df.head(3).to_pandas()
    print(f'修正後ファイル: {os.path.basename(after_files[0])}')
    print(f'行数: {after_rows}')
    print('サンプル行:')
    print(after_sample)

# 修正前と修正後の合計行数を計算
before_total = sum(len(cudf.read_parquet(f)) for f in before_files)
after_total = sum(len(cudf.read_parquet(f)) for f in after_files)
print(f'\\n合計行数: 修正前 {before_total}行, 修正後 {after_total}行')
print(f'行数の一致: {"はい" if before_total == after_total else "いいえ"}')
"

# GPU使用率の解析
echo "GPU使用率の解析:"
echo "修正前:"
grep "GPU-Util" "$BEFORE_DIR/gpu_usage.log" | head -10 

echo "修正後:"
grep "GPU-Util" "$AFTER_DIR/gpu_usage.log" | head -10

echo "======================================================"
echo "テスト完了"
echo "======================================================"
