#!/bin/bash
# PostgreSQLデータをマルチGPUで処理するシェルスクリプトラッパー

# デフォルト値
TABLE="customer"
TOTAL_ROWS=6000000
NUM_GPUS=0  # 0の場合は利用可能なすべてのGPUを使用
GPU_IDS=""  # カンマ区切りのGPU ID (例: "0,1,2")
OUTPUT_DIR="./parquet_output"
CHUNK_SIZE=""  # 空の場合は自動計算
DB_NAME="postgres"
DB_USER="postgres"
DB_PASSWORD="postgres"
DB_HOST="localhost"

# コマンドライン引数の解析
show_help() {
    echo "使用方法: $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  -t, --table TABLE       処理するテーブル名 (デフォルト: $TABLE)"
    echo "  -r, --rows ROWS         処理する合計行数 (デフォルト: $TOTAL_ROWS)"
    echo "  -g, --gpus NUM_GPUS     使用するGPU数 (デフォルト: すべて)"
    echo "  -i, --gpu-ids IDS       使用するGPU IDのカンマ区切りリスト (例: '0,1,2')"
    echo "  -o, --output DIR        出力ディレクトリ (デフォルト: $OUTPUT_DIR)"
    echo "  -c, --chunk SIZE        チャンクサイズ (デフォルト: 自動計算)"
    echo "  -d, --dbname NAME       データベース名 (デフォルト: $DB_NAME)"
    echo "  -u, --user USER         データベースユーザー (デフォルト: $DB_USER)"
    echo "  -p, --password PASS     データベースパスワード (デフォルト: $DB_PASSWORD)"
    echo "  -h, --host HOST         データベースホスト (デフォルト: $DB_HOST)"
    echo "  --help                  このヘルプを表示"
    exit 1
}

# 引数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--table)
            TABLE="$2"
            shift 2
            ;;
        -r|--rows)
            TOTAL_ROWS="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -i|--gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--chunk)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -d|--dbname)
            DB_NAME="$2"
            shift 2
            ;;
        -u|--user)
            DB_USER="$2"
            shift 2
            ;;
        -p|--password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        -h|--host)
            DB_HOST="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "未知のオプション: $1"
            show_help
            ;;
    esac
done

# conda環境の確認
if ! command -v conda &> /dev/null; then
    echo "conda がインストールされていません。"
    exit 1
fi

# 実行前にconda環境をアクティブ化
echo "numba-cuda 環境をアクティブ化..."
eval "$(conda shell.bash hook)"
conda activate numba-cuda

# Rayが利用可能か確認
if ! python -c "import ray" &> /dev/null; then
    echo "Rayパッケージがインストールされていません。以下のコマンドでインストールしてください："
    echo "conda activate numba-cuda && pip install ray[default]"
    exit 1
fi

# PYTHONPATHの設定
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# コマンド構築（環境変数を渡す）
CMD="PYTHONPATH=\$PYTHONPATH:$PROJECT_ROOT VERIFY_SAMPLE=$VERIFY_SAMPLE python examples/ray_distributed_parquet.py --table $TABLE --total_rows $TOTAL_ROWS"

if [ $NUM_GPUS -gt 0 ]; then
    CMD="$CMD --num_gpus $NUM_GPUS"
fi

if [ ! -z "$GPU_IDS" ]; then
    CMD="$CMD --gpu_ids \"$GPU_IDS\""
fi

CMD="$CMD --output_dir $OUTPUT_DIR"

if [ ! -z "$CHUNK_SIZE" ]; then
    CMD="$CMD --chunk_size $CHUNK_SIZE"
fi

CMD="$CMD --db_name $DB_NAME --db_user $DB_USER --db_password $DB_PASSWORD --db_host $DB_HOST"

# 実行前に確認
echo "実行コマンド:"
echo "$CMD"
echo ""
echo "処理を開始します..."
eval $CMD

# 終了ステータスの確認
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "処理が正常に完了しました。出力先: $OUTPUT_DIR"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "処理中にエラーが発生しました。ログを確認してください。"
    echo "=============================================="
fi
