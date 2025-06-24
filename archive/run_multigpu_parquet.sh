#!/bin/bash
# マルチGPUでPostgreSQLデータをParquetファイルに変換するスクリプト

# デフォルト値
TABLE="customer"
ROWS=10000
OUTPUT_DIR="./multigpu_output"
NUM_GPUS=0  # 0=自動検出
GPU_IDS=""
CHUNK_SIZE=0  # 0=自動計算
DB_NAME="postgres"
DB_USER="postgres"
DB_PASSWORD="postgres"
DB_HOST="localhost"

# 引数の解析
while [ $# -gt 0 ]; do
    case "$1" in
        -t|--table)
            TABLE="$2"
            shift 2
            ;;
        -r|--rows)
            ROWS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -i|--gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        -c|--chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -d|--db)
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
            echo "使用方法: $0 [オプション]"
            echo "オプション:"
            echo "  -t, --table TABLE       処理するテーブル名 (デフォルト: customer)"
            echo "  -r, --rows ROWS         処理する総行数 (デフォルト: 10000)"
            echo "  -o, --output DIR        出力ディレクトリ (デフォルト: ./multigpu_output)"
            echo "  -g, --gpus NUM          使用するGPU数 (デフォルト: 自動検出)"
            echo "  -i, --gpu_ids IDS       使用するGPU IDのカンマ区切りリスト (例: '0,2')"
            echo "  -c, --chunk_size SIZE   チャンクサイズ (デフォルト: 自動計算)"
            echo "  -d, --db NAME           データベース名 (デフォルト: postgres)"
            echo "  -u, --user USER         データベースユーザー (デフォルト: postgres)"
            echo "  -p, --password PASS     データベースパスワード (デフォルト: postgres)"
            echo "  -h, --host HOST         データベースホスト (デフォルト: localhost)"
            echo "  --help                  このヘルプを表示"
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            echo "ヘルプを表示するには: $0 --help"
            exit 1
            ;;
    esac
done

# 実行コマンドの表示
echo "実行コマンド:"
CMD="PYTHONPATH=\$PYTHONPATH:$(dirname $(dirname $(realpath $0))) python examples/multigpu/simple_multigpu_parquet.py --table $TABLE --rows $ROWS --output $OUTPUT_DIR --db_name $DB_NAME --db_user $DB_USER --db_password $DB_PASSWORD --db_host $DB_HOST"

# オプションパラメータの追加
if [ "$NUM_GPUS" -gt 0 ]; then
    CMD="$CMD --gpus $NUM_GPUS"
fi

if [ ! -z "$GPU_IDS" ]; then
    CMD="$CMD --gpu_ids \"$GPU_IDS\""
fi

if [ "$CHUNK_SIZE" -gt 0 ]; then
    CMD="$CMD --chunk_size $CHUNK_SIZE"
fi

echo $CMD

# 処理開始
echo -e "\n処理を開始します..."
eval $CMD

# 処理結果の確認
if [ $? -eq 0 ]; then
    echo -e "\n==============================================\n処理が正常に完了しました。\n出力ディレクトリ: $OUTPUT_DIR\n=============================================="
else
    echo -e "\n==============================================\n処理中にエラーが発生しました。ログを確認してください。\n=============================================="
fi
