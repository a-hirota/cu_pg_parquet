#!/bin/bash

# PostgreSQL kvikio直接ヒープ読み込みベンチマーク実行スクリプト
# 
# このスクリプトは、postgresユーザ権限でkvikioベンチマークを実行します。
# PostgreSQLヒープファイルへの読み取りアクセスには管理者権限が必要です。

# 設定
GPUPGPARSER_DIR="/home/ubuntu/gpupgparser"
DATABASE="${1:-postgres}"
TABLE="${2:-lineorder}"
POSTGRES_DATA_DIR="${3:-/var/lib/postgresql/data}"

echo "=== PostgreSQL kvikio直接ヒープ読み込みベンチマーク ==="
echo "データベース: $DATABASE"
echo "テーブル: $TABLE"
echo "データディレクトリ: $POSTGRES_DATA_DIR"
echo ""

# 現在のユーザー確認
CURRENT_USER=$(whoami)
echo "現在のユーザー: $CURRENT_USER"

if [ "$CURRENT_USER" != "postgres" ]; then
    echo ""
    echo "⚠️  PostgreSQLヒープファイルにアクセスするには、postgresユーザ権限が必要です。"
    echo ""
    echo "以下のコマンドで実行してください:"
    echo "sudo su - postgres -c \""
    echo "  export PYTHONPATH=\$PYTHONPATH:$GPUPGPARSER_DIR && \\"
    echo "  export GPUPASER_PG_DSN='dbname=$DATABASE user=postgres host=localhost port=5432' && \\"
    echo "  export POSTGRES_DATA_DIR='$POSTGRES_DATA_DIR' && \\"
    echo "  python $GPUPGPARSER_DIR/benchmark/benchmark_kvikio_heap.py --table $TABLE --database $DATABASE\""
    echo ""
    echo "または:"
    echo "sudo su - postgres"
    echo "bash $GPUPGPARSER_DIR/benchmark/run_kvikio_heap_benchmark.sh $DATABASE $TABLE $POSTGRES_DATA_DIR"
    exit 1
fi

# postgresユーザとして実行
echo "✅ postgresユーザとして実行中..."

# 環境変数設定
export PYTHONPATH=$PYTHONPATH:$GPUPGPARSER_DIR
export GPUPASER_PG_DSN="dbname=$DATABASE user=postgres host=localhost port=5432"
export POSTGRES_DATA_DIR="$POSTGRES_DATA_DIR"

# CUDA確認
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi コマンドが見つかりません。CUDAが正しくインストールされていません。"
    exit 1
fi

# Python確認
if ! command -v python &> /dev/null; then
    echo "❌ python コマンドが見つかりません。"
    exit 1
fi

# ベンチマーク実行
echo ""
echo "kvikioベンチマーク開始..."
python $GPUPGPARSER_DIR/benchmark/benchmark_kvikio_heap.py \
    --table $TABLE \
    --database $DATABASE

echo ""
echo "=== ベンチマーク完了 ==="