#!/bin/bash
# このスクリプトは、OFFSET/LIMITを使用してpublic.lineorderテーブル全体を60000行毎にチャンク分割し、
# process_large_dataset.py を用いてSQLクエリによりParquet変換を行います。
#
# 前提:
#   - 環境変数等でPostgreSQL接続情報が設定済みであること。
#   - process_large_dataset.py および gpupaser モジュールが正しく機能すること。
#
# この実装では、ダブルバッファリングの考え方を取り入れ、各チャンクのフェッチをバックグラウンドで開始し、
# 現在のチャンク処理と並行して次チャンクの取得を試みます。フェッチ完了はポーリングによって確認し、
# GPU処理待ち時間を最小化します。

# 設定
CHUNK_SIZE=8000000
# TOTAL_ROWS取得時に正しいユーザーでクエリを実行するために -U postgres を指定
TOTAL_ROWS=$(psql -U postgres -qt -A -c "SELECT COUNT(*) FROM public.lineorder" | tr -d '[:space:]')
OUTPUT_DIR="lineorder_offset_output"
TEMP_DIR="/dev/shm/lineorder_tmp"
TABLE="public.lineorder"

# 出力ディレクトリの作成
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEMP_DIR}"

echo "テーブル全体の行数: ${TOTAL_ROWS}"
echo "チャンクサイズ: ${CHUNK_SIZE} 行"
echo "Parquet出力ディレクトリ: ${OUTPUT_DIR}"

# GPUメモリクリア用関数
function cleanup_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPUメモリをクリアします..."
        nvidia-smi --gpu-reset 2>/dev/null || echo "GPU reset not supported, continuing..."
    fi
}

# 次のチャンクをバックグラウンドでフェッチする関数
function fetch_next_chunk() {
    local offset=$1
    local output_file=$2

    echo "OFFSET ${offset} のデータをフェッチします（バックグラウンド）..."
    # 新規接続でバイナリ形式のデータを取得
    psql -U postgres -v ON_ERROR_STOP=1 -c "COPY (
        SELECT * FROM ${TABLE}
        OFFSET ${offset}
        LIMIT ${CHUNK_SIZE}
    ) TO STDOUT WITH (FORMAT BINARY)" > "${output_file}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR" > "${output_file}.status"
    else
        echo "SUCCESS" > "${output_file}.status"
    fi
}

# 次のチャンクのデータを直接SQLクエリで処理する関数
function process_chunk_sql() {
    local offset=$1
    local parquet_output=$2
    local SQL_QUERY="SELECT * FROM ${TABLE} OFFSET ${offset} LIMIT ${CHUNK_SIZE}"
    echo "チャンクデータをParquetに変換中 (SQL: ${SQL_QUERY})..."
    python process_large_dataset.py --sql "${SQL_QUERY}" \
                                       --output-format parquet \
                                       --parquet "${parquet_output}" \
                                       --chunk-size ${CHUNK_SIZE} \
                                       --no-debug-files
    return $?
}

# 初期チャンクをバックグラウンドでフェッチ (OFFSET 0)
echo "初期チャンクを取得中..."
CURRENT_CHUNK_FILE="${TEMP_DIR}/chunk_current.bin"
NEXT_CHUNK_FILE="${TEMP_DIR}/chunk_next.bin"

fetch_next_chunk 0 "${CURRENT_CHUNK_FILE}"
# 確認のため、すぐにファイルサイズを取得（フェッチ完了まで数秒かかる可能性あり）
FILESIZE=$(stat -c%s "${CURRENT_CHUNK_FILE}" 2>/dev/null || echo 0)
if [ ${FILESIZE} -eq 0 ]; then
    echo "初期チャンクのデータが取得できませんでした。終了します。"
    rm -f "${CURRENT_CHUNK_FILE}" "${CURRENT_CHUNK_FILE}.status"
    exit 1
fi
echo "初期チャンクのファイルサイズ: ${FILESIZE} bytes"

# チャンク処理のメインループ
CHUNK=1
PROCESSED_ROWS=0
NEXT_OFFSET=${CHUNK_SIZE}

# 次のチャンクをバックグラウンドでフェッチ開始
fetch_next_chunk ${NEXT_OFFSET} "${NEXT_CHUNK_FILE}" &
# 非同期フェッチの完了は後でポーリングで確認
FETCH_PID=$!

while true; do
    CURRENT_OFFSET=$(( (CHUNK-1)*CHUNK_SIZE ))
    if [ ${CURRENT_OFFSET} -ge ${TOTAL_ROWS} ]; then
        echo "すべてのチャンクが処理されました。"
        break
    fi
    echo "【チャンク ${CHUNK} の処理開始】 (OFFSET: ${CURRENT_OFFSET})"
    PARQUET_OUTPUT="${OUTPUT_DIR}/lineorder_part${CHUNK}.parquet"
    
    # 現在のチャンクデータはファイルからSQLモードで処理（Binary → Parquet）
    process_chunk_sql ${CURRENT_OFFSET} "${PARQUET_OUTPUT}"
    PROCESS_RESULT=$?
    if [ ${PROCESS_RESULT} -ne 0 ]; then
        echo "チャンク ${CHUNK} の変換処理でエラーが発生しました（コード: ${PROCESS_RESULT}）"
        break
    else
        echo "チャンク ${CHUNK} の処理が完了しました"
        PROCESSED_ROWS=$((PROCESSED_ROWS + CHUNK_SIZE))
        if [ ${PROCESSED_ROWS} -gt ${TOTAL_ROWS} ]; then
            PROCESSED_ROWS=${TOTAL_ROWS}
        fi
        PROGRESS=$((PROCESSED_ROWS * 100 / TOTAL_ROWS))
        echo "進行状況: ${PROCESSED_ROWS}/${TOTAL_ROWS} 行処理完了 (${PROGRESS}%)"
    fi
    
    cleanup_gpu_memory

    # 次のチャンクのフェッチ完了をポーリング（待機時間を短縮）
    echo "【次のチャンクの準備待ち】"
    while [ ! -f "${NEXT_CHUNK_FILE}.status" ]; do
        sleep 0.5
    done
    STATUS=$(cat "${NEXT_CHUNK_FILE}.status")
    if [ "${STATUS}" != "SUCCESS" ]; then
        echo "次のチャンク取得時にエラーが発生しました。処理を終了します。"
        break
    fi

    # 現在のチャンクファイルを次のチャンクに差し替え
    mv "${NEXT_CHUNK_FILE}" "${CURRENT_CHUNK_FILE}"
    rm -f "${NEXT_CHUNK_FILE}.status"
    
    NEXT_OFFSET=$((NEXT_OFFSET + CHUNK_SIZE))
    if [ ${NEXT_OFFSET} -lt ${TOTAL_ROWS} ]; then
        fetch_next_chunk ${NEXT_OFFSET} "${NEXT_CHUNK_FILE}" &
        FETCH_PID=$!
    else
        echo "すべてのチャンクがフェッチされました。次のオフセット (${NEXT_OFFSET}) がテーブルの行数 (${TOTAL_ROWS}) を超えています。"
    fi
    CHUNK=$((CHUNK + 1))
done

rm -f "${CURRENT_CHUNK_FILE}" "${NEXT_CHUNK_FILE}" "${CURRENT_CHUNK_FILE}.status" "${NEXT_CHUNK_FILE}.status"

echo "全チャンクの処理が完了しました。"
echo "処理レコード数: ${PROCESSED_ROWS}/${TOTAL_ROWS} 行"
echo "Parquetファイル出力ディレクトリ: ${OUTPUT_DIR}"
