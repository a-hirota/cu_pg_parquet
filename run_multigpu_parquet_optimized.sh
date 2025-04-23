#!/bin/bash
# このスクリプトは、複合キー (lo_orderkey, lo_linenumber) を用いたキーセットページネーションにより、
# lineorder テーブル全体を60000行毎にチャンク分割し、各チャンクをParquet変換します。
#
# 前提: 環境変数等でPostgreSQL接続情報が設定済みであること。
#       process_large_dataset.py が適切に実行できること。

# 初期境界の設定（テーブル内の最小値に合わせて適宜変更）
CURRENT_ORDERKEY=0
CURRENT_LINENUMBER=0

CHUNK_SIZE=65000
CHUNK=1
OUTPUT_DIR="lineorder_keyset_output"

# 出力ディレクトリの作成
mkdir -p ${OUTPUT_DIR}

# 並列実行のための最大プロセス数（環境に応じて調整）
MAX_PARALLEL=1
active_processes=0

# GPUメモリクリア用の関数
function cleanup_gpu_memory() {
    # NVIDIA-SMIがある場合のみ実行
    if command -v nvidia-smi &> /dev/null; then
        echo "GPUメモリをクリアします..."
        nvidia-smi --gpu-reset 2>/dev/null || echo "GPU reset not supported, continuing..."
    fi
}

while true; do
    echo "チャンク ${CHUNK} 処理開始: 現在の境界 = (${CURRENT_ORDERKEY}, ${CURRENT_LINENUMBER})"
    
    # キーセットページネーションを使用したSQLクエリ
    SQL_QUERY="SELECT * FROM lineorder WHERE lo_orderkey > ${CURRENT_ORDERKEY} OR (lo_orderkey = ${CURRENT_ORDERKEY} AND lo_linenumber > ${CURRENT_LINENUMBER}) ORDER BY lo_orderkey, lo_linenumber LIMIT ${CHUNK_SIZE}"
    echo "DEBUG SQL_QUERY: ${SQL_QUERY}"
    
    # process_large_dataset.pyを使ってSQLクエリを処理、結果をParquetに保存
    PARQUET_OUTPUT="${OUTPUT_DIR}/lineorder_part${CHUNK}.parquet"
    
    # プロセス実行とバックグラウンド処理（オプション）
    if [ ${MAX_PARALLEL} -gt 1 ]; then
        # 並列処理の場合
        python process_large_dataset.py --sql "${SQL_QUERY}" \
                                        --output-format parquet \
                                        --parquet "${PARQUET_OUTPUT}" \
                                        --chunk-size ${CHUNK_SIZE} \
                                        --no-debug-files &
        
        # バックグラウンドプロセスのPIDを記録
        pids[${active_processes}]=$!
        active_processes=$((active_processes + 1))
        
        # 最大並列数に達したら、いずれかのプロセスの完了を待つ
        if [ ${active_processes} -ge ${MAX_PARALLEL} ]; then
            wait -n
            active_processes=$((active_processes - 1))
        fi
    else
        # 逐次処理の場合
        python process_large_dataset.py --sql "${SQL_QUERY}" \
                                       --output-format parquet \
                                       --parquet "${PARQUET_OUTPUT}" \
                                       --chunk-size ${CHUNK_SIZE} \
                                       --no-debug-files
    
        # 適宜GPUメモリのクリア
        cleanup_gpu_memory
    fi
    
    # キーセットページネーションのため、今回抽出したチャンクの最終行のキーを取得する
    LAST_BOUNDARY=$(psql -U postgres -t -A -c "SELECT lo_orderkey, lo_linenumber FROM (
      ${SQL_QUERY}
    ) AS sub ORDER BY lo_orderkey DESC, lo_linenumber DESC LIMIT 1;")
    
    # 結果が空ならば終了
    if [ -z "${LAST_BOUNDARY}" ]; then
        echo "これ以上のデータはありません。処理を終了します。"
        break
    fi
    
    # LAST_BOUNDARYは "値|値" の形式で返るので、分解する
    NEW_ORDERKEY=$(echo ${LAST_BOUNDARY} | cut -d"|" -f1)
    NEW_LINENUMBER=$(echo ${LAST_BOUNDARY} | cut -d"|" -f2)
    
    # 最新の境界値が取得できなければ終了
    if [ -z "${NEW_ORDERKEY}" ] || [ -z "${NEW_LINENUMBER}" ]; then
        echo "境界値の取得に失敗しました。終了します。"
        break
    fi
    
    echo "新しい境界値: (${NEW_ORDERKEY}, ${NEW_LINENUMBER})"
    
    # 境界値に変化がなければ処理を終了（無限ループ防止）
    if [ "${NEW_ORDERKEY}" = "${CURRENT_ORDERKEY}" ] && [ "${NEW_LINENUMBER}" = "${CURRENT_LINENUMBER}" ]; then
        echo "境界値に変化がありません。処理を終了します。"
        break
    fi
    
    # 境界値を更新
    CURRENT_ORDERKEY=${NEW_ORDERKEY}
    CURRENT_LINENUMBER=${NEW_LINENUMBER}
    
    CHUNK=$((CHUNK + 1))
done

# 残りの並列プロセスがあれば完了を待つ
if [ ${MAX_PARALLEL} -gt 1 ]; then
    echo "残りのプロセスの完了を待っています..."
    wait
fi

echo "全チャンクの処理が完了しました。"
echo "出力ディレクトリ: ${OUTPUT_DIR}"
