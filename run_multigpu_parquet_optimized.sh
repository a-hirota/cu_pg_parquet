#!/bin/bash
# 複数GPUを使った並列処理の実行スクリプト
# GPUカーネルのブロック数とスレッド数を設定（必要に応じて調整）
export GPUPASER_BLOCK_SIZE=256
export GPUPASER_THREAD_COUNT=1024
# 各GPUは別々のSQLクエリやパーティションテーブルを担当

# 出力ディレクトリを作成
mkdir -p lineorder_multigpu_output

# 開始時間の記録
START_TIME=$(date +%s.%N)

# GPU 0で前半のデータを処理（例：lineorder_part1テーブルまたはWHERE句で範囲指定）
CUDA_VISIBLE_DEVICES=0 python process_large_dataset.py \
  --gpuid 0 \
  --sql "SELECT * FROM lineorder limit 600000" \
  --output-format parquet \
  --parquet lineorder_multigpu_output/lineorder_part1.parquet \
  --no-debug-files \
  --quiet &
  
PID1=$!
echo "GPU 0で処理を開始 (PID: $PID1)"

# # GPU 1で後半のデータを処理
# CUDA_VISIBLE_DEVICES=1 python process_large_dataset.py \
#   --gpuid 1 \
#   --sql "SELECT * FROM lineorder  limit 600000 offset 60000" \
#   --output-format parquet \
#   --parquet lineorder_multigpu_output/lineorder_part2.parquet \
#   --no-debug-files \
#   --quiet &
  
# PID2=$!
# echo "GPU 1で処理を開始 (PID: $PID2)"

# すべての処理の完了を待機
wait $PID1
echo "GPU 0の処理が完了"
# wait $PID2
# echo "GPU 1の処理が完了"

# 終了時間の記録と経過時間の計算
END_TIME=$(date +%s.%N)
ELAPSED_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# 結果の確認
echo "==== 処理結果 ===="
echo "処理時間: ${ELAPSED_TIME}秒"
ls -lh lineorder_multigpu_output/
echo "処理完了"
