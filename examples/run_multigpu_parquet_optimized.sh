#!/bin/bash
# 複数GPUを使った並列処理の実行スクリプト
# 各GPUは別々のSQLクエリやパーティションテーブルを担当

# 出力ディレクトリを作成
mkdir -p lineorder_multigpu_output

# GPU 0で前半のデータを処理
CUDA_VISIBLE_DEVICES=0 python process_large_dataset.py \
  --sql "SELECT * FROM lineorder WHERE lo_orderkey < 1000000" \
  --output-format parquet \
  --parquet lineorder_multigpu_output/lineorder_part1.parquet &
  
PID1=$!
echo "GPU 0で処理を開始しました (PID: $PID1)"

# GPU 1で後半のデータを処理
CUDA_VISIBLE_DEVICES=1 python process_large_dataset.py \
  --sql "SELECT * FROM lineorder WHERE lo_orderkey >= 1000000" \
  --output-format parquet \
  --parquet lineorder_multigpu_output/lineorder_part2.parquet &
  
PID2=$!
echo "GPU 1で処理を開始しました (PID: $PID2)"

# すべての処理の完了を待機
wait $PID1
echo "GPU 0の処理が完了しました"
wait $PID2
echo "GPU 1の処理が完了しました"

# 結果の確認
echo "==== 処理結果 ===="
ls -lh lineorder_multigpu_output/
echo "処理完了"
