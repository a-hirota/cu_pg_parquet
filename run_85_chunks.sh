#!/bin/bash
# 85チャンクで100%処理を実行

echo "=== 85チャンクで100%処理 ==="
echo "開始時刻: $(date)"
echo ""

# 環境変数の設定
export TOTAL_CHUNKS=85
export GPUPGPARSER_TEST_MODE=0
export PATH=/home/ubuntu/miniforge/bin:$PATH

# ベンチマーク実行
cd /home/ubuntu/gpupgparser
python docs/benchmark/benchmark_rust_gpu_direct.py 2>&1 | tee 85chunks_result.log

# 結果の確認
echo ""
echo "=== 処理結果 ==="
grep -E "(総処理行数|処理率|Missing)" 85chunks_result.log || echo "結果が見つかりません"

echo ""
echo "終了時刻: $(date)"
echo ""
echo "期待される結果: 246,012,324行（100%）"
