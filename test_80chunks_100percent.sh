#!/bin/bash
# 80チャンクで100%処理を確認するスクリプト

echo "=== 80チャンクで100%処理テスト ==="
echo "最終チャンク拡張実装済み"
echo ""

# 環境変数の設定
export TOTAL_CHUNKS=80
export GPUPGPARSER_TEST_MODE=0  # 本番モード

# ベンチマークを実行
cd /home/ubuntu/gpupgparser
time python docs/benchmark/benchmark_rust_gpu_direct.py

# 結果を確認
echo ""
echo "=== 処理結果の確認 ==="
echo "期待される行数: 246,012,324"
echo "処理された行数を上記の出力から確認してください"