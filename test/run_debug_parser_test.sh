#!/bin/bash
# Debug Parser Test Script
# whileループ終了原因分析とスレッド境界問題の診断

echo "=== Ultra Fast Parser Debug Test ==="
echo "whileループ終了原因トラッキング + 境界オーバーラップ強化版"
echo ""

# 環境変数設定
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/gpupgparser
export GPUPASER_PG_DSN='dbname=postgres user=postgres host=localhost port=5432'

# PostgreSQL接続確認
echo "PostgreSQL接続確認中..."
if ! psql $GPUPASER_PG_DSN -c "SELECT 1" > /dev/null 2>&1; then
    echo "❌ PostgreSQL接続失敗"
    exit 1
fi
echo "✅ PostgreSQL接続OK"
echo ""

# 段階的テスト実行
echo "段階1: 小規模テスト（10万行）で動作確認"
echo "----------------------------------------"
python benchmark/benchmark_gpu_parse_debug.py --rows 100000 --debug

echo ""
echo "段階2: 中規模テスト（50万行）で見逃し位置分析"
echo "--------------------------------------------"
python benchmark/benchmark_gpu_parse_debug.py --rows 500000 --debug

echo ""
echo "段階3: 大規模テスト（100万行）で完全分析"
echo "----------------------------------------"
python benchmark/benchmark_gpu_parse_debug.py --rows 1000000 --debug

echo ""
echo "=== Debug Test 完了 ==="