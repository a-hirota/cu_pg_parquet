#!/bin/bash
# 256MB境界問題をデバッグするためのベンチマーク実行

# 環境設定
export RUST_LOG=info
export RUST_PARALLEL_CONNECTIONS=16
export GPUPGPARSER_TEST_MODE=1
export GPUPGPARSER_DEBUG=1

# 実行
echo "=== Customer テーブル（130GB）のベンチマーク開始 ==="
echo "時刻: $(date)"
echo ""

# Rustプログラムを実行
cd /home/ubuntu/gpupgparser/rust_bench_optimized
timeout 300 ./target/release/pg_fast_copy_single_chunk customer 2>&1 | tee /tmp/benchmark_debug.log

echo ""
echo "=== ベンチマーク完了 ==="
echo "時刻: $(date)"

# ログから重要な情報を抽出
echo ""
echo "=== 検出された行数 ==="
grep -E "検出された行数|Detected rows|検出行数|行を検出" /tmp/benchmark_debug.log | tail -10

echo ""
echo "=== 256MB境界付近の情報 ==="
grep -E "256MB|0x10000000|Thread 1398101|Thread 3404032" /tmp/benchmark_debug.log | tail -20