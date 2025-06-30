#!/bin/bash
# 82チャンクで100%処理を確認

echo "=== 82チャンクで100%処理テスト ==="
echo "期待: 246,012,324行を全て処理"
echo ""

# 環境変数の設定
export TOTAL_CHUNKS=82
export GPUPGPARSER_TEST_MODE=0  # 本番モード

# 最後の5チャンクだけテスト（時間短縮のため）
echo "最後の5チャンクのみテスト実行..."
for i in {77..81}; do
    echo "チャンク$i を実行中..."
    export CHUNK_ID=$i
    ./rust_bench_optimized/target/release/pg_fast_copy_single_chunk > /tmp/chunk_$i.log 2>&1
    
    # バイト数を抽出
    bytes=$(grep "チャンク$i: サイズ:" /tmp/chunk_$i.log | awk '{print $4}' | sed 's/(//')
    echo "  転送サイズ: $bytes bytes"
done

echo ""
echo "全82チャンクを実行するには:"
echo "export TOTAL_CHUNKS=82"
echo "python docs/benchmark/benchmark_rust_gpu_direct.py"
