#!/bin/bash
# 32チャンクで再実行して500行差分を確認

echo "=== 32チャンクでの実行（元の設定） ==="
echo "期待: 246,012,324行（-500行程度）"
echo ""

# 環境変数の設定
export TOTAL_CHUNKS=32
export GPUPGPARSER_TEST_MODE=0
export PATH=/home/ubuntu/miniforge/bin:$PATH

# ディレクトリ移動
cd /home/ubuntu/gpupgparser

# 単一チャンクのテスト実行（チャンク0のみ）
echo "チャンク0のテスト実行..."
export CHUNK_ID=0
./rust_bench_optimized/target/release/pg_fast_copy_single_chunk > /tmp/chunk0_32.log 2>&1

# 結果確認
echo ""
echo "チャンク0の結果:"
grep -E "(サイズ:|行数:|ページ範囲:)" /tmp/chunk0_32.log || echo "情報が見つかりません"

# バイト数から推定行数を計算
bytes=$(grep "チャンク0: サイズ:" /tmp/chunk0_32.log | grep -o '[0-9]\+ bytes' | awk '{print $1}')
if [ ! -z "$bytes" ]; then
    rows=$((bytes / 352))
    echo "推定行数: $rows"
fi

echo ""
echo "フル実行するには:"
echo "export TOTAL_CHUNKS=32"
echo "python docs/benchmark/benchmark_rust_gpu_direct.py"
echo ""
echo "注意: 32チャンクの実行には約30秒かかります"