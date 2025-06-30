#!/bin/bash
# 1チャンクで全データを取得して100%確認

echo "=== 1チャンクで100%処理テスト ==="
echo "ctid制限なしで全データを取得"
echo ""

# 環境変数の設定
export TOTAL_CHUNKS=1
export CHUNK_ID=0
export GPUPASER_PG_DSN="$GPUPASER_PG_DSN"
export TABLE_NAME=lineorder
export RUST_PARALLEL_CONNECTIONS=16

# Rust転送のみテスト
echo "Rust転送を実行中..."
time ./rust_bench_optimized/target/release/pg_fast_copy_single_chunk > /tmp/1chunk_test.log 2>&1

# 結果確認
echo ""
echo "=== 結果 ==="
grep -E "(サイズ:|行数:|ページ範囲:|COPY)" /tmp/1chunk_test.log

# バイト数から推定行数を計算
bytes=$(grep "チャンク0: サイズ:" /tmp/1chunk_test.log | grep -o '[0-9]\+ bytes' | awk '{print $1}')
if [ ! -z "$bytes" ]; then
    rows=$((bytes / 352))
    echo ""
    echo "転送バイト数: $bytes"
    echo "推定行数（352バイト/行）: $rows"
    
    # 実際のバイト/行
    actual_bytes_per_row=$((bytes / 246012324))
    echo "実際のバイト/行: $actual_bytes_per_row"
fi

echo ""
echo "期待される行数: 246,012,324"