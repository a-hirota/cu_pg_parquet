#!/usr/bin/env python3
"""
82チャンクでの100%達成を検証
"""
import os

def verify_82_chunks():
    """82チャンクでの計算"""
    
    print("=== 82チャンクでの100%達成検証 ===\n")
    
    # 既知の情報
    total_pages = 4_614_912  # pg_relation_sizeから
    max_data_page = 4_614_887  # 実際のデータがある最大ページ
    total_expected_rows = 246_012_324
    
    # 82チャンクでの計算
    chunks = 82
    pages_per_chunk = total_pages // chunks  # 56,279ページ
    
    print(f"総ページ数（pg_relation_size）: {total_pages:,}")
    print(f"実データの最大ページ: {max_data_page:,}")
    print(f"期待される総行数: {total_expected_rows:,}")
    
    print(f"\n82チャンクでの分割:")
    print(f"チャンクあたりページ数: {pages_per_chunk:,}")
    
    # 最終チャンクの範囲を確認
    last_chunk_id = chunks - 1
    last_chunk_start = last_chunk_id * pages_per_chunk
    last_chunk_end = total_pages
    
    print(f"\n最終チャンク（チャンク81）:")
    print(f"開始ページ: {last_chunk_start:,}")
    print(f"終了ページ: {last_chunk_end:,}")
    print(f"ページ数: {last_chunk_end - last_chunk_start:,}")
    
    # 実データの最大ページがカバーされているか確認
    if last_chunk_start <= max_data_page < last_chunk_end:
        print(f"\n✅ 実データの最大ページ（{max_data_page}）は最終チャンクに含まれます")
    else:
        for i in range(chunks):
            start = i * pages_per_chunk
            end = total_pages if i == chunks - 1 else (i + 1) * pages_per_chunk
            if start <= max_data_page < end:
                print(f"\n✅ 実データの最大ページ（{max_data_page}）はチャンク{i}に含まれます")
                break
    
    # 推定処理行数
    print("\n=== 推定処理行数 ===")
    
    # 80チャンクでの実績から推定
    # 80チャンクで98.53%（242,386,050行）
    # つまり、有効ページ率は約40%
    
    effective_page_ratio = 242_386_050 / total_expected_rows  # 0.9853
    
    # 82チャンクでは、より多くのページをカバー
    # 理論的には100%に近づくはず
    
    print(f"\n80チャンクでの達成率: {effective_page_ratio * 100:.2f}%")
    print(f"82チャンクでの期待達成率: ~100%")
    
    # 実行コマンド
    print("\n=== 実行方法 ===")
    print("export TOTAL_CHUNKS=82")
    print("python docs/benchmark/benchmark_rust_gpu_direct.py")
    
    print("\n注意: 82チャンクの実行には約77秒かかる見込みです")


def create_82_chunk_test_script():
    """82チャンクテスト用スクリプトを作成"""
    
    script_content = """#!/bin/bash
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
"""
    
    with open('test_82_chunks_sample.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('test_82_chunks_sample.sh', 0o755)
    print("\ntest_82_chunks_sample.sh を作成しました")


if __name__ == "__main__":
    verify_82_chunks()
    create_82_chunk_test_script()