#!/usr/bin/env python3
"""
チャンク79（最終チャンク）の詳細分析
"""
import psycopg2
import os

def analyze_final_chunk():
    """最終チャンクの詳細分析"""
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("エラー: GPUPASER_PG_DSN環境変数が設定されていません")
        return
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    print("=== チャンク79（最終チャンク）の詳細分析 ===\n")
    
    # チャンク79の範囲
    total_pages = 4_614_888
    chunks = 80
    pages_per_chunk = total_pages // chunks
    chunk_79_start = 79 * pages_per_chunk
    chunk_79_end = total_pages
    
    print(f"チャンク79の範囲:")
    print(f"  開始ページ: {chunk_79_start:,}")
    print(f"  終了ページ: {chunk_79_end:,}")
    print(f"  ページ数: {chunk_79_end - chunk_79_start:,}")
    
    # 現在のRustコードのCOPY文をシミュレート
    print("\n現在のRustコードのCOPY文:")
    print(f"COPY (SELECT * FROM lineorder WHERE ctid >= '({chunk_79_start},0)'::tid AND ctid < '({chunk_79_end},0)'::tid)")
    
    # この条件での行数を確認
    cur.execute(f"""
        SELECT COUNT(*) 
        FROM lineorder 
        WHERE ctid >= '({chunk_79_start},0)'::tid 
          AND ctid < '({chunk_79_end},0)'::tid
    """)
    rows_with_current_condition = cur.fetchone()[0]
    print(f"\n現在の条件での行数: {rows_with_current_condition:,}")
    
    # 終了ページを含めた場合の行数
    cur.execute(f"""
        SELECT COUNT(*) 
        FROM lineorder 
        WHERE ctid >= '({chunk_79_start},0)'::tid
    """)
    rows_with_no_upper_limit = cur.fetchone()[0]
    print(f"上限なしの場合の行数: {rows_with_no_upper_limit:,}")
    
    # 差分
    missing_in_chunk_79 = rows_with_no_upper_limit - rows_with_current_condition
    print(f"\nチャンク79で欠損している行数: {missing_in_chunk_79:,}")
    
    # ページ4,614,888以降のデータを確認
    print(f"\nページ{chunk_79_end}以降のデータ:")
    for page in range(chunk_79_end, chunk_79_end + 5):
        cur.execute(f"""
            SELECT COUNT(*) 
            FROM lineorder 
            WHERE ctid >= '({page},0)'::tid 
              AND ctid < '({page + 1},0)'::tid
        """)
        rows = cur.fetchone()[0]
        if rows > 0:
            print(f"  ページ{page}: {rows}行")
    
    # 全体の欠損と比較
    print("\n=== 欠損行の分析まとめ ===")
    total_missing = 3_626_274
    print(f"全体の欠損行数: {total_missing:,}")
    print(f"チャンク79での欠損: {missing_in_chunk_79:,}")
    print(f"その他の欠損: {total_missing - missing_in_chunk_79:,}")
    
    # 解決策
    print("\n=== 100%達成のための解決策 ===")
    print("1. 最終チャンクの条件を変更:")
    print("   現在: ctid < '(4614888,0)'::tid")
    print("   修正: 条件なし（WHERE ctid >= '(4557194,0)'::tid のみ）")
    print("\n2. または82チャンクに増やす")
    print("   各チャンクが56,279ページをカバー")
    print("   より均等な分散で100%カバー")
    
    cur.close()
    conn.close()


def simulate_82_chunks():
    """82チャンクでのシミュレーション"""
    
    print("\n\n=== 82チャンクでのシミュレーション ===")
    
    total_pages = 4_614_888
    chunks = 82
    pages_per_chunk = total_pages // chunks  # 56,279
    
    # 各チャンクの範囲を表示（最後の5チャンクのみ）
    print("\n最後の5チャンクの範囲（82チャンクの場合）:")
    for chunk_id in range(77, 82):
        start_page = chunk_id * pages_per_chunk
        if chunk_id == chunks - 1:
            end_page = total_pages
        else:
            end_page = (chunk_id + 1) * pages_per_chunk
        
        print(f"  チャンク{chunk_id}: ページ{start_page:,} - {end_page:,}")
        
        if chunk_id == chunks - 1:
            # 最終チャンクのページ数
            final_chunk_pages = end_page - start_page
            print(f"    最終チャンクのページ数: {final_chunk_pages:,}")
            print(f"    通常チャンクとの差: {final_chunk_pages - pages_per_chunk:,}ページ")


if __name__ == "__main__":
    analyze_final_chunk()
    simulate_82_chunks()