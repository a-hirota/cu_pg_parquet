#!/usr/bin/env python3
"""
欠損している3,626,274行の詳細な位置を特定
"""
import psycopg2
import os

def investigate_missing_rows():
    """欠損行の詳細調査"""
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("エラー: GPUPASER_PG_DSN環境変数が設定されていません")
        return
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    print("=== 欠損行の詳細調査 ===\n")
    
    # 1. テーブルの実際の行数
    cur.execute("SELECT COUNT(*) FROM lineorder")
    total_rows = cur.fetchone()[0]
    print(f"実際の総行数: {total_rows:,}")
    
    # 2. 最大ctidを確認
    cur.execute("SELECT MAX(ctid) FROM lineorder")
    max_ctid = cur.fetchone()[0]
    print(f"最大ctid: {max_ctid}")
    
    # ctidをページ番号とタプル番号に分解
    page_num, tuple_num = map(int, max_ctid.strip('()').split(','))
    print(f"最大ページ番号: {page_num:,}")
    print(f"最大タプル番号: {tuple_num}")
    
    # 3. 80チャンクでカバーされる範囲を計算
    total_pages = 4_614_888
    chunks = 80
    pages_per_chunk = total_pages // chunks
    last_chunk_end = total_pages  # 80番目のチャンクの終了ページ
    
    print(f"\n80チャンクでのカバー範囲:")
    print(f"総ページ数（pg_relation_size）: {total_pages:,}")
    print(f"チャンクあたりページ数: {pages_per_chunk:,}")
    print(f"最後のチャンク終了ページ: {last_chunk_end:,}")
    
    # 4. カバーされていない範囲を確認
    if page_num >= last_chunk_end:
        print(f"\n⚠️ 最大ctidのページ({page_num})が80チャンクの範囲外です！")
        
        # カバーされていない範囲の行数を確認
        cur.execute(f"""
            SELECT COUNT(*) 
            FROM lineorder 
            WHERE ctid >= '({last_chunk_end},0)'::tid
        """)
        uncovered_rows = cur.fetchone()[0]
        print(f"カバーされていない行数: {uncovered_rows:,}")
    else:
        print(f"\n✓ 最大ctidのページ({page_num})は80チャンクの範囲内です")
        
        # 最後の数ページの行数を確認
        print("\n最後の10ページの行数分布:")
        for i in range(10):
            check_page = last_chunk_end - i - 1
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM lineorder 
                WHERE ctid >= '({check_page},0)'::tid 
                  AND ctid < '({check_page + 1},0)'::tid
            """)
            rows_in_page = cur.fetchone()[0]
            if rows_in_page > 0:
                print(f"  ページ{check_page}: {rows_in_page}行")
    
    # 5. チャンクごとの実際の行数を確認（最後の5チャンク）
    print("\n最後の5チャンクの推定行数:")
    for chunk_id in range(75, 80):
        start_page = chunk_id * pages_per_chunk
        end_page = last_chunk_end if chunk_id == 79 else (chunk_id + 1) * pages_per_chunk
        
        cur.execute(f"""
            SELECT COUNT(*) 
            FROM lineorder 
            WHERE ctid >= '({start_page},0)'::tid 
              AND ctid < '({end_page},0)'::tid
        """)
        rows_in_chunk = cur.fetchone()[0]
        print(f"  チャンク{chunk_id}: {rows_in_chunk:,}行 (ページ{start_page:,}-{end_page:,})")
    
    # 6. 必要なチャンク数を正確に計算
    print("\n=== 100%達成のための計算 ===")
    
    # 実際のデータが存在する最大ページまでカバーする必要がある
    required_chunks = (page_num + pages_per_chunk) // pages_per_chunk
    print(f"データが存在する最大ページ: {page_num:,}")
    print(f"100%カバーに必要なチャンク数: {required_chunks}")
    
    # 7. 82チャンクでの予測
    if required_chunks <= 82:
        print(f"\n✓ 82チャンクで100%カバー可能です！")
    else:
        print(f"\n⚠️ {required_chunks}チャンク必要です")
    
    cur.close()
    conn.close()


def calculate_exact_chunk_distribution():
    """正確なチャンク分布を計算"""
    
    print("\n\n=== 正確なチャンク分布計算 ===")
    
    total_pages = 4_614_888
    max_data_page = 4_614_888  # 実際にはもっと少ない可能性
    
    for chunks in [80, 81, 82, 85]:
        pages_per_chunk = total_pages // chunks
        last_chunk_start = (chunks - 1) * pages_per_chunk
        last_chunk_end = total_pages
        
        print(f"\n{chunks}チャンクの場合:")
        print(f"  チャンクあたり: {pages_per_chunk:,}ページ")
        print(f"  最終チャンク: ページ{last_chunk_start:,}-{last_chunk_end:,}")
        print(f"  カバー率: {min(100, (chunks * pages_per_chunk / max_data_page) * 100):.2f}%")


if __name__ == "__main__":
    investigate_missing_rows()
    calculate_exact_chunk_distribution()