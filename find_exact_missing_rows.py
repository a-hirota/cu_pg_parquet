#!/usr/bin/env python3
"""
実際の欠損行を正確に特定する
"""
import psycopg2
import os
import json

def find_missing_rows():
    """欠損行の正確な特定"""
    
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("エラー: GPUPASER_PG_DSN環境変数が設定されていません")
        return
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    print("=== 欠損行の正確な特定 ===\n")
    
    # 1. 実際の総行数
    cur.execute("SELECT COUNT(*) FROM lineorder")
    total_rows = cur.fetchone()[0]
    print(f"PostgreSQLの実際の行数: {total_rows:,}")
    
    # 2. 80チャンクで処理される行数を正確に計算
    total_pages = 4_614_888
    chunks = 80
    pages_per_chunk = total_pages // chunks
    
    total_processed = 0
    chunk_details = []
    
    print("\n各チャンクの実際の行数を計算中...")
    
    for chunk_id in range(chunks):
        start_page = chunk_id * pages_per_chunk
        if chunk_id == chunks - 1:
            # Rustコードの実装に従う
            end_page = total_pages
        else:
            end_page = (chunk_id + 1) * pages_per_chunk
        
        # Rustコードと同じ条件
        cur.execute(f"""
            SELECT COUNT(*) 
            FROM lineorder 
            WHERE ctid >= '({start_page},0)'::tid 
              AND ctid < '({end_page},0)'::tid
        """)
        chunk_rows = cur.fetchone()[0]
        total_processed += chunk_rows
        
        chunk_details.append({
            'chunk_id': chunk_id,
            'start_page': start_page,
            'end_page': end_page,
            'rows': chunk_rows
        })
        
        # 進捗表示
        if chunk_id % 10 == 0 or chunk_id == chunks - 1:
            print(f"  チャンク{chunk_id}まで処理: {total_processed:,}行")
    
    # 3. 欠損の詳細
    missing = total_rows - total_processed
    print(f"\n処理された総行数: {total_processed:,}")
    print(f"欠損行数: {missing:,} ({missing/total_rows*100:.2f}%)")
    
    # 4. 欠損の原因を特定
    print("\n=== 欠損原因の分析 ===")
    
    # チャンク間のギャップを確認
    print("\nチャンク間のギャップを確認中...")
    gaps_found = False
    
    for i in range(len(chunk_details) - 1):
        current_chunk = chunk_details[i]
        next_chunk = chunk_details[i + 1]
        
        if current_chunk['end_page'] != next_chunk['start_page']:
            gaps_found = True
            print(f"  ギャップ発見: チャンク{i}の終了({current_chunk['end_page']}) != チャンク{i+1}の開始({next_chunk['start_page']})")
    
    if not gaps_found:
        print("  チャンク間にギャップはありません")
    
    # 5. ページ単位での欠損を確認
    print("\n空でないページの総数を確認中...")
    cur.execute("""
        SELECT COUNT(DISTINCT (ctid::text::point)[0]::int) 
        FROM lineorder
    """)
    non_empty_pages = cur.fetchone()[0]
    print(f"空でないページ数: {non_empty_pages:,}")
    print(f"pg_relation_sizeのページ数: {total_pages:,}")
    print(f"空ページ率: {(1 - non_empty_pages/total_pages)*100:.1f}%")
    
    # 6. 最も可能性の高い原因
    print("\n=== 欠損の最も可能性の高い原因 ===")
    
    # Binary COPYヘッダーの問題を確認
    print("\n1. Binary COPYのヘッダー/フッターの扱い")
    print("   各チャンクのBinary COPYにはヘッダー（11バイト）とフッター（2バイト）が含まれる")
    print("   これらが行数カウントに影響している可能性")
    
    # 削除済みタプルの確認
    print("\n2. 削除済みタプル（VACUUM未実行）")
    cur.execute("""
        SELECT n_dead_tup 
        FROM pg_stat_user_tables 
        WHERE tablename = 'lineorder'
    """)
    result = cur.fetchone()
    if result:
        dead_tuples = result[0]
        print(f"   削除済みタプル数: {dead_tuples:,}")
    
    # 7. 100%達成の確実な方法
    print("\n=== 100%達成の確実な方法 ===")
    
    print("\n方法1: 全データを1つのCOPYで取得（デバッグ用）")
    print("   COPY (SELECT * FROM lineorder) TO STDOUT (FORMAT BINARY)")
    print("   これで確実に全行を取得できる")
    
    print("\n方法2: 最終チャンクを拡張")
    print("   最後のチャンクで上限を外す")
    print("   ctid >= '(start,0)'::tid のみの条件")
    
    print("\n方法3: 追加の「スイープ」チャンク")
    print("   81番目のチャンクで残りを全て回収")
    
    cur.close()
    conn.close()
    
    # チャンクの詳細を保存
    with open('chunk_row_distribution.json', 'w') as f:
        json.dump({
            'total_rows': total_rows,
            'total_processed': total_processed,
            'missing': missing,
            'chunks': chunk_details
        }, f, indent=2)
    print("\n詳細をchunk_row_distribution.jsonに保存しました")


if __name__ == "__main__":
    find_missing_rows()