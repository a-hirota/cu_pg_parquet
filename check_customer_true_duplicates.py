#!/usr/bin/env python3
"""
customerテーブルで真の重複データ（全列が同一）を確認
"""

import os
import psycopg
import hashlib

def check_customer_true_duplicates():
    """customerテーブルで真の重複データを確認"""
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            print("=== customerテーブルの真の重複チェック ===\n")
            
            # 1. まず全データのハッシュ値を計算して重複を見つける
            print("全行のハッシュ値を計算中...")
            cur.execute("""
                WITH row_hashes AS (
                    SELECT 
                        ctid,
                        c_custkey,
                        c_name,
                        c_address,
                        c_city,
                        c_nation,
                        c_region,
                        c_phone,
                        c_mktsegment,
                        md5(
                            COALESCE(c_custkey::text, '') || '|' ||
                            COALESCE(c_name, '') || '|' ||
                            COALESCE(c_address, '') || '|' ||
                            COALESCE(c_city, '') || '|' ||
                            COALESCE(c_nation, '') || '|' ||
                            COALESCE(c_region, '') || '|' ||
                            COALESCE(c_phone, '') || '|' ||
                            COALESCE(c_mktsegment, '')
                        ) as row_hash
                    FROM customer
                ),
                duplicate_hashes AS (
                    SELECT row_hash, COUNT(*) as cnt
                    FROM row_hashes
                    GROUP BY row_hash
                    HAVING COUNT(*) > 1
                )
                SELECT 
                    rh.ctid,
                    rh.c_custkey,
                    rh.c_name,
                    rh.c_address,
                    rh.c_city,
                    rh.c_nation,
                    rh.c_region,
                    rh.c_phone,
                    rh.c_mktsegment,
                    rh.row_hash
                FROM row_hashes rh
                INNER JOIN duplicate_hashes dh ON rh.row_hash = dh.row_hash
                ORDER BY rh.row_hash, rh.ctid
            """)
            
            duplicates = cur.fetchall()
            
            if not duplicates:
                print("✅ 完全に同一の重複行は見つかりません\n")
                
                # 2. c_custkeyの重複をチェック
                print("c_custkeyの重複をチェック中...")
                cur.execute("""
                    SELECT c_custkey, COUNT(*) as cnt
                    FROM customer
                    GROUP BY c_custkey
                    HAVING COUNT(*) > 1
                    ORDER BY cnt DESC
                    LIMIT 10
                """)
                
                key_dups = cur.fetchall()
                if key_dups:
                    print("\nc_custkeyが重複している行:")
                    for key, cnt in key_dups:
                        print(f"  c_custkey={key}: {cnt}個")
                else:
                    print("✅ c_custkeyの重複もありません")
                
                return
            
            # 3. 重複データの詳細を表示
            print(f"\n⚠️  完全に同一の重複行が見つかりました！")
            print(f"重複行数: {len(duplicates)}行\n")
            
            # ハッシュ値でグループ化
            hash_groups = {}
            for row in duplicates:
                hash_val = row[-1]
                if hash_val not in hash_groups:
                    hash_groups[hash_val] = []
                hash_groups[hash_val].append(row[:-1])  # ハッシュ値以外を保存
            
            print(f"重複グループ数: {len(hash_groups)}個\n")
            
            # 各グループの詳細を表示（最初の5グループのみ）
            for i, (hash_val, rows) in enumerate(list(hash_groups.items())[:5]):
                print(f"\n【重複グループ {i+1}】 (ハッシュ: {hash_val[:8]}...)")
                print(f"重複数: {len(rows)}個")
                print("-" * 80)
                
                # 最初の行のデータを表示
                first_row = rows[0]
                print(f"データ内容:")
                print(f"  c_custkey: {first_row[1]}")
                print(f"  c_name: {first_row[2]}")
                print(f"  c_address: {first_row[3]}")
                print(f"  c_city: {first_row[4]}")
                print(f"  c_nation: {first_row[5]}")
                print(f"  c_region: {first_row[6]}")
                print(f"  c_phone: {first_row[7]}")
                print(f"  c_mktsegment: {first_row[8]}")
                
                print(f"\nTID（物理的位置）:")
                for row in rows:
                    tid = row[0]
                    print(f"  - {tid}", end="")
                    # TIDを解析
                    tid_str = str(tid)
                    if ',' in tid_str:
                        tid_parts = tid_str.strip('()').split(',')
                        page_num = int(tid_parts[0])
                        item_num = int(tid_parts[1])
                        print(f" (ページ: {page_num}, アイテム: {item_num})")
                    else:
                        print()
                
                # 異なるページに存在するか確認
                pages = set()
                for row in rows:
                    tid_str = str(row[0])
                    if ',' in tid_str:
                        tid_parts = tid_str.strip('()').split(',')
                        pages.add(int(tid_parts[0]))
                
                if len(pages) > 1:
                    print(f"\n⚠️  この重複データは{len(pages)}個の異なるページに分散しています: {sorted(pages)}")
                else:
                    print(f"\n📍 この重複データは同じページ({list(pages)[0]})内にあります")
            
            # 4. Rustのクエリシミュレーション
            if len(hash_groups) > 0:
                print("\n\n=== Rustクエリのシミュレーション ===")
                
                # 最初の重複グループのc_custkeyを使用
                sample_key = list(hash_groups.values())[0][0][1]
                
                print(f"\nc_custkey = {sample_key} でRustと同じクエリを実行:")
                cur.execute("""
                    SELECT ctid, * FROM customer 
                    WHERE c_custkey = %s
                    ORDER BY c_custkey
                """, (sample_key,))
                
                rust_results = cur.fetchall()
                print(f"\n結果: {len(rust_results)}行")
                
                for i, row in enumerate(rust_results):
                    print(f"\n行{i+1}:")
                    print(f"  TID: {row[0]}")
                    print(f"  データ: c_custkey={row[1]}, c_name={row[2][:20]}...")
                
                if len(rust_results) > 1:
                    print(f"\n✅ 確認: Rustが同じSQLで{len(rust_results)}個の異なるTIDから同じデータを取得する可能性があります")


def main():
    check_customer_true_duplicates()


if __name__ == "__main__":
    main()