#!/usr/bin/env python3
"""
PostgreSQLで同じデータが異なるTIDから取得できることを確認
Rustと同じSQLクエリを使用してTIDの違いを検証
"""

import os
import psycopg
from typing import List, Dict, Any

def check_tid_duplicates(table_name: str = "customer", sample_keys: List[int] = None):
    """
    指定されたテーブルで重複キーのTIDを確認
    
    Args:
        table_name: 対象テーブル名
        sample_keys: チェックするキーのリスト（Noneの場合は重複キーを自動検出）
    """
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        raise RuntimeError("環境変数 GPUPASER_PG_DSN が設定されていません")
    
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # テーブルのキー列を特定
            cur.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' 
                AND ordinal_position = 1
            """)
            key_column = cur.fetchone()
            
            if not key_column:
                print(f"❌ テーブル {table_name} のキー列が見つかりません")
                return
            
            key_column_name = key_column[0]
            print(f"テーブル: {table_name}")
            print(f"キー列: {key_column_name}")
            print("=" * 80)
            
            # sample_keysが指定されていない場合は重複キーを検出
            if sample_keys is None:
                print("\n重複キーを検出中...")
                cur.execute(f"""
                    SELECT {key_column_name}, COUNT(*) as cnt
                    FROM {table_name}
                    GROUP BY {key_column_name}
                    HAVING COUNT(*) > 1
                    ORDER BY cnt DESC, {key_column_name}
                    LIMIT 10
                """)
                duplicates = cur.fetchall()
                
                if not duplicates:
                    print("✅ 重複キーが見つかりません")
                    return
                
                print(f"\n見つかった重複キー（上位10個）:")
                for key, count in duplicates:
                    print(f"  {key_column_name}={key}: {count}個")
                
                sample_keys = [dup[0] for dup in duplicates[:5]]  # 上位5個を詳細分析
            
            print(f"\n\n詳細分析対象キー: {sample_keys}")
            print("=" * 80)
            
            # 各キーについてTIDを含む詳細情報を取得
            for key in sample_keys:
                print(f"\n【{key_column_name} = {key}】")
                
                # Rustと同じように全列を取得し、TIDも含める
                # ctidは特殊な列で、物理的な行の位置を示す
                cur.execute(f"""
                    SELECT ctid, * 
                    FROM {table_name}
                    WHERE {key_column_name} = %s
                    ORDER BY ctid
                """, (key,))
                
                rows = cur.fetchall()
                print(f"重複数: {len(rows)}個")
                
                # 列名を取得
                col_names = [desc[0] for desc in cur.description]
                
                # 各行の詳細を表示
                for i, row in enumerate(rows):
                    print(f"\n  行 {i+1}:")
                    print(f"    TID: {row[0]}")  # ctidは最初の列
                    
                    # TIDを(page, item)形式で解析
                    tid_str = str(row[0])
                    if ',' in tid_str:
                        tid_parts = tid_str.strip('()').split(',')
                        page_num = int(tid_parts[0])
                        item_num = int(tid_parts[1])
                        print(f"    └─ ページ番号: {page_num}, アイテム番号: {item_num}")
                    
                    # 最初の5列のデータを表示（ctid以外）
                    print(f"    データ:")
                    for j in range(1, min(6, len(row))):
                        if j-1 < len(col_names)-1:
                            print(f"      {col_names[j]}: {row[j]}")
                
                # 同じデータが異なるTIDに存在することを確認
                if len(rows) > 1:
                    # データ部分（ctid以外）が同じかチェック
                    first_data = rows[0][1:]  # 最初の行のデータ部分
                    all_same = all(row[1:] == first_data for row in rows[1:])
                    
                    if all_same:
                        print(f"\n  ✅ 確認: 同じデータが異なるTIDに存在します")
                        tids = [str(row[0]) for row in rows]
                        print(f"  TIDリスト: {tids}")
                    else:
                        print(f"\n  ⚠️  データに差異があります（キーは同じだが値が異なる）")
                
                print("-" * 60)
            
            # Rustが使用するクエリをシミュレート
            print("\n\n【Rustクエリのシミュレーション】")
            print("=" * 80)
            
            # ORDER BY句とLIMIT/OFFSETでチャンク分割した場合の動作確認
            chunk_size = 1000000  # 仮のチャンクサイズ
            
            for key in sample_keys[:3]:  # 最初の3つのキーで確認
                print(f"\n{key_column_name} = {key} のチャンク境界での出現:")
                
                # このキーが出現する行番号を取得
                cur.execute(f"""
                    WITH numbered_rows AS (
                        SELECT {key_column_name}, ctid,
                               ROW_NUMBER() OVER (ORDER BY {key_column_name}) as rn
                        FROM {table_name}
                    )
                    SELECT rn, ctid
                    FROM numbered_rows
                    WHERE {key_column_name} = %s
                    ORDER BY rn
                """, (key,))
                
                positions = cur.fetchall()
                
                for rn, tid in positions:
                    chunk_id = (rn - 1) // chunk_size
                    position_in_chunk = (rn - 1) % chunk_size
                    print(f"  行番号: {rn:,} (チャンク{chunk_id}, 位置{position_in_chunk:,}) - TID: {tid}")
                
                # チャンク境界をまたぐかチェック
                if len(positions) > 1:
                    chunks = set((rn - 1) // chunk_size for rn, _ in positions)
                    if len(chunks) > 1:
                        print(f"  ⚠️  このキーは複数のチャンク({sorted(chunks)})に分散しています")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='PostgreSQL TID重複チェック')
    parser.add_argument('--table', type=str, default='customer', help='対象テーブル名')
    parser.add_argument('--keys', type=int, nargs='+', help='チェックするキー値のリスト')
    args = parser.parse_args()
    
    check_tid_duplicates(args.table, args.keys)


if __name__ == "__main__":
    main()