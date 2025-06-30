#!/usr/bin/env python3
"""
16チャンクでのテーブルカバレッジを検証
"""
import os
import sys
import psycopg

def main():
    # PostgreSQL接続
    dsn = os.environ.get("GPUPASER_PG_DSN", "dbname=postgres user=postgres host=localhost port=5432")
    
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cursor:
            # テーブルの総行数
            cursor.execute("SELECT COUNT(*) FROM lineorder")
            total_rows = cursor.fetchone()[0]
            print(f"PostgreSQL lineorderテーブル総行数: {total_rows:,}")
            
            # テーブルの総ページ数
            cursor.execute("SELECT relpages FROM pg_class WHERE relname = 'lineorder'")
            total_pages = cursor.fetchone()[0]
            print(f"総ページ数: {total_pages:,}")
            
            # チャンクごとのページ範囲を計算（16チャンクの場合）
            pages_per_chunk = total_pages // 16
            print(f"\n16チャンクでのページ分割:")
            print(f"ページ/チャンク: {pages_per_chunk}")
            
            # 最後のチャンクのページ数
            last_chunk_pages = total_pages - (pages_per_chunk * 15)
            print(f"最後のチャンク（チャンク15）のページ数: {last_chunk_pages}")
            
            # 実際の処理行数との比較
            processed_rows = 148_915_673
            missing_rows = total_rows - processed_rows
            coverage = processed_rows / total_rows * 100
            
            print(f"\n処理結果:")
            print(f"処理された行数: {processed_rows:,}")
            print(f"欠損行数: {missing_rows:,}")
            print(f"カバレッジ: {coverage:.1f}%")
            
            # チャンクごとの推定行数
            print(f"\nチャンクごとの推定:")
            rows_per_page = total_rows / total_pages if total_pages > 0 else 0
            print(f"平均行数/ページ: {rows_per_page:.1f}")
            
            estimated_chunk_rows = pages_per_chunk * rows_per_page
            print(f"推定行数/チャンク: {estimated_chunk_rows:,.0f}")
            
            # 16チャンクでの推定総行数
            estimated_total = estimated_chunk_rows * 16
            print(f"16チャンクでの推定総行数: {estimated_total:,.0f}")
            
            if estimated_total < total_rows:
                print(f"\n⚠️  警告: チャンク分割で一部のデータが欠落する可能性があります")
                print(f"   不足分: {total_rows - estimated_total:,.0f} 行")

if __name__ == "__main__":
    main()