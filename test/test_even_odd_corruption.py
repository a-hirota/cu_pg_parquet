#!/usr/bin/env python3
"""
偶数行破損問題の調査テスト
========================

偶数チャンクのParquetファイルが破損する問題を調査
"""

import os
import sys
import subprocess
import psycopg2
import cudf
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_table(conn):
    """偶数/奇数行パターンを持つテストテーブルを作成"""
    cur = conn.cursor()
    
    # テーブル削除
    cur.execute("DROP TABLE IF EXISTS test_even_odd")
    
    # テーブル作成
    cur.execute("""
        CREATE TABLE test_even_odd (
            row_id INTEGER PRIMARY KEY,
            row_type VARCHAR(10),
            test_int INTEGER,
            test_text TEXT,
            test_decimal DECIMAL(10,2)
        )
    """)
    
    # テストデータ挿入（1000行）
    data = []
    for i in range(1000):
        row_type = 'EVEN' if i % 2 == 0 else 'ODD'
        data.append((
            i,
            row_type,
            i * 100,
            f"Test row {i}: {row_type}",
            float(i) * 1.23
        ))
    
    cur.executemany("""
        INSERT INTO test_even_odd (row_id, row_type, test_int, test_text, test_decimal)
        VALUES (%s, %s, %s, %s, %s)
    """, data)
    
    conn.commit()
    cur.close()
    
    print(f"テストテーブル作成完了: 1000行")


def run_gpu_processing(table_name, output_dir, chunks=2):
    """GPU処理を実行し、結果を返す"""
    env = os.environ.copy()
    env["GPUPASER_PG_DSN"] = "dbname=postgres user=postgres host=localhost port=5432"
    env["GPUPGPARSER_TEST_MODE"] = "1"  # テストモード有効化
    env["RUST_PARALLEL_CONNECTIONS"] = str(chunks)
    
    cmd = [
        "source", "/home/ubuntu/miniconda3/etc/profile.d/conda.sh", "&&",
        "conda", "activate", "cudf_dev", "&&",
        "python", "cu_pg_parquet.py",
        "--test",
        "--table", table_name,
        "--parallel", str(chunks),
        "--chunks", str(chunks),
        "--output", output_dir
    ]
    
    cmd_str = " ".join(cmd)
    
    result = subprocess.run(
        cmd_str,
        shell=True,
        executable='/bin/bash',
        env=env,
        capture_output=True,
        text=True,
        cwd="/home/ubuntu/gpupgparser"
    )
    
    return result


def analyze_parquet_files(output_dir):
    """生成されたParquetファイルを分析"""
    parquet_files = sorted(Path(output_dir).glob("*.parquet"))
    
    print(f"\n=== Parquetファイル分析 ===")
    print(f"ファイル数: {len(parquet_files)}")
    
    analysis = {}
    
    for idx, pf in enumerate(parquet_files):
        print(f"\n--- ファイル {idx + 1}: {pf.name} ---")
        
        try:
            # cuDFで読み込み
            df = cudf.read_parquet(pf)
            rows = len(df)
            
            # Pandas DataFrameに変換して詳細分析
            pdf = df.to_pandas()
            
            # 偶数/奇数行の数をカウント
            if 'row_type' in pdf.columns:
                even_count = len(pdf[pdf['row_type'] == 'EVEN'])
                odd_count = len(pdf[pdf['row_type'] == 'ODD'])
            else:
                # row_idから判定
                even_count = len(pdf[pdf['row_id'] % 2 == 0])
                odd_count = len(pdf[pdf['row_id'] % 2 == 1])
            
            # 最初と最後の5行を表示
            print(f"行数: {rows}")
            print(f"偶数行: {even_count}, 奇数行: {odd_count}")
            
            if rows > 0:
                print("\n最初の5行:")
                print(pdf.head())
                print("\n最後の5行:")
                print(pdf.tail())
                
                # 行IDの連続性チェック
                if 'row_id' in pdf.columns:
                    row_ids = pdf['row_id'].values
                    gaps = []
                    for i in range(1, len(row_ids)):
                        if row_ids[i] != row_ids[i-1] + 1:
                            gaps.append((row_ids[i-1], row_ids[i]))
                    
                    if gaps:
                        print(f"\n行IDのギャップ検出: {len(gaps)}箇所")
                        for gap in gaps[:5]:  # 最初の5つ
                            print(f"  {gap[0]} -> {gap[1]} (差: {gap[1] - gap[0]})")
            
            analysis[pf.name] = {
                'rows': rows,
                'even_count': even_count,
                'odd_count': odd_count,
                'status': 'OK' if even_count > 0 and odd_count > 0 else 'CORRUPTED'
            }
            
        except Exception as e:
            print(f"エラー: {type(e).__name__}: {e}")
            analysis[pf.name] = {
                'rows': 0,
                'even_count': 0,
                'odd_count': 0,
                'status': 'ERROR',
                'error': str(e)
            }
    
    return analysis


def check_grid_boundary_info(stdout):
    """Grid境界スレッド情報から偶数/奇数パターンを分析"""
    print("\n=== Grid境界スレッド分析 ===")
    
    lines = stdout.split('\n')
    in_grid_section = False
    thread_info = []
    
    for line in lines:
        if "Grid境界スレッドデバッグ情報" in line:
            in_grid_section = True
        elif "デバッグ情報をJSONファイルに保存" in line:
            in_grid_section = False
        elif in_grid_section and "Thread ID:" in line:
            thread_id = int(line.split(":")[1].strip())
            thread_info.append({'thread_id': thread_id})
        elif in_grid_section and "Row Position:" in line:
            parts = line.split(",")
            row_pos = int(parts[0].split(":")[1].strip())
            if thread_info:
                thread_info[-1]['row_position'] = row_pos
    
    # スレッドの偶数/奇数分析
    even_threads = 0
    odd_threads = 0
    
    for ti in thread_info:
        if 'thread_id' in ti:
            if ti['thread_id'] % 2 == 0:
                even_threads += 1
            else:
                odd_threads += 1
    
    print(f"偶数Thread ID: {even_threads}個")
    print(f"奇数Thread ID: {odd_threads}個")
    
    # チャンクごとの分析も必要
    return thread_info


def main():
    """メインテスト実行"""
    print("=== 偶数行破損問題調査テスト ===")
    
    # PostgreSQL接続
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        host="localhost",
        port=5432
    )
    
    try:
        # 1. テストテーブル作成
        create_test_table(conn)
        
        # 2. GPU処理実行（2チャンク）
        output_dir = tempfile.mkdtemp(prefix="test_even_odd_")
        print(f"\n出力ディレクトリ: {output_dir}")
        
        result = run_gpu_processing("test_even_odd", output_dir, chunks=2)
        
        print(f"\n実行結果: リターンコード={result.returncode}")
        
        if result.returncode != 0:
            print("標準エラー:")
            print(result.stderr[:2000])
            return
        
        # 3. Grid境界情報分析
        if result.stdout:
            thread_info = check_grid_boundary_info(result.stdout)
        
        # 4. Parquetファイル分析
        analysis = analyze_parquet_files(output_dir)
        
        # 5. 結果サマリー
        print("\n=== 結果サマリー ===")
        for fname, info in analysis.items():
            print(f"{fname}: {info['status']} - 偶数:{info['even_count']}, 奇数:{info['odd_count']}")
        
        # 6. 破損パターンの特定
        corrupted = [f for f, info in analysis.items() if info['status'] == 'CORRUPTED']
        if corrupted:
            print(f"\n⚠️ 破損ファイル検出: {len(corrupted)}個")
            for cf in corrupted:
                # チャンク番号を推定
                if 'chunk' in cf:
                    chunk_num = int(cf.split('chunk')[1].split('.')[0])
                    print(f"  - {cf} (チャンク番号: {chunk_num}, {'偶数' if chunk_num % 2 == 0 else '奇数'})")
        
        # クリーンアップ
        shutil.rmtree(output_dir)
        
    finally:
        # テーブル削除
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS test_even_odd")
        conn.commit()
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()