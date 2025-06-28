#!/usr/bin/env python3
"""
lineorderテーブルの偶数チャンク破損調査
=====================================

実際のlineorderテーブルで偶数チャンクの破損を調査
"""

import os
import sys
import subprocess
import cudf
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_gpu_processing_with_output(chunks=4):
    """GPU処理を実行し、出力ディレクトリと結果を返す"""
    output_dir = tempfile.mkdtemp(prefix="test_lineorder_even_")
    
    env = os.environ.copy()
    env["GPUPASER_PG_DSN"] = "dbname=postgres user=postgres host=localhost port=5432"
    env["GPUPGPARSER_TEST_MODE"] = "1"  # テストモード有効化
    env["RUST_PARALLEL_CONNECTIONS"] = str(chunks)
    
    cmd = [
        "source", "/home/ubuntu/miniconda3/etc/profile.d/conda.sh", "&&",
        "conda", "activate", "cudf_dev", "&&",
        "python", "cu_pg_parquet.py",
        "--test",
        "--table", "lineorder",
        "--parallel", str(chunks),
        "--chunks", str(chunks),
        "--output", output_dir
    ]
    
    cmd_str = " ".join(cmd)
    
    print(f"実行コマンド: {cmd_str}")
    print(f"出力ディレクトリ: {output_dir}")
    
    result = subprocess.run(
        cmd_str,
        shell=True,
        executable='/bin/bash',
        env=env,
        capture_output=True,
        text=True,
        cwd="/home/ubuntu/gpupgparser"
    )
    
    return output_dir, result


def analyze_chunk_patterns(output_dir):
    """チャンクごとのParquetファイルを分析"""
    parquet_files = sorted(Path(output_dir).glob("*.parquet"))
    
    print(f"\n=== Parquetファイル分析 ===")
    print(f"ファイル数: {len(parquet_files)}")
    
    chunk_analysis = []
    
    for pf in parquet_files:
        print(f"\n--- ファイル: {pf.name} ---")
        
        # チャンク番号を抽出
        if 'chunk' in pf.name:
            try:
                chunk_num = int(pf.name.split('chunk')[1].split('.')[0])
            except:
                chunk_num = -1
        else:
            chunk_num = -1
        
        try:
            # cuDFで読み込み
            df = cudf.read_parquet(pf)
            rows = len(df)
            
            # 最初の10行を取得して検証
            if rows > 0:
                pdf_head = df.head(10).to_pandas()
                
                # データの妥当性チェック
                null_counts = pdf_head.isnull().sum()
                
                # lo_orderkeyの範囲チェック
                if 'lo_orderkey' in pdf_head.columns:
                    orderkey_min = pdf_head['lo_orderkey'].min()
                    orderkey_max = pdf_head['lo_orderkey'].max()
                else:
                    orderkey_min = orderkey_max = None
                
                chunk_info = {
                    'file': pf.name,
                    'chunk_num': chunk_num,
                    'is_even_chunk': chunk_num % 2 == 0 if chunk_num >= 0 else None,
                    'rows': rows,
                    'null_counts': null_counts.to_dict(),
                    'orderkey_range': (orderkey_min, orderkey_max),
                    'status': 'OK'
                }
                
                print(f"チャンク番号: {chunk_num} ({'偶数' if chunk_num % 2 == 0 else '奇数'})")
                print(f"行数: {rows:,}")
                print(f"NULL値: {null_counts.sum()}")
                if orderkey_min is not None:
                    print(f"lo_orderkey範囲: {orderkey_min} - {orderkey_max}")
                
                # 偶数チャンクの場合、詳細確認
                if chunk_num % 2 == 0:
                    print("\n[偶数チャンク詳細]")
                    print("最初の5行:")
                    print(pdf_head.head())
            else:
                chunk_info = {
                    'file': pf.name,
                    'chunk_num': chunk_num,
                    'is_even_chunk': chunk_num % 2 == 0 if chunk_num >= 0 else None,
                    'rows': 0,
                    'status': 'EMPTY'
                }
                print("⚠️ 空のファイル")
            
        except Exception as e:
            chunk_info = {
                'file': pf.name,
                'chunk_num': chunk_num,
                'is_even_chunk': chunk_num % 2 == 0 if chunk_num >= 0 else None,
                'rows': 0,
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ エラー: {type(e).__name__}: {e}")
        
        chunk_analysis.append(chunk_info)
    
    return chunk_analysis


def extract_grid_debug_info(stdout):
    """Grid境界デバッグ情報を抽出"""
    # JSONファイルパスを探す
    for line in stdout.split('\n'):
        if "デバッグ情報をJSONファイルに保存" in line and ".json" in line:
            json_path = line.split(": ")[-1].strip()
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    return json.load(f)
    return None


def analyze_thread_patterns(debug_info):
    """スレッドパターンを分析"""
    if not debug_info:
        return
    
    print("\n=== スレッドパターン分析 ===")
    
    # Block Xごとにグループ化
    block_groups = {}
    for entry in debug_info:
        block_x = entry['block_x']
        if block_x not in block_groups:
            block_groups[block_x] = []
        block_groups[block_x].append(entry)
    
    # 偶数/奇数ブロックの分析
    even_blocks = [bx for bx in block_groups.keys() if bx % 2 == 0]
    odd_blocks = [bx for bx in block_groups.keys() if bx % 2 == 1]
    
    print(f"偶数ブロック: {len(even_blocks)}個")
    print(f"奇数ブロック: {len(odd_blocks)}個")
    
    # 各ブロックの行位置を確認
    print("\n偶数ブロックの行位置サンプル:")
    for bx in sorted(even_blocks)[:5]:  # 最初の5つ
        entries = block_groups[bx]
        positions = [e['row_position'] for e in entries]
        print(f"  Block {bx}: positions={positions[:3]}...")


def main():
    """メインテスト実行"""
    print("=== lineorderテーブル偶数チャンク破損調査 ===")
    
    # 1. GPU処理実行（4チャンク）
    output_dir, result = run_gpu_processing_with_output(chunks=4)
    
    print(f"\n実行結果: リターンコード={result.returncode}")
    
    if result.returncode != 0:
        print("\n標準エラー:")
        print(result.stderr[:3000])
    else:
        # 2. Grid境界デバッグ情報を抽出
        debug_info = extract_grid_debug_info(result.stdout)
        if debug_info:
            analyze_thread_patterns(debug_info)
        
        # 3. チャンクパターン分析
        chunk_analysis = analyze_chunk_patterns(output_dir)
        
        # 4. 結果サマリー
        print("\n=== 結果サマリー ===")
        even_chunks = [ca for ca in chunk_analysis if ca.get('is_even_chunk') == True]
        odd_chunks = [ca for ca in chunk_analysis if ca.get('is_even_chunk') == False]
        
        print(f"偶数チャンク: {len(even_chunks)}個")
        for ec in even_chunks:
            status_icon = "✅" if ec['status'] == 'OK' else "❌"
            print(f"  {status_icon} チャンク{ec['chunk_num']}: {ec['rows']:,}行 ({ec['status']})")
        
        print(f"\n奇数チャンク: {len(odd_chunks)}個")
        for oc in odd_chunks:
            status_icon = "✅" if oc['status'] == 'OK' else "❌"
            print(f"  {status_icon} チャンク{oc['chunk_num']}: {oc['rows']:,}行 ({oc['status']})")
        
        # 5. 破損パターンの特定
        corrupted_even = [ec for ec in even_chunks if ec['status'] != 'OK']
        if corrupted_even:
            print(f"\n⚠️ 偶数チャンクの問題検出: {len(corrupted_even)}個")
            for ce in corrupted_even:
                print(f"  - チャンク{ce['chunk_num']}: {ce.get('error', ce['status'])}")
    
    # クリーンアップ
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"\n出力ディレクトリを削除しました: {output_dir}")


if __name__ == "__main__":
    main()