#!/usr/bin/env python3
"""
チャンク出力の分析スクリプト
========================

テストモードで実行されたGPU処理のログを解析し、
偶数チャンクの問題を特定する
"""

import re
import sys
import json
from collections import defaultdict


def parse_log(log_content):
    """ログ内容を解析してチャンク情報を抽出"""
    chunk_data = defaultdict(dict)
    sort_debug_info = []
    
    # パターン定義
    patterns = {
        'chunk_complete': r'\[Consumer-\d+\] チャンク (\d+) \((偶数|奇数)チャンク\) GPU処理完了 \([\d.]+秒, ([\d,]+)行\)',
        'chunk_debug': r'\[CHUNK DEBUG\] チャンク (\d+)',
        'file_size': r'- ファイルサイズ: ([\d.]+) MB',
        'rows': r'- 検出行数: ([\d,]+)行',
        'parse_time': r'- GPUパース時間: ([\d.]+)秒',
        'bytes_per_row': r'- 行あたり: ([\d.]+) bytes/row',
        'sort_debug': r'\[SORT DEBUG\] (.+)',
    }
    
    lines = log_content.split('\n')
    current_chunk = None
    
    for line in lines:
        # チャンク完了情報
        match = re.search(patterns['chunk_complete'], line)
        if match:
            chunk_id = int(match.group(1))
            parity = match.group(2)
            rows = int(match.group(3).replace(',', ''))
            chunk_data[chunk_id]['parity'] = parity
            chunk_data[chunk_id]['rows'] = rows
        
        # チャンクデバッグ情報
        match = re.search(patterns['chunk_debug'], line)
        if match:
            current_chunk = int(match.group(1))
        
        # ファイルサイズ
        if current_chunk and re.search(patterns['file_size'], line):
            match = re.search(patterns['file_size'], line)
            chunk_data[current_chunk]['file_size_mb'] = float(match.group(1))
        
        # ソートデバッグ情報
        if '[SORT DEBUG]' in line:
            sort_debug_info.append(line)
    
    return chunk_data, sort_debug_info


def analyze_chunks(chunk_data):
    """チャンクデータを分析"""
    even_chunks = []
    odd_chunks = []
    
    for chunk_id, data in sorted(chunk_data.items()):
        if data.get('parity') == '偶数':
            even_chunks.append((chunk_id, data))
        else:
            odd_chunks.append((chunk_id, data))
    
    print("=== チャンク分析結果 ===\n")
    
    # 偶数チャンク
    print("【偶数チャンク】")
    even_total_rows = 0
    for chunk_id, data in even_chunks:
        rows = data.get('rows', 0)
        even_total_rows += rows
        print(f"  チャンク {chunk_id}: {rows:,}行")
        if rows == 0:
            print("    ⚠️ 行数が0！")
    
    # 奇数チャンク
    print("\n【奇数チャンク】")
    odd_total_rows = 0
    for chunk_id, data in odd_chunks:
        rows = data.get('rows', 0)
        odd_total_rows += rows
        print(f"  チャンク {chunk_id}: {rows:,}行")
    
    # 統計
    print("\n【統計】")
    if even_chunks:
        even_avg = even_total_rows / len(even_chunks)
        print(f"偶数チャンク平均: {even_avg:,.0f}行")
    
    if odd_chunks:
        odd_avg = odd_total_rows / len(odd_chunks)
        print(f"奇数チャンク平均: {odd_avg:,.0f}行")
    
    if even_chunks and odd_chunks:
        ratio = even_avg / odd_avg if odd_avg > 0 else 0
        print(f"行数比率（偶数/奇数）: {ratio:.3f}")
        
        if ratio < 0.5:
            print("\n⚠️ 問題検出: 偶数チャンクの行数が異常に少ない")
        elif ratio > 1.5:
            print("\n⚠️ 問題検出: 偶数チャンクの行数が異常に多い")
        else:
            print("\n✅ 偶数/奇数チャンクの行数は正常範囲内")


def analyze_sort_debug(sort_debug_info):
    """ソートデバッグ情報を分析"""
    if not sort_debug_info:
        return
    
    print("\n\n=== GPUソートデバッグ情報 ===")
    
    for line in sort_debug_info:
        print(line)
        
        # 無効な行が検出された場合
        if "無効な行位置" in line:
            print("  ⚠️ 負の行位置が検出されました！")
        
        # 偶数/奇数の偏りをチェック
        if "有効行の偶数位置=" in line:
            match = re.search(r'偶数位置=(\d+)行.*奇数位置=(\d+)行', line)
            if match:
                even = int(match.group(1))
                odd = int(match.group(2))
                if even == 0 and odd > 0:
                    print("  ⚠️ 偶数位置の行が1つも検出されていません！")
                elif odd == 0 and even > 0:
                    print("  ⚠️ 奇数位置の行が1つも検出されていません！")


def main():
    """メイン処理"""
    print("チャンク出力分析ツール")
    print("=" * 50)
    
    # 標準入力またはファイルから読み込み
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            log_content = f.read()
    else:
        print("使用方法: python analyze_chunk_output.py < ログファイル")
        print("または: python analyze_chunk_output.py ログファイル")
        log_content = sys.stdin.read()
    
    # ログ解析
    chunk_data, sort_debug_info = parse_log(log_content)
    
    if not chunk_data:
        print("\nチャンク情報が見つかりませんでした。")
        print("テストモードで実行していることを確認してください:")
        print("  export GPUPGPARSER_TEST_MODE=1")
        return
    
    # 分析実行
    analyze_chunks(chunk_data)
    analyze_sort_debug(sort_debug_info)


if __name__ == "__main__":
    main()