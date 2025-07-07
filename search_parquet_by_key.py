#!/usr/bin/env python3
"""
Parquetファイル内の特定のc_custkeyを検索して詳細情報を表示
"""

import cudf
import sys
from pathlib import Path

def search_customer_in_parquet(key_value, parquet_pattern="output/chunk_*_queue.parquet"):
    """
    指定されたc_custkeyをParquetファイルから検索
    
    Args:
        key_value: 検索するc_custkeyの値
        parquet_pattern: Parquetファイルのパターン
    """
    print(f"=== c_custkey = {key_value} の検索 ===\n")
    
    # Parquetファイルを検索
    parquet_files = sorted(Path(".").glob(parquet_pattern))
    
    if not parquet_files:
        print(f"❌ Parquetファイルが見つかりません: {parquet_pattern}")
        return
    
    print(f"検索対象ファイル数: {len(parquet_files)}\n")
    
    total_found = 0
    
    for pf in parquet_files:
        try:
            # Parquetファイルを読み込み
            df = cudf.read_parquet(pf)
            
            # c_custkeyでフィルタリング
            if 'c_custkey' in df.columns:
                # Decimal型の場合は整数に変換
                if hasattr(df['c_custkey'].dtype, 'precision'):
                    matches = df[df['c_custkey'].astype('int64') == int(key_value)]
                else:
                    matches = df[df['c_custkey'] == key_value]
                
                if len(matches) > 0:
                    print(f"\n📁 {pf.name}: {len(matches)}件見つかりました")
                    print("-" * 80)
                    
                    # 各行の詳細を表示
                    for idx in range(len(matches)):
                        print(f"\n【行 {idx + 1}】")
                        
                        # 基本データ
                        row_data = matches.iloc[idx]
                        
                        # 主要列を表示
                        main_cols = ['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 
                                    'c_region', 'c_phone', 'c_mktsegment']
                        
                        print("基本データ:")
                        # まず利用可能な列を確認
                        available_cols = [col for col in main_cols if col in matches.columns]
                        if not available_cols:
                            print(f"  ⚠️ 基本列が見つかりません。利用可能な列: {list(matches.columns)}")
                        
                        for col in main_cols:
                            if col in matches.columns:
                                try:
                                    value = matches[col].iloc[idx]
                                    # bpchar型の場合は末尾の空白を表示
                                    if isinstance(value, str):
                                        print(f"  {col}: '{value}' (長さ: {len(value)})")
                                    else:
                                        print(f"  {col}: {value}")
                                except Exception as e:
                                    print(f"  {col}: エラー - {e}")
                        
                        # デバッグ情報があれば表示
                        debug_cols = ['_thread_id', '_row_position', '_thread_start_pos', '_thread_end_pos']
                        debug_found = False
                        
                        for col in debug_cols:
                            if col in matches.columns:
                                if not debug_found:
                                    print("\nデバッグ情報:")
                                    debug_found = True
                                try:
                                    value = matches[col].iloc[idx]
                                    print(f"  {col}: {value}")
                                except Exception as e:
                                    print(f"  {col}: エラー - {e}")
                        
                        # thread情報から処理範囲を計算
                        if all(col in matches.columns for col in ['_thread_id', '_thread_start_pos', '_thread_end_pos']):
                            thread_id = matches['_thread_id'].iloc[idx]
                            start_pos = matches['_thread_start_pos'].iloc[idx]
                            end_pos = matches['_thread_end_pos'].iloc[idx]
                            thread_range = end_pos - start_pos
                            print(f"\n  処理範囲: {thread_range} バイト")
                            print(f"  スレッド {thread_id} が処理した範囲: [{start_pos:,} - {end_pos:,}]")
                        
                        # row_positionから位置情報を表示
                        if '_row_position' in matches.columns:
                            row_pos = matches['_row_position'].iloc[idx]
                            print(f"\n  バイナリファイル内の位置: {row_pos:,} バイト目")
                            
                            # チャンク内での相対位置を計算（仮定：各チャンク8GB）
                            chunk_size = 8 * 1024**3
                            chunk_id = int(pf.name.split('_')[1])
                            global_pos = chunk_id * chunk_size + row_pos
                            print(f"  全体での推定位置: {global_pos:,} バイト目")
                    
                    total_found += len(matches)
                    
                    # 重複がある場合の分析
                    if len(matches) > 1:
                        print(f"\n⚠️  このファイル内で{len(matches)}個の重複が見つかりました！")
                        
                        if '_thread_id' in matches.columns:
                            thread_ids = matches['_thread_id'].unique().to_pandas()
                            print(f"  処理したスレッドID: {sorted(thread_ids)}")
                            
                            # 各スレッドごとの詳細
                            for tid in sorted(thread_ids):
                                tid_matches = matches[matches['_thread_id'] == tid]
                                print(f"\n  スレッド {tid}: {len(tid_matches)}行")
                                
                                if '_row_position' in tid_matches.columns:
                                    positions = tid_matches['_row_position'].to_pandas()
                                    for i, pos in enumerate(positions):
                                        print(f"    - 行{i+1}: position={pos:,}")
            
            else:
                print(f"⚠️  {pf.name}: c_custkey列が見つかりません")
                print(f"    利用可能な列: {list(df.columns)[:10]}...")
                
        except Exception as e:
            print(f"❌ {pf.name}: エラー - {e}")
    
    print(f"\n\n=== 検索結果サマリー ===")
    print(f"合計 {total_found} 件の c_custkey={key_value} が見つかりました")
    
    if total_found > 1:
        print(f"\n⚠️  重複データが検出されました！")
        print(f"この値は本来ユニークであるべきですが、{total_found}回出現しています。")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python search_parquet_by_key.py <c_custkey値>")
        print("例: python search_parquet_by_key.py 535")
        sys.exit(1)
    
    try:
        key_value = int(sys.argv[1])
    except ValueError:
        print(f"エラー: c_custkeyは整数である必要があります: {sys.argv[1]}")
        sys.exit(1)
    
    search_customer_in_parquet(key_value)


if __name__ == "__main__":
    main()