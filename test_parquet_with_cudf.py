#!/usr/bin/env python3
"""
生成されたParquetファイルをPyArrow（およびcuDF）で読み込み、型情報を検証するテストスクリプト
"""

import os
import sys
import glob
import argparse

def test_parquet_file(file_path):
    """Parquetファイルを読み込み、データ型とスキーマを表示
    
    まずPyArrowを試し、cuDFがインストールされている場合はcuDFでも検証
    """
    
    # ファイルの存在確認
    if not os.path.exists(file_path):
        print(f"エラー: ファイル {file_path} が見つかりません")
        return False
    
    # ファイルサイズの確認
    file_size = os.path.getsize(file_path)
    print(f"ファイルサイズ: {file_size / 1024:.2f} KB")
    
    if file_size == 0:
        print("エラー: ファイルサイズが0です")
        return False
    
    success = False  # 最終的な成功フラグ
    pa_success = False  # PyArrow成功フラグ
    
    # 1. PyArrowでテスト
    try:
        import pyarrow.parquet as pq
        import pandas as pd
        
        print(f"\n=== PyArrowでParquetファイルをテスト: {file_path} ===")
        
        # スキーマのみ読み取り
        schema = pq.read_schema(file_path)
        print("\n--- PyArrow スキーマ情報 ---")
        print(schema)
        
        # 全データ読み込み
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # 基本情報の表示
        num_rows = len(df)
        num_cols = len(df.columns)
        print(f"行数: {num_rows}, 列数: {num_cols}")
        
        # スキーマ情報の表示
        print("\n--- Pandas データ型情報 ---")
        for col_name, dtype in df.dtypes.items():
            print(f"{col_name}: {dtype}")
        
        # データサンプルの表示
        print("\n--- データサンプル (先頭5行) ---")
        print(df.head(5))
        
        # 各カラムの非NULL値カウントと一意値カウント
        print("\n--- カラム統計 ---")
        null_counts = df.isnull().sum()
        valid_counts = num_rows - null_counts
        
        for col_name in df.columns:
            try:
                unique_count = df[col_name].nunique()
                print(f"{col_name}: 有効値={valid_counts[col_name]}, NULL={null_counts[col_name]}, 一意値={unique_count}")
            except Exception as e:
                print(f"{col_name}: 有効値={valid_counts[col_name]}, NULL={null_counts[col_name]}, 一意値=計算エラー ({e})")
        
        print("\n--- データ型サマリー ---")
        type_counts = {}
        for dtype in df.dtypes:
            type_name = str(dtype)
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
        
        for type_name, count in type_counts.items():
            print(f"{type_name}: {count}カラム")
            
        success = True
    
    except ImportError:
        print("警告: PyArrowがインストールされていません。PyArrowをインストールしてください。")
    
    except Exception as e:
        print(f"エラー: PyArrowでのParquetファイル読み込み中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. cuDFでテスト（インストールされている場合）
    try:
        import cudf
        print(f"\n=== cuDFでParquetファイルをテスト: {file_path} ===")
        
        # ファイルの存在確認
        if not os.path.exists(file_path):
            print(f"エラー: ファイル {file_path} が見つかりません")
            return False
        
        # ファイルサイズの確認
        file_size = os.path.getsize(file_path)
        print(f"ファイルサイズ: {file_size / 1024:.2f} KB")
        
        if file_size == 0:
            print("エラー: ファイルサイズが0です")
            return False
        
        # cuDFでファイル読み込み
        df = cudf.read_parquet(file_path)
        
        # 基本情報の表示
        num_rows = len(df)
        num_cols = len(df.columns)
        print(f"行数: {num_rows}, 列数: {num_cols}")
        
        # スキーマ情報の表示
        print("\n--- スキーマ情報 ---")
        for col_name, dtype in df.dtypes.items():
            print(f"{col_name}: {dtype}")
        
        # データサンプルの表示
        print("\n--- データサンプル (先頭5行) ---")
        print(df.head(5))
        
        # 各カラムの非NULL値カウントと一意値カウント
        print("\n--- カラム統計 ---")
        null_counts = df.isnull().sum()
        valid_counts = num_rows - null_counts
        
        for col_name in df.columns:
            try:
                unique_count = df[col_name].nunique()
                print(f"{col_name}: 有効値={valid_counts[col_name]}, NULL={null_counts[col_name]}, 一意値={unique_count}")
            except Exception as e:
                print(f"{col_name}: 有効値={valid_counts[col_name]}, NULL={null_counts[col_name]}, 一意値=計算エラー ({e})")
        
        print("\n--- データ型サマリー ---")
        type_counts = {}
        for dtype in df.dtypes:
            type_name = str(dtype)
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
        
        for type_name, count in type_counts.items():
            print(f"{type_name}: {count}カラム")
        
        success = True
        return True
    
    except ImportError:
        print("エラー: cuDFがインストールされていません")
        # PyArrowでのテスト結果を返す
        return success
    
    except Exception as e:
        print(f"エラー: cuDFでのParquetファイル読み込み中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        # PyArrowでのテスト結果を返す
        return success

def test_directory(directory_path, pattern="*.parquet"):
    """ディレクトリ内の全Parquetファイルをテスト"""
    
    # ディレクトリ内のParquetファイルを検索
    file_pattern = os.path.join(directory_path, pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"警告: {file_pattern} に一致するファイルが見つかりません")
        return
    
    print(f"=== {len(files)}個のParquetファイルをテスト ===")
    
    # 各ファイルをテスト
    success_count = 0
    for file_path in files:
        result = test_parquet_file(file_path)
        if result:
            success_count += 1
    
    print(f"\n=== テスト結果サマリー ===")
    print(f"合計: {len(files)}ファイル")
    print(f"成功: {success_count}ファイル")
    print(f"失敗: {len(files) - success_count}ファイル")

def main():
    """メイン関数"""
    
    parser = argparse.ArgumentParser(description="cuDFでParquetファイルを読み込み検証するテストスクリプト")
    
    # 引数の追加
    parser.add_argument("--file", help="テストする単一のParquetファイルパス")
    parser.add_argument("--dir", help="テストするParquetファイルを含むディレクトリパス")
    parser.add_argument("--pattern", default="*.parquet", help="ファイル検索パターン (デフォルト: *.parquet)")
    
    args = parser.parse_args()
    
    # 引数の検証
    if args.file:
        test_parquet_file(args.file)
    elif args.dir:
        test_directory(args.dir, args.pattern)
    else:
        # デフォルトでlineorder_multigpu_outputディレクトリをテスト
        default_dir = "lineorder_multigpu_output"
        if os.path.exists(default_dir):
            print(f"デフォルトディレクトリをテスト: {default_dir}")
            test_directory(default_dir)
        else:
            parser.print_help()
            print(f"\nエラー: テストするファイルまたはディレクトリを指定してください")
            sys.exit(1)

if __name__ == "__main__":
    main()
