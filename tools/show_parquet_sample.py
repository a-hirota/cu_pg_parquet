#!/usr/bin/env python3
"""
大規模Parquetファイルの読み込みと表示サンプル
cuDFを使用したGPU高速処理版
"""

import cudf
import dask_cudf
import time
import sys
from pathlib import Path
import gc
import pandas as pd
import argparse

# pandas表示設定を全列表示に変更
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def check_gpu_environment():
    """GPU環境の確認"""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✓ GPU数: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # nvmlDeviceGetNameの戻り値の型をチェック
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"  GPU {i}: {name}")
            print(f"    メモリ: {mem_info.total / 1024**3:.1f} GB (使用中: {mem_info.used / 1024**3:.1f} GB)")
    except Exception as e:
        print(f"⚠ GPU情報取得エラー: {e}")
        print("  cuDFは動作しますが、メモリ情報は表示できません")

def read_parquet_files(file_paths, use_dask=False):
    """
    Parquetファイルの読み込み
    
    Args:
        file_paths: 読み込むファイルパスのリスト
        use_dask: 大規模データ用にDask-cuDFを使用するか
    
    Returns:
        読み込んだデータフレームのリスト
    """
    dfs = []
    
    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"⚠ ファイルが見つかりません: {file_path}")
            continue
            
        print(f"\n📁 読み込み中: {file_path}")
        start_time = time.time()
        
        try:
            if use_dask:
                # Dask-cuDFを使用（超大規模データ用）
                df = dask_cudf.read_parquet(file_path)
                # 必要に応じて計算を実行
                df = df.persist()
            else:
                # 通常のcuDF読み込み
                df = cudf.read_parquet(file_path)
            
            elapsed = time.time() - start_time
            print(f"  ✓ 読み込み完了 ({elapsed:.2f}秒)")
            
            # 基本情報の表示
            if use_dask:
                print(f"  パーティション数: {df.npartitions}")
                # Daskの場合は.compute()で実際の値を取得
                shape = (len(df), len(df.columns))
            else:
                shape = df.shape
            
            print(f"  データサイズ: {shape[0]:,} 行 × {shape[1]} 列")
            
            # メモリ使用量の推定
            if not use_dask:
                memory_usage = df.memory_usage(deep=True).sum() / 1024**3
                print(f"  GPUメモリ使用量: {memory_usage:.2f} GB")
            
            dfs.append(df)
            
        except Exception as e:
            print(f"  ✗ エラー: {e}")
            continue
    
    return dfs

def check_corrupted_string(s):
    """文字列が破損しているかチェック"""
    if not isinstance(s, str):
        return False
    
    # 制御文字（改行、タブ以外）や異常な文字をチェック
    control_chars = 0
    for c in s:
        if ord(c) < 32 and c not in '\n\t\r':
            control_chars += 1
    
    # 制御文字が多い、または異常に長い場合は破損と判定
    return control_chars > 2 or len(s) > 100

def display_sample_data(df, n_rows=10, name="", filter_column=None, filter_value=None, file_path=None, thread_id=None):
    """サンプルデータの表示"""
    print(f"\n{'='*60}")
    print(f"📊 サンプルデータ{' - ' + name if name else ''}")
    print(f"{'='*60}")
    
    # 旧形式のthread_idサポート（互換性のため）
    if thread_id is not None and filter_column is None:
        filter_column = '_thread_id'
        filter_value = str(thread_id)
    
    # カラムでフィルタリング
    if filter_column is not None and filter_value is not None:
        if filter_column in df.columns:
            print(f"\n【{filter_column} = {filter_value} でフィルタリング】")
            
            # データ型に応じて値を変換
            dtype_str = str(df[filter_column].dtype)
            original_filter_value = filter_value
            
            try:
                # 数値型への変換を試みる
                if dtype_str in ['int8', 'int16', 'int32', 'int64']:
                    filter_value = int(filter_value)
                elif dtype_str in ['float32', 'float64']:
                    filter_value = float(filter_value)
                elif 'decimal' in dtype_str.lower():
                    # Decimal型の場合はintに変換（cuDFのdecimal比較の互換性のため）
                    filter_value = int(filter_value)
                elif dtype_str == 'object' or 'string' in dtype_str.lower():
                    # 文字列型の場合は文字列のまま
                    filter_value = str(filter_value)
                else:
                    # その他の型は可能な限り元の値を使用
                    pass
            except (ValueError, TypeError) as e:
                print(f"  警告: 値の変換に失敗しました ({original_filter_value} -> {dtype_str}): {e}")
                # 変換に失敗した場合は元の値を使用
            
            # cuDFでのフィルタリング
            mask = df[filter_column] == filter_value
            filtered_df = df[mask]
            
            if len(filtered_df) == 0:
                print(f"  ⚠️ {filter_column} = {filter_value} のレコードが見つかりません")
                # 存在する値の範囲を表示
                if not isinstance(df, dask_cudf.DataFrame):
                    try:
                        unique_values = df[filter_column].unique().to_pandas()
                        if len(unique_values) <= 20:
                            print(f"  存在する値: {sorted(unique_values)}")
                        else:
                            # 数値型の場合は最小値と最大値を表示
                            if str(df[filter_column].dtype) in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
                                min_val = df[filter_column].min()
                                max_val = df[filter_column].max()
                                # cuDFのScalarを処理
                                if hasattr(min_val, 'compute'):
                                    min_val = min_val.compute()
                                if hasattr(max_val, 'compute'):
                                    max_val = max_val.compute()
                                print(f"  値の範囲: {min_val} 〜 {max_val}")
                            print(f"  ユニークな値の数: {len(unique_values):,}")
                    except:
                        pass
                return
            else:
                print(f"  ✓ {len(filtered_df):,} 件のレコードが見つかりました")
                df = filtered_df
        else:
            print(f"  ⚠️ '{filter_column}' カラムが存在しません")
            print(f"  利用可能なカラム: {', '.join(df.columns)}")
            return
    
    # データ型情報
    print("\n【カラム情報】")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # 表示する行数の調整（フィルタリング時は全件表示）
    if filter_column is not None and filter_value is not None:
        display_rows = min(len(df), 100)  # フィルタリング時は最大100行まで
        if display_rows < len(df):
            print(f"\n【{filter_column} = {filter_value} の最初の {display_rows} 行（全 {len(df)} 行中）】")
        else:
            print(f"\n【{filter_column} = {filter_value} の全 {display_rows} 行】")
    else:
        display_rows = min(n_rows, len(df))
        print(f"\n【先頭{display_rows}行】")
    
    # データの表示
    try:
        # フィルタリング時の詳細表示
        if filter_column is not None and filter_value is not None and len(df) > 0:
            # cuDFからPandasに変換して表示
            if isinstance(df, dask_cudf.DataFrame):
                display_df = df.head(display_rows).to_pandas()
            else:
                display_df = df.head(display_rows).to_pandas()
            
            # decimal128型を文字列に変換
            for col in display_df.columns:
                if 'decimal' in str(df[col].dtype).lower():
                    display_df[col] = display_df[col].astype(str)
            
            # 詳細表示モード
            for idx, row in display_df.iterrows():
                print(f"\n  --- レコード {idx + 1} ---")
                
                field_shift_detected = False
                
                for col_name in display_df.columns:
                    value = row[col_name]
                    
                    # 値を安全に表示
                    if pd.isna(value):
                        display_value = "NULL"
                    else:
                        display_value = str(value)
                    
                    # customerテーブルの場合のフィールドシフト検出
                    if 'customer' in str(file_path).lower() if file_path else False:
                        if col_name == 'c_custkey' and str(value) == '0':
                            field_shift_detected = True
                            print(f"  ⚠️ {col_name}: {display_value} [フィールドシフトの可能性]")
                        elif col_name == 'c_name' and not str(value).startswith('Customer#'):
                            print(f"  ⚠️ {col_name}: {display_value} [c_addressの値の可能性]")
                        else:
                            print(f"  {col_name}: {display_value}")
                    else:
                        print(f"  {col_name}: {display_value}")
                
                if field_shift_detected:
                    print(f"\n  🔴 フィールドシフトが検出されました（c_custkeyから順に1つずつ前にシフト）")
        else:
            # 通常の表示（フィルタ指定なしの場合）
            if isinstance(df, dask_cudf.DataFrame):
                sample_df = df.head(display_rows).to_pandas()
            else:
                sample_df = df.head(display_rows).to_pandas()
            
            # decimal128型を文字列に変換
            for col in sample_df.columns:
                if 'decimal' in str(df[col].dtype).lower():
                    sample_df[col] = sample_df[col].astype(str)
            
            print(sample_df.to_string())
                
    except Exception as e:
        print(f"\n  ❌ データ表示エラー: {type(e).__name__}")
        print(f"     {str(e)[:200]}...")
        
        # エラーが発生した場合は、少なくともメタ情報を表示
        try:
            print("\n  【行数と基本情報のみ表示】")
            print(f"  総行数: {len(df)}")
            if '_thread_id' in df.columns:
                thread_counts = df['_thread_id'].value_counts().to_pandas()
                print(f"  Thread別行数:")
                for tid, count in thread_counts.items():
                    print(f"    Thread {tid}: {count}行")
        except:
            print("  基本情報も取得できません")
    
    # 基本統計量（数値列のみ）
    print("\n【基本統計量】")
    try:
        # cuDFのselect_dtypesは'number'ではなく具体的な型を指定する必要がある
        numeric_types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
        numeric_cols = []
        
        for col in df.columns:
            if str(df[col].dtype) in numeric_types:
                numeric_cols.append(col)
        
        if len(numeric_cols) > 0:
            if isinstance(df, dask_cudf.DataFrame):
                stats = df[numeric_cols].describe().compute()
                print(stats.to_pandas().to_string())
            else:
                stats = df[numeric_cols].describe()
                print(stats.to_pandas().to_string())
        else:
            print("  数値列がありません")
    except Exception as e:
        print(f"  統計量計算エラー: {e}")

def process_large_parquet_files(filter_column=None, filter_value=None, target_dir=None):
    """メイン処理フロー"""
    print("🚀 cuDF Parquetファイル処理開始")
    print("="*60)
    
    if filter_column is not None and filter_value is not None:
        print(f"\n🔍 {filter_column} = {filter_value} を検索します")
    
    # 1. GPU環境確認
    print("\n【GPU環境確認】")
    check_gpu_environment()
    
    # 2. ディレクトリ内の全parquetファイルを取得
    if target_dir:
        output_dir = Path(target_dir)
    else:
        # デフォルトは現在のディレクトリまたはoutputディレクトリ
        if Path("output").exists():
            output_dir = Path("output")
        else:
            output_dir = Path(".")
    
    if not output_dir.exists():
        print(f"\n✗ ディレクトリが見つかりません: {output_dir}")
        return
        
    file_paths = sorted(output_dir.glob("*.parquet"))
    
    if not file_paths:
        print(f"\n✗ {output_dir}ディレクトリ内にparquetファイルが見つかりません")
        return
        
    print(f"\n【検出されたファイル】")
    print(f"ディレクトリ: {output_dir}")
    print(f"ファイル数: {len(file_paths)}")
    for file_path in file_paths:
        print(f"  - {file_path.name}")
    
    # 3. ファイルサイズの確認（大規模データ判定）
    total_size = 0
    for file_path in file_paths:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / 1024**3
            print(f"\n📁 {file_path}: {size:.2f} GB")
            total_size += size
    
    # 4. 読み込み方法の選択（10GB以上ならDask使用を推奨）
    use_dask = total_size > 10
    if use_dask:
        print(f"\n⚡ 大規模データ検出 (合計 {total_size:.2f} GB)")
        print("  → Dask-cuDFを使用して分散処理します")
    
    # 5. ファイル読み込み
    print("\n【ファイル読み込み】")
    dfs = read_parquet_files(file_paths, use_dask=use_dask)
    
    if not dfs:
        print("\n✗ 読み込み可能なファイルがありませんでした")
        return
    
    # 6. 各ファイルのサンプル表示
    for i, (df, file_path) in enumerate(zip(dfs, file_paths)):
        display_sample_data(df, n_rows=5, name=f"ファイル{i} ({Path(file_path).name})", 
                          filter_column=filter_column, filter_value=filter_value, file_path=file_path)


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description='Parquetファイルの読み込みと表示（cuDF使用）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # 通常の表示
  python show_parquet_sample.py
  
  # 特定のthread_idでフィルタリング（旧形式、互換性のため残す）
  python show_parquet_sample.py --thread_id 1852295
  
  # 任意のカラムでフィルタリング
  python show_parquet_sample.py --filter _thread_id=1852295
  python show_parquet_sample.py --filter c_custkey=3045312
  python show_parquet_sample.py --filter c_name="Customer#003045312"
  python show_parquet_sample.py --filter c_nation="JAPAN"
  
  # 特定のディレクトリを指定
  python show_parquet_sample.py --dir /path/to/parquet/files
  
  # 組み合わせ
  python show_parquet_sample.py --filter c_region="ASIA" --dir .
        '''
    )
    
    parser.add_argument(
        '--thread_id',
        type=int,
        help='フィルタリングするthread_id（非推奨: --filter _thread_id=値 を使用してください）'
    )
    
    parser.add_argument(
        '--filter',
        type=str,
        help='フィルタリング条件（形式: カラム名=値）'
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        help='Parquetファイルが格納されているディレクトリ（デフォルト: outputまたは現在のディレクトリ）'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # コマンドライン引数のパース
        args = parse_args()
        
        # フィルタリング条件の解析
        filter_column = None
        filter_value = None
        
        if args.filter:
            # --filter カラム名=値 の形式をパース
            if '=' in args.filter:
                filter_column, filter_value = args.filter.split('=', 1)
                # 引用符を除去
                filter_value = filter_value.strip('"\'')
            else:
                print(f"✗ フィルタ形式が正しくありません: {args.filter}")
                print("  正しい形式: --filter カラム名=値")
                sys.exit(1)
        elif args.thread_id is not None:
            # 旧形式の --thread_id をサポート（互換性のため）
            filter_column = '_thread_id'
            filter_value = str(args.thread_id)
        
        # メイン処理実行
        process_large_parquet_files(
            filter_column=filter_column,
            filter_value=filter_value,
            target_dir=args.dir
        )
        
    except ImportError as e:
        print(f"\n✗ インポートエラー: {e}")
        print("\n【インストール方法】")
        print("conda install -c rapidsai -c conda-forge -c nvidia cudf")
        print("または")
        print("pip install cudf-cu11  # CUDA 11.x向け")
        print("pip install cudf-cu12  # CUDA 12.x向け")
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
