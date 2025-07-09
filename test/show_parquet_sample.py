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

def display_sample_data(df, n_rows=10, name=""):
    """サンプルデータの表示"""
    print(f"\n{'='*60}")
    print(f"📊 サンプルデータ{' - ' + name if name else ''}")
    print(f"{'='*60}")
    
    # データ型情報
    print("\n【カラム情報】")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # 先頭n行の表示
    print(f"\n【先頭{n_rows}行】")
    if isinstance(df, dask_cudf.DataFrame):
        # Daskの場合
        sample_df = df.head(n_rows)
        # pandasに変換して全列表示
        print(sample_df.to_pandas().to_string())
    else:
        # 通常のcuDFの場合
        # pandasに変換して全列表示
        print(df.head(n_rows).to_pandas().to_string())
    
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

def process_large_parquet_files():
    """メイン処理フロー"""
    print("🚀 cuDF Parquetファイル処理開始")
    print("="*60)
    
    # 1. GPU環境確認
    print("\n【GPU環境確認】")
    check_gpu_environment()
    
    # 2. outputディレクトリ内の全parquetファイルを取得
    output_dir = Path("output")
    if not output_dir.exists():
        print(f"\n✗ outputディレクトリが見つかりません: {output_dir}")
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
        display_sample_data(df, n_rows=5, name=f"ファイル{i} ({Path(file_path).name})")


if __name__ == "__main__":
    try:
        # メイン処理実行
        process_large_parquet_files()
        
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
