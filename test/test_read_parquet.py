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

def check_gpu_environment():
    """GPU環境の確認"""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✓ GPU数: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
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
        print(df.head(n_rows))
    else:
        # 通常のcuDFの場合
        print(df.head(n_rows))
    
    # 基本統計量（数値列のみ）
    print("\n【基本統計量】")
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            if isinstance(df, dask_cudf.DataFrame):
                stats = df[numeric_cols].describe().compute()
            else:
                stats = df[numeric_cols].describe()
            print(stats)
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
    
    # 2. ファイルパスの設定
    file_paths = [
        "output/customer_chunk_0_queue.parquet",
        "output/customer_chunk_1_queue.parquet"
    ]
    
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
    
    # 7. 並列処理での結合例（必要に応じて）
    if len(dfs) == 2:
        print("\n【データ結合例】")
        try:
            # 共通カラムの確認
            common_cols = set(dfs[0].columns) & set(dfs[1].columns)
            print(f"共通カラム: {common_cols}")
            
            # 垂直結合（行方向）の例
            if use_dask:
                combined_df = dask_cudf.concat(dfs, axis=0)
                print(f"\n結合後のサイズ: 約{len(dfs[0]) + len(dfs[1]):,} 行")
            else:
                combined_df = cudf.concat(dfs, axis=0)
                print(f"\n結合後のサイズ: {combined_df.shape[0]:,} 行 × {combined_df.shape[1]} 列")
                
        except Exception as e:
            print(f"結合エラー: {e}")
    
    # 8. メモリクリーンアップ
    print("\n【クリーンアップ】")
    del dfs
    gc.collect()
    cudf._lib.nvtx.nvtx_range_pop()  # GPUメモリ解放
    print("✓ 処理完了")

def advanced_operations_example():
    """高度な操作の例"""
    print("\n" + "="*60)
    print("📈 高度な操作例（参考）")
    print("="*60)
    
    example_code = """
# 1. 条件フィルタリング（GPU高速処理）
filtered_df = df[df['column_name'] > threshold]

# 2. グループ集計
grouped = df.groupby('category').agg({
    'value': ['sum', 'mean', 'count'],
    'amount': 'sum'
})

# 3. 並列ソート
sorted_df = df.sort_values(['col1', 'col2'], ascending=[True, False])

# 4. カスタム関数の適用（GPU最適化）
df['new_col'] = df.apply_rows(custom_gpu_function, 
                              incols=['col1', 'col2'],
                              outcols={'new_col': 'float32'})

# 5. 大規模JOIN（ハッシュJOIN on GPU）
merged = df1.merge(df2, on='key', how='inner')

# 6. CPU/GPU間のデータ転送
pandas_df = cudf_df.to_pandas()  # GPU → CPU
cudf_df = cudf.from_pandas(pandas_df)  # CPU → GPU
"""
    print(example_code)

if __name__ == "__main__":
    try:
        # メイン処理実行
        process_large_parquet_files()
        
        # 高度な操作例の表示
        advanced_operations_example()
        
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
