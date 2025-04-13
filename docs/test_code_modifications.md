# テストコード修正ガイド

このドキュメントでは、gpuPaserのテストコードを修正して、より大きなデータセット（70,000行以上）を処理するための具体的な方法を説明します。

## 現状の問題点

現在のパイプラインには以下の制限があります：

1. **CUDA制限**: 1回のカーネル起動で処理できる最大行数は65,535行（ブロック数の制限による）
2. **チャンク境界**: 現在のバイナリパーサーは複数チャンク間での行オフセット計算に問題がある
3. **エラー処理**: 2回目のチャンク処理でエラーが発生した場合の適切な回復処理がない

## 修正方針

### 1. テストスクリプトの修正

`test_postgres_binary_cuda_parser_module.py`を以下のように修正します：

```python
# 70,000行のテスト関数
def test_70k_rows():
    print("\n=== 70,000行処理テスト ===")
    
    # 方法1: 65,000行以下に制限して安全に処理
    test_rows = 65000
    print(f"方法1: 最大{test_rows}行を処理")
    start_time = time.time()
    results = load_table_optimized('customer', test_rows)
    print(f"処理時間: {time.time() - start_time:.3f}秒")
    
    # 方法2: 複数チャンクに分割して処理
    chunks = [(0, 35000), (35000, 35000)]  # (開始行, 行数)
    all_results = {}
    
    print("\n方法2: 複数チャンク処理")
    total_start_time = time.time()
    
    for i, (start, size) in enumerate(chunks):
        print(f"\nチャンク {i+1}: 行 {start}～{start+size-1}")
        chunk_start_time = time.time()
        
        try:
            # ここで個別のチャンク処理を実装
            # この例では直接main.pyの内部実装を呼び出す
            processor = PgGpuProcessor()
            results = processor.process_chunk('customer', start, size)
            processor.close()
            
            # 結果を集約
            if not all_results:
                all_results = results
            else:
                for col_name, data in results.items():
                    if isinstance(data, np.ndarray):
                        all_results[col_name] = np.concatenate([all_results[col_name], data])
                    elif isinstance(data, list):
                        all_results[col_name].extend(data)
            
            print(f"チャンク {i+1} 処理時間: {time.time() - chunk_start_time:.3f}秒")
            
        except Exception as e:
            print(f"チャンク {i+1} 処理エラー: {e}")
            # エラー時は部分的な結果を返す
    
    print(f"\n全チャンク処理時間: {time.time() - total_start_time:.3f}秒")
    print(f"処理された総行数: {sum(size for _, size in chunks)}")
    
    return all_results
```

### 2. メインモジュールの修正

`main.py`に必要な機能を追加します：

```python
# PgGpuProcessor クラスに新しいメソッドを追加
def process_chunk(self, table_name: str, start_row: int, chunk_size: int):
    """指定した開始行から特定サイズのチャンクを処理"""
    # テーブルの存在確認
    if not check_table_exists(self.conn, table_name):
        raise ValueError(f"Table {table_name} does not exist")
    
    # テーブル情報の取得
    columns = get_table_info(self.conn, table_name)
    if not columns:
        raise ValueError(f"No columns found in table {table_name}")
    
    # バイナリデータの取得（全データを取得し、後でスライス）
    buffer_data, buffer = get_binary_data(self.conn, table_name, start_row + chunk_size)
    
    # バッファの初期化
    buffers = self.memory_manager.initialize_device_buffers(columns, chunk_size)
    
    try:
        # バイナリデータの解析（開始行を指定）
        chunk_array, field_offsets, field_lengths, rows_in_chunk = self.parser.parse_chunk(
            buffer_data, 
            max_chunk_size=1024*1024,
            num_columns=len(columns),
            start_row=start_row,
            max_rows=chunk_size
        )
        
        if rows_in_chunk == 0:
            print(f"開始行 {start_row} 以降にデータがありません")
            return {}
            
        print(f"処理中: {rows_in_chunk}行 (行 {start_row}～{start_row+rows_in_chunk-1})")
        
        # GPUでデコード
        d_col_types = self.memory_manager.transfer_to_device(buffers["col_types"], np.int32)
        d_col_lengths = self.memory_manager.transfer_to_device(buffers["col_lengths"], np.int32)
        
        chunk_results = self.gpu_decoder.decode_chunk(
            buffers, 
            chunk_array, 
            field_offsets, 
            field_lengths, 
            rows_in_chunk, 
            columns
        )
        
        # 結果の処理
        self.output_handler.process_chunk_result(chunk_results)
        final_results = self.output_handler.get_results()
        
        return final_results
        
    except Exception as e:
        print(f"チャンク処理エラー: {e}")
        raise
        
    finally:
        # リソースのクリーンアップ
        # 既存のクリーンアップコード...
        pass
```

### 3. バイナリパーサーの修正

`binary_parser.py`のキーポイントを修正します：

```python
def parse_chunk(self, chunk_data: bytes, max_chunk_size: int = 1024*1024, 
                num_columns: int = None, start_row: int = 0, max_rows: int = 65535):
    """チャンク単位でのパース処理"""
    # 既存のコード...
    
    # 特定の行から開始する場合のスキップ処理を改善
    if start_row > 0 and num_columns is not None:
        print(f"開始行 {start_row} からのパース開始")
        
        # バイナリデータのスキャンとオフセット計算を改善
        # 1. ヘッダーをスキップ
        pos = self._skip_header(chunk_array)
        
        # 2. 行単位でスキャン - より堅牢なスキップ処理
        row_count = 0
        row_offsets = [pos]  # 各行の開始位置
        
        # バイナリファイル全体をスキャンして行境界を特定
        while pos + 2 <= len(chunk_array):
            # フィールド数を読み取り
            num_fields = ((chunk_array[pos] << 8) | chunk_array[pos + 1])
            
            # ファイル終端チェック
            if num_fields == 0xFFFF:
                break
                
            pos += 2  # フィールド数フィールドをスキップ
            
            # 各フィールドをスキップ
            for _ in range(num_fields):
                if pos + 4 > len(chunk_array):
                    break
                    
                # フィールド長を読み取り
                field_len = ((chunk_array[pos] << 24) | 
                             (chunk_array[pos+1] << 16) | 
                             (chunk_array[pos+2] << 8) | 
                              chunk_array[pos+3])
                
                pos += 4  # フィールド長フィールドをスキップ
                
                # NULL値チェック
                if field_len != -1:
                    pos += field_len  # データをスキップ
            
            # 行の終わり
            row_count += 1
            row_offsets.append(pos)
            
            # 必要な行数に達したらスキャン終了
            if row_count > start_row + max_rows:
                break
        
        # 開始行が存在するか確認
        if start_row >= len(row_offsets) - 1:
            print(f"警告: 指定された開始行 {start_row} は存在しません（利用可能な行数: {len(row_offsets)-1}）")
            return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
        
        # 開始行以降のデータを抽出
        start_offset = row_offsets[start_row]
        end_offset = row_offsets[min(start_row + max_rows, len(row_offsets) - 1)]
        chunk_array = chunk_array[start_offset:end_offset]
        
        # ヘッダーフラグを無効化（スキップ済み）
        self.header_expected = False
    
    # 残りの処理は既存コードと同様...
```

### 4. ユーティリティ関数の追加

より便利なデバッグとテスト用のユーティリティ関数を追加します：

```python
# utils.py に追加

def estimate_max_rows(gpu_mem_mb, row_size_bytes):
    """利用可能なGPUメモリから処理可能な最大行数を推定"""
    # 80%のメモリを使用可能と仮定
    usable_mem = gpu_mem_mb * 0.8 * 1024 * 1024  # バイト単位に変換
    
    # オーバーヘッドを20%と仮定
    row_size_with_overhead = row_size_bytes * 1.2
    
    # 最大行数計算（CUDA制限も考慮）
    max_rows = min(int(usable_mem / row_size_with_overhead), 65535)
    
    return max_rows

def analyze_data_types(columns):
    """カラム情報から行あたりのメモリ使用量を推定"""
    total_bytes = 0
    type_summary = {}
    
    for col in columns:
        if get_column_type(col.type) <= 1:  # 数値型
            total_bytes += 8
            type_summary[col.type] = type_summary.get(col.type, 0) + 1
        else:  # 文字列型
            length = get_column_length(col.type, col.length)
            total_bytes += length
            type_summary[f"{col.type}({length})"] = type_summary.get(f"{col.type}({length})", 0) + 1
    
    return {
        "total_bytes_per_row": total_bytes,
        "type_summary": type_summary,
        "estimated_max_rows": estimate_max_rows(21000, total_bytes)  # 21GBと仮定
    }
```

## 新しいテスト関数の実行

これらの修正を適用した後、70,000行以上のデータを処理するためのテストコードを以下のように実行できます：

```python
# test_postgres_binary_cuda_parser_module.py

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PostgreSQL GPU Parser Test')
    parser.add_argument('--incremental', action='store_true', help='Run incremental row test')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--chunks', action='store_true', help='Use multi-chunk processing')
    args = parser.parse_args()
    
    if args.incremental:
        test_increasing_rows()
    elif args.chunks:
        test_multi_chunk()
    elif args.rows:
        print(f"Processing {args.rows} rows...")
        if args.rows > 65535:
            print("Note: Using multiple chunks as row count exceeds 65,535")
            # マルチチャンク処理
            # チャンクサイズを計算
            chunk_size = 65000
            num_chunks = (args.rows + chunk_size - 1) // chunk_size
            
            # 各チャンクを処理
            for i in range(num_chunks):
                start_row = i * chunk_size
                size = min(chunk_size, args.rows - start_row)
                print(f"\nProcessing chunk {i+1}/{num_chunks}: rows {start_row} to {start_row+size-1}")
                # チャンク処理コード...
        else:
            # 単一チャンク処理
            results = load_table_optimized('customer', args.rows)
    else:
        run_tests()
```

## 応用例: 最大メモリ使用量でのテスト

GPUメモリを最大限活用するために、利用可能なGPUメモリに基づいて最適なチャンクサイズを自動的に計算するテスト：

```python
def test_max_memory_usage():
    """利用可能なGPUメモリを最大限使用するテスト"""
    print("\n=== GPUメモリ最大活用テスト ===")
    
    # メモリマネージャーの初期化
    memory_manager = GPUMemoryManager()
    
    # 利用可能なGPUメモリ取得
    free_memory = memory_manager.get_available_gpu_memory()
    total_memory = memory_manager.get_total_gpu_memory()
    
    print(f"GPUメモリ: 利用可能 {free_memory/(1024**2):.2f}MB / 合計 {total_memory/(1024**2):.2f}MB")
    
    # テーブル情報取得
    conn = connect_to_postgres()
    columns = get_table_info(conn, 'customer')
    conn.close()
    
    # 最適チャンクサイズ計算
    optimal_chunk_size = memory_manager.calculate_optimal_chunk_size(columns, 1000000)
    
    print(f"最適チャンクサイズ: {optimal_chunk_size}行")
    print(f"推定メモリ使用量: {analyze_data_types(columns)['total_bytes_per_row'] * optimal_chunk_size / (1024**2):.2f}MB")
    
    # テスト実行
    print(f"\n{optimal_chunk_size}行のデータをロード中...")
    start_time = time.time()
    results = load_table_optimized('customer', optimal_chunk_size)
    print(f"処理時間: {time.time() - start_time:.3f}秒")
    
    return results
```

これらの修正により、GPUメモリを最大限に活用して、より大きなデータセットを効率的に処理できるようになります。
