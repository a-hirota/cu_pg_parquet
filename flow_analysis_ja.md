# GPU PostgreSQL to Parquet Parser - 実装詳細解析

## 概要
PostgreSQLのデータをGPU上で高速にParquetファイルに変換するシステムの実装詳細。Producer-Consumerパターンを採用し、データ抽出とGPU処理を並列化。

## アーキテクチャ概要
1. **Rustバイナリ** - PostgreSQLからCOPY BINARYでデータ抽出
2. **GPUパーサー** - PostgreSQLバイナリ形式をGPU上でパース
3. **cuDFコンバーター** - パース済みデータからcuDF DataFrameを構築
4. **Parquetライター** - GPU加速でParquetファイル書き込み

## 実行フロー

### 1. エントリポイント: `cu_pg_parquet.py`
```python
main() -> benchmark_main()
```
- コマンドライン引数をパース
- 環境変数を設定（RUST_PARALLEL_CONNECTIONS, TABLE_NAME）
- 既存のparquetファイルをクリーンアップ
- `benchmark_rust_gpu_direct.main()`を呼び出し

### 2. ベンチマークオーケストレーター: `benchmark_rust_gpu_direct.py`

#### 2.1 メイン関数
```python
def main(total_chunks: int = 16, table_name: str = "lineorder", 
         test_mode: bool = False, test_duplicate_keys: bool = False):
    # RMMメモリプール設定（24GB）
    setup_rmm_pool()
    
    # PostgreSQLメタデータ取得
    columns = get_postgresql_metadata(table_name)
    
    # GPU warmup実行（初回JITコンパイル対策）
    if warmup:
        gpu_warmup(columns)
    
    # 並列パイプライン実行
    df = run_parallel_pipeline(columns, total_chunks, table_name, test_mode)
```

#### 2.2 メタデータ取得
```python
def get_postgresql_metadata(table_name: str) -> List[ColumnMeta]:
    conn = psycopg2.connect(DSN)
    return fetch_column_meta(conn, f"SELECT * FROM {table_name} LIMIT 0")
```

PostgreSQL OIDからArrow型へのマッピング:
```python
PG_OID_TO_ARROW = {
    20: (pa.int64(), 8),       # bigint
    21: (pa.int16(), 2),       # smallint  
    23: (pa.int32(), 4),       # integer
    700: (pa.float32(), 4),    # real
    701: (pa.float64(), 8),    # double precision
    1700: (pa.decimal128(38, 4), 16),  # numeric
    16: (pa.bool_(), 1),       # boolean
    25: (pa.string(), None),   # text
    1043: (pa.string(), None), # varchar
    1082: (pa.date32(), 4),    # date
    1114: (pa.timestamp('us'), 8),  # timestamp
}
```

#### 2.3 並列パイプライン実行
```python
def run_parallel_pipeline(columns, total_chunks, table_name, test_mode):
    chunk_queue = queue.Queue()
    stats_queue = queue.Queue()
    
    # Producerスレッド起動
    producer = threading.Thread(
        target=rust_producer,
        args=(chunk_queue, total_chunks, stats_queue, table_name)
    )
    
    # Consumerスレッド起動（CPU数の70%）
    consumer_count = max(1, int(psutil.cpu_count() * 0.7))
    consumers = []
    for i in range(consumer_count):
        consumer = threading.Thread(
            target=gpu_consumer,
            args=(chunk_queue, columns, i, stats_queue, total_chunks, 
                  table_name, test_mode)
        )
        consumers.append(consumer)
```

### 3. Producerスレッド: Rustデータ抽出

```python
def rust_producer(chunk_queue, total_chunks, stats_queue, table_name):
    """Rustバイナリを実行してPostgreSQLからデータ抽出"""
    for chunk_id in range(total_chunks):
        # 環境変数設定
        env = os.environ.copy()
        env['CHUNK_ID'] = str(chunk_id)
        env['TOTAL_CHUNKS'] = str(total_chunks)
        env['TABLE_NAME'] = table_name
        
        # Rustバイナリ実行
        cmd = ['./rust_bench_optimized/target/release/pg_fast_copy_single_chunk']
        result = subprocess.run(cmd, env=env, capture_output=True)
        
        # チャンク情報をキューに追加
        chunk_file = f"/dev/shm/{table_name}_chunk_{chunk_id}.bin"
        chunk_queue.put((chunk_id, chunk_file, table_name))
```

Rustバイナリの処理:
- PostgreSQLに並列接続（デフォルト16接続）
- `COPY table TO STDOUT (FORMAT BINARY)`実行
- バイナリデータを`/dev/shm/{table}_chunk_{id}.bin`に書き込み

### 4. Consumerスレッド: GPU処理

```python
def gpu_consumer(chunk_queue, columns, consumer_id, stats_queue, 
                total_chunks, table_name, test_mode):
    """チャンクをGPUで処理してParquetに変換"""
    while True:
        chunk_id, chunk_file, table_name = chunk_queue.get()
        
        # kvikioでGPU直接転送
        file_size = os.path.getsize(chunk_file)
        gpu_buffer = rmm.DeviceBuffer(size=file_size)
        
        with kvikio.CuFile(chunk_file, "rb") as f:
            gpu_array = cp.asarray(gpu_buffer).view(dtype=cp.uint8)
            bytes_read = f.read(gpu_array)
        
        # GPU処理実行
        output_path = f"output/{table_name}_chunk_{chunk_id}.parquet"
        row_count, timings = postgresql_to_cudf_parquet_direct(
            gpu_buffer, columns, output_path, 
            debug=False, chunk_size=None,
            rows_per_thread=int(os.environ.get('GPUPGPARSER_ROWS_PER_THREAD', '32'))
        )
```

### 5. GPUパーシングパイプライン: `src/postgres_to_cudf.py`

#### 5.1 メイン処理関数
```python
def postgresql_to_cudf_parquet_direct(
    binary_data: rmm.DeviceBuffer,
    columns: List[ColumnDefinition],
    output_parquet_path: str,
    debug: bool = False,
    chunk_size: Optional[int] = None,
    rows_per_thread: int = 32
) -> Tuple[int, Dict[str, float]]:
    """PostgreSQLバイナリデータをGPU上で直接パース"""
    
    # バイナリヘッダー検証（19バイト）
    header = binary_data.copy_to_host()[:19]
    if header[:11] != b'PGCOPY\n\xff\r\n\x00':
        raise ValueError("Invalid PostgreSQL binary format")
    
    # CUDAカーネル実行
    result = parse_rows_and_fields_lite(
        binary_data, columns, header_size=19,
        debug=debug, rows_per_thread=rows_per_thread
    )
    
    # カラム抽出とDataFrame構築
    df = extract_columns_and_create_dataframe(
        binary_data, result, columns
    )
    
    # Parquet書き込み
    df.to_parquet(output_parquet_path, compression='snappy')
```

### 6. CUDAカーネル実装: `src/cuda_kernels/postgres_binary_parser.py`

#### 6.1 行・フィールド検出カーネル
```python
@cuda.jit
def parse_rows_and_fields_lite(
    data, data_size, chunk_offset, field_count,
    row_offsets, row_field_indices, max_rows,
    row_count_result, current_row_global, rows_per_thread
):
    """PostgreSQLバイナリから行とフィールドを高速検出"""
    
    # グリッドストライドループ
    thread_id = cuda.grid(1)
    grid_size = cuda.gridsize(1)
    
    # 各スレッドが処理する開始位置を計算
    stride_size = (data_size - chunk_offset) // grid_size
    start_pos = chunk_offset + thread_id * stride_size
    
    # 行検出ループ
    pos = start_pos
    while pos < end_pos:
        # 行ヘッダー検証（2バイトのフィールド数）
        if pos + 2 <= data_size:
            field_count_in_row = (data[pos] << 8) | data[pos + 1]
            
            if field_count_in_row == field_count:
                # 有効な行を発見
                row_idx = cuda.atomic.add(row_count_result, 0, 1)
                if row_idx < max_rows:
                    row_offsets[row_idx] = pos
                    
                    # フィールドオフセット記録
                    field_pos = pos + 2
                    for field_idx in range(field_count):
                        idx = row_idx * field_count + field_idx
                        row_field_indices[idx] = field_pos
                        
                        # フィールド長読み取り（4バイト）
                        field_len = read_int32_big_endian(data, field_pos)
                        field_pos += 4
                        if field_len > 0:
                            field_pos += field_len
```

最適化ポイント:
- **スレッドストライド**: 各スレッドが`rows_per_thread`行を処理
- **アトミック操作**: 行カウントの競合回避
- **共有メモリ最小化**: <1KBで大規模データ対応
- **単一パス**: 行検出とフィールド抽出を同時実行

#### 6.2 固定長カラム抽出
```python
@cuda.jit
def extract_fixed_columns_optimized(
    data, row_offsets, row_field_indices,
    total_rows, column_info, output_buffers,
    null_masks, field_count, chunk_offset
):
    """固定長カラム（数値、日付等）の高速抽出"""
    
    row_idx = cuda.grid(1)
    if row_idx >= total_rows:
        return
    
    # 各カラムを処理
    for col_idx in range(column_info.shape[0]):
        field_idx = column_info[col_idx, 0]  # フィールドインデックス
        oid = column_info[col_idx, 1]        # PostgreSQL OID
        col_size = column_info[col_idx, 2]   # カラムサイズ
        
        # フィールド位置計算
        field_offset = row_field_indices[row_idx * field_count + field_idx]
        field_length = read_int32_big_endian(data, field_offset)
        
        if field_length == -1:  # NULL値
            null_masks[col_idx * ((total_rows + 7) // 8) + row_idx // 8] |= (1 << (row_idx % 8))
        else:
            # データ抽出とエンディアン変換
            data_offset = field_offset + 4
            if oid == 23:  # INT32
                value = read_int32_big_endian(data, data_offset)
                output_buffers[col_idx][row_idx] = value
            elif oid == 20:  # INT64
                value = read_int64_big_endian(data, data_offset)
                output_buffers[col_idx][row_idx] = value
            # ... 他の型も同様
```

#### 6.3 文字列カラム抽出
```python
@cuda.jit
def copy_string_data_direct(
    data, str_src_offsets, str_lengths,
    str_dest_offsets, str_count, chars_buffer,
    chunk_offset
):
    """文字列データの直接コピー"""
    
    idx = cuda.grid(1)
    if idx >= str_count:
        return
    
    # ソースとデスティネーションのオフセット
    src_start = str_src_offsets[idx] + chunk_offset
    src_length = str_lengths[idx]
    dest_start = str_dest_offsets[idx]
    
    # バイト単位でコピー（シンプルだが効率的）
    for i in range(src_length):
        chars_buffer[dest_start + i] = data[src_start + i]
```

### 7. cuDF DataFrame構築

```python
def create_cudf_columns(column_buffers, null_masks, columns, row_count):
    """GPU上のバッファからcuDFカラムを構築"""
    
    cudf_columns = {}
    for i, col in enumerate(columns):
        if col.data_type == pa.string():
            # 文字列カラム
            offset_buffer = column_buffers[f"{col.name}_offsets"]
            data_buffer = column_buffers[f"{col.name}_data"]
            cudf_columns[col.name] = cudf.core.column.string.StringColumn(
                children=(offset_buffer, data_buffer)
            )
        else:
            # 固定長カラム
            data_buffer = column_buffers[col.name]
            null_mask = null_masks[i] if col.nullable else None
            cudf_columns[col.name] = cudf.core.column.as_column(
                data_buffer, dtype=col.numpy_dtype, mask=null_mask
            )
    
    return cudf.DataFrame(cudf_columns)
```

### 8. パフォーマンス最適化

#### 8.1 環境変数による制御
```bash
# スレッドあたり処理行数（デフォルト: 32）
export GPUPGPARSER_ROWS_PER_THREAD=32

# SMあたりブロック数（デフォルト: 4）
export GPUPGPARSER_BLOCKS_PER_SM=4

# ゼロコピー強制（デフォルト: 1）
export GPUPGPARSER_FORCE_ZERO_COPY=1
```

#### 8.2 最適化実装
```python
# 最適なスレッド数計算
device = cuda.get_current_device()
sm_count = device.MULTIPROCESSOR_COUNT
threads_per_block = 256
blocks_per_sm = int(os.environ.get('GPUPGPARSER_BLOCKS_PER_SM', '4'))
optimal_threads = sm_count * blocks_per_sm * threads_per_block

# グリッドサイズ計算
total_threads_needed = (row_count + rows_per_thread - 1) // rows_per_thread
blocks = (total_threads_needed + threads_per_block - 1) // threads_per_block
```

## エラーハンドリング

```python
# カーネル内エラー検出
if field_count_in_row < 0 or field_count_in_row > 100:
    # 不正な行をスキップ
    continue

# バッファオーバーフロー防止
if row_idx >= max_rows:
    # 警告を記録して停止
    if row_idx == max_rows:
        printf("Warning: Row limit reached\n")
    return
```

## デバッグ・プロファイリング

```bash
# 詳細ログ有効化
export GPUPGPARSER_DEBUG=1

# CUDAエラー即時検出
export CUDA_LAUNCH_BLOCKING=1

# プロファイリング実行
nsys profile python cu_pg_parquet.py --table lineorder
```

## 処理時間の内訳（典型例）
- kvikio読み込み: 0.5-0.7秒
- GPUパース: 0.2-0.3秒（最適化後）
- Parquet書き込み: 1.0-1.7秒（現在のボトルネック）