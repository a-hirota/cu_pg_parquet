# GPU Parser デバッグツール

## 既存ツール

### 1. show_parquet_sample.py
**用途**: Parquetファイルの内容確認とサンプリング

```bash
# 基本的な使用方法
python tools/show_parquet_sample.py output.parquet

# オプション
--rows N          # 表示行数（デフォルト: 10）
--columns col1,col2  # 特定カラムのみ表示
--stats           # 統計情報表示
--schema          # スキーマ情報表示
--sort column     # ソート表示
--gap-detection   # ギャップ検出（連番チェック）
```

### 2. compare_tables.py
**用途**: PostgreSQLテーブルとParquetファイルの比較

```bash
# データ整合性チェック
python tools/compare_tables.py \
    --pg-table lineorder \
    --parquet-file output/lineorder.parquet \
    --sample-size 1000

# 全件比較
python tools/compare_tables.py \
    --pg-table customer \
    --parquet-file output/customer.parquet \
    --full-scan
```

### 3. profile_gpu.py
**用途**: GPU性能プロファイリング

```bash
# CUDAカーネルプロファイル
python tools/profile_gpu.py \
    --kernel parse_rows_and_fields_lite \
    --iterations 100

# メモリ帯域幅測定
python tools/profile_gpu.py \
    --measure bandwidth \
    --data-size 500MB
```

## 推奨追加ツール

### 1. binary_inspector.py
**用途**: PostgreSQL COPY BINARYフォーマットの可視化

```python
#!/usr/bin/env python3
"""PostgreSQL バイナリフォーマット検査ツール"""

import struct
import sys
from typing import BinaryIO

def inspect_binary_file(filename: str, max_rows: int = 10):
    """バイナリファイルの構造を可視化"""
    with open(filename, 'rb') as f:
        # ヘッダー解析
        signature = f.read(11)  # 'PGCOPY\n\377\r\n\0'
        flags = struct.unpack('>I', f.read(4))[0]
        extension_length = struct.unpack('>I', f.read(4))[0]

        print(f"Signature: {signature}")
        print(f"Flags: {flags:032b}")
        print(f"Extension Length: {extension_length}")

        # 行データ解析
        for row_num in range(max_rows):
            field_count = struct.unpack('>h', f.read(2))[0]
            if field_count == -1:  # EOF marker
                break

            print(f"\nRow {row_num}: {field_count} fields")
            for field_num in range(field_count):
                field_length = struct.unpack('>i', f.read(4))[0]
                if field_length == -1:
                    print(f"  Field {field_num}: NULL")
                else:
                    data = f.read(field_length)
                    print(f"  Field {field_num}: {field_length} bytes = {data[:20]}...")

if __name__ == "__main__":
    inspect_binary_file(sys.argv[1])
```

### 2. memory_trace.py
**用途**: GPU/CPUメモリ使用量の追跡

```python
#!/usr/bin/env python3
"""メモリ使用量追跡ツール"""

import cupy
import psutil
import time
from contextlib import contextmanager

@contextmanager
def memory_trace(label: str):
    """メモリ使用量を追跡するコンテキストマネージャ"""
    # 開始時のメモリ
    cpu_start = psutil.Process().memory_info().rss / 1024 / 1024
    gpu_start = cupy.cuda.MemoryPool().used_bytes() / 1024 / 1024

    yield

    # 終了時のメモリ
    cpu_end = psutil.Process().memory_info().rss / 1024 / 1024
    gpu_end = cupy.cuda.MemoryPool().used_bytes() / 1024 / 1024

    print(f"\n{label}:")
    print(f"  CPU Memory: {cpu_start:.1f}MB → {cpu_end:.1f}MB (Δ{cpu_end-cpu_start:+.1f}MB)")
    print(f"  GPU Memory: {gpu_start:.1f}MB → {gpu_end:.1f}MB (Δ{gpu_end-gpu_start:+.1f}MB)")
```

### 3. kernel_profiler.py
**用途**: CUDAカーネルの詳細プロファイリング

```python
#!/usr/bin/env python3
"""CUDAカーネルプロファイラー"""

import cupy
from cupy import prof

def profile_kernel(kernel_name: str, grid_size: tuple, block_size: tuple):
    """カーネル実行時間を測定"""

    # ウォームアップ
    for _ in range(10):
        kernel[grid_size, block_size](...)

    # プロファイリング
    times = []
    for _ in range(100):
        start = cupy.cuda.Event()
        end = cupy.cuda.Event()

        start.record()
        kernel[grid_size, block_size](...)
        end.record()
        end.synchronize()

        elapsed_time = cupy.cuda.get_elapsed_time(start, end)
        times.append(elapsed_time)

    print(f"Kernel: {kernel_name}")
    print(f"  Average: {np.mean(times):.3f}ms")
    print(f"  Min: {np.min(times):.3f}ms")
    print(f"  Max: {np.max(times):.3f}ms")
    print(f"  Std: {np.std(times):.3f}ms")
```

### 4. type_validator.py
**用途**: データ型変換の検証

```python
#!/usr/bin/env python3
"""型変換検証ツール"""

import psycopg2
import pyarrow.parquet as pq
import numpy as np

def validate_type_conversion(pg_table: str, parquet_file: str):
    """PostgreSQLとParquetの型変換を検証"""

    # PostgreSQLから型情報取得
    conn = psycopg2.connect(...)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT column_name, data_type, numeric_precision, numeric_scale
        FROM information_schema.columns
        WHERE table_name = '{pg_table}'
    """)
    pg_schema = cur.fetchall()

    # Parquetスキーマ取得
    pq_file = pq.ParquetFile(parquet_file)
    pq_schema = pq_file.schema

    # 型マッピング検証
    for pg_col, pg_type, precision, scale in pg_schema:
        pq_field = pq_schema.field(pg_col)
        print(f"{pg_col}:")
        print(f"  PostgreSQL: {pg_type}({precision},{scale})")
        print(f"  Parquet: {pq_field.type}")

        # 型互換性チェック
        if not is_compatible(pg_type, pq_field.type):
            print(f"  WARNING: Type mismatch!")
```

## 使用例とワークフロー

### 1. 新機能開発時のデバッグフロー

```bash
# 1. バイナリフォーマット確認
python tools/binary_inspector.py /dev/shm/chunk_0.bin

# 2. メモリ使用量追跡
python tools/memory_trace.py --watch "python cu_pg_parquet.py --table lineorder"

# 3. カーネルプロファイリング
python tools/kernel_profiler.py --kernel parse_rows_and_fields_lite

# 4. 出力検証
python tools/show_parquet_sample.py output/lineorder_chunk_0.parquet --stats
python tools/compare_tables.py --pg-table lineorder --parquet-file output/lineorder.parquet
```

### 2. パフォーマンス調査

```bash
# GPU プロファイル
nsys profile --stats=true python cu_pg_parquet.py --table lineorder

# メモリ帯域幅測定
python tools/profile_gpu.py --measure bandwidth

# ボトルネック分析
python tools/profile_gpu.py --analyze bottlenecks
```

### 3. データ整合性検証

```bash
# 型変換検証
python tools/type_validator.py --pg-table lineorder --parquet output/lineorder.parquet

# サンプリング比較
python tools/compare_tables.py --sample-size 10000 --random

# 完全性チェック
python tools/compare_tables.py --check-completeness
```

## 環境変数によるデバッグ制御

```bash
# 詳細ログ出力
export GPUPGPARSER_DEBUG=1

# CUDAエラーの即時検出
export CUDA_LAUNCH_BLOCKING=1

# メモリアロケーション追跡
export GPUPGPARSER_TRACE_MEMORY=1

# カーネル実行時間出力
export GPUPGPARSER_PROFILE_KERNELS=1
```

## トラブルシューティングガイド

### よくある問題と対処法

1. **GPU Out of Memory**
   ```bash
   # メモリ使用量確認
   nvidia-smi
   python tools/memory_trace.py --gpu-only

   # チャンクサイズ調整
   export GPUPGPARSER_CHUNK_SIZE=100MB
   ```

2. **データ不整合**
   ```bash
   # 詳細比較
   python tools/compare_tables.py --verbose --show-diff

   # 特定行の調査
   python tools/binary_inspector.py chunk.bin --row 12345
   ```

3. **パフォーマンス低下**
   ```bash
   # プロファイル比較
   python tools/profile_gpu.py --compare baseline.json current.json

   # 環境変数最適化
   python tools/tune_parameters.py --auto
   ```
