"""GPUメモリダンプ調査ツール

大規模データ処理時の各ステップでGPUメモリ状態をダンプし、
奇数行の文字列破損の原因を特定する
"""

import os
import numpy as np
import cupy as cp
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.main_postgres_to_parquet import ZeroCopyProcessor
from src.direct_column_extractor import DirectColumnExtractor
import psycopg
import rmm

def dump_gpu_memory(data_gpu, offset, length, label):
    """GPUメモリの特定範囲をダンプ"""
    try:
        if isinstance(data_gpu, cuda.cudadrv.devicearray.DeviceNDArray):
            # Numba配列の場合
            data_host = data_gpu[offset:offset+length].copy_to_host()
        else:
            # CuPy配列の場合
            data_host = data_gpu[offset:offset+length].get()
        
        print(f"\n=== {label} ===")
        print(f"Offset: {offset}, Length: {length}")
        print(f"Hex: {' '.join(f'{b:02x}' for b in data_host[:min(64, len(data_host))])}")
        if length <= 256:
            try:
                print(f"ASCII: {data_host.tobytes().decode('ascii', errors='replace')}")
            except:
                print("ASCII: (decode error)")
        return data_host
    except Exception as e:
        print(f"Dump error: {e}")
        return None

print("=== GPUメモリダンプ調査 ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# lo_orderpriority列のインデックスを取得
lo_orderpriority_idx = None
for i, col in enumerate(columns):
    if col.name == 'lo_orderpriority':
        lo_orderpriority_idx = i
        break

print(f"lo_orderpriority列インデックス: {lo_orderpriority_idx}")

# チャンクファイルを探す
chunk_file = None
for f in ['/dev/shm/chunk_0.bin', '/dev/shm/chunk_2.bin']:
    if os.path.exists(f):
        chunk_file = f
        break

if not chunk_file:
    print("チャンクファイルが見つかりません")
    exit(1)

# チャンク読み込み（大規模データ）
print(f"\nチャンク読み込み: {chunk_file}")
with open(chunk_file, 'rb') as f:
    data = f.read()

print(f"チャンクサイズ: {len(data) / (1024**3):.2f} GB")

raw_host = np.frombuffer(data, dtype=np.uint8)
raw_dev = cuda.to_device(raw_host)

# ヘッダーサイズ検出
header_sample = raw_dev[:128].copy_to_host()
header_size = detect_pg_header_size(header_sample)

print("\n=== ステップ1: GPUパース ===")
field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
    raw_dev, columns, header_size=header_size
)

rows = field_offsets_dev.shape[0]
print(f"検出行数: {rows}")

# フィールド情報をホストにコピー
field_offsets_host = field_offsets_dev.copy_to_host()
field_lengths_host = field_lengths_dev.copy_to_host()

# 調査対象の行を選定（偶数行と奇数行のペア）
test_pairs = [
    (0, 1),      # 最初のペア
    (100, 101),  # 100行目のペア
    (1000, 1001),
    (10000, 10001),
    (100000, 100001) if rows > 100001 else None,
    (1000000, 1000001) if rows > 1000001 else None
]

print("\n=== ステップ2: フィールドデータのダンプ ===")
for pair in test_pairs:
    if pair is None:
        continue
    
    even_row, odd_row = pair
    if odd_row >= rows:
        continue
    
    print(f"\n--- 行ペア {even_row}/{odd_row} ---")
    
    # lo_orderpriority列のデータをダンプ
    for row_idx in [even_row, odd_row]:
        offset = field_offsets_host[row_idx, lo_orderpriority_idx]
        length = field_lengths_host[row_idx, lo_orderpriority_idx]
        
        if length > 0:
            label = f"行{row_idx}({'偶数' if row_idx % 2 == 0 else '奇数'}) lo_orderpriority"
            dump_gpu_memory(raw_dev, offset, min(length, 32), label)

print("\n=== ステップ3: 文字列バッファ作成 ===")
processor = ZeroCopyProcessor()
string_buffers = processor.create_string_buffers(
    columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
)

# 文字列バッファの内容をダンプ
if 'lo_orderpriority' in string_buffers:
    buffer_info = string_buffers['lo_orderpriority']
    if buffer_info['data'] is not None:
        print("\n=== 文字列バッファメモリダンプ ===")
        
        # オフセット配列を取得
        offsets_buffer = buffer_info['offsets']
        if isinstance(offsets_buffer, rmm.DeviceBuffer):
            # RMM DeviceBufferの場合
            offsets_ptr = offsets_buffer.ptr
            offsets_size = offsets_buffer.size
            offsets_cupy = cp.ndarray(
                shape=(rows + 1,),
                dtype=cp.int32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(offsets_ptr, offsets_size, offsets_buffer),
                    0
                )
            )
        else:
            # Numba配列の場合
            offsets_ptr = offsets_buffer.__cuda_array_interface__['data'][0]
            offsets_cupy = cp.asarray(cp.ndarray(
                shape=(rows + 1,),
                dtype=cp.int32,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(offsets_ptr, (rows + 1) * 4, offsets_buffer),
                    0
                )
            ))
        
        offsets_host = offsets_cupy.get()
        
        # データバッファを取得
        data_buffer = buffer_info['data']
        if isinstance(data_buffer, rmm.DeviceBuffer):
            data_cupy = cp.ndarray(
                shape=(buffer_info['actual_size'],),
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(data_buffer.ptr, data_buffer.size, data_buffer),
                    0
                )
            )
        else:
            data_ptr = data_buffer.__cuda_array_interface__['data'][0]
            data_cupy = cp.asarray(cp.ndarray(
                shape=(buffer_info['actual_size'],),
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(data_ptr, buffer_info['actual_size'], data_buffer),
                    0
                )
            ))
        
        # 調査対象行の文字列データをダンプ
        for pair in test_pairs:
            if pair is None:
                continue
            even_row, odd_row = pair
            if odd_row >= rows:
                continue
            
            print(f"\n--- 文字列バッファ 行ペア {even_row}/{odd_row} ---")
            
            for row_idx in [even_row, odd_row]:
                start = offsets_host[row_idx]
                end = offsets_host[row_idx + 1]
                length = end - start
                
                print(f"\n行{row_idx}({'偶数' if row_idx % 2 == 0 else '奇数'}):")
                print(f"  オフセット: {start} -> {end} (長さ: {length})")
                
                if length > 0 and length <= 100:
                    data_slice = data_cupy[start:end].get()
                    print(f"  Hex: {' '.join(f'{b:02x}' for b in data_slice)}")
                    print(f"  ASCII: {repr(data_slice.tobytes())}")
                    
                    # 期待される文字列パターンとの比較
                    expected_patterns = [b'1-URGENT', b'2-HIGH', b'3-MEDIUM', b'4-NOT SPECI', b'5-LOW']
                    is_valid = any(data_slice.tobytes().startswith(p) for p in expected_patterns)
                    if not is_valid and row_idx % 2 == 1:
                        print(f"  ⚠️ 奇数行で無効なデータ！")

print("\n=== ステップ4: cuDF DataFrame作成前後の比較 ===")
# DirectColumnExtractorのデバッグモードを有効化するため、
# 一時的にデバッグ情報を追加
try:
    extractor = DirectColumnExtractor()
    cudf_df = extractor.extract_columns_direct(
        raw_dev, field_offsets_dev, field_lengths_dev,
        columns, string_buffers
    )
    
    # DataFrame内のデータを確認
    print("\n=== DataFrame内の文字列データ ===")
    for pair in test_pairs[:3]:  # 最初の3ペアのみ
        if pair is None:
            continue
        even_row, odd_row = pair
        if odd_row >= len(cudf_df):
            continue
        
        print(f"\n行ペア {even_row}/{odd_row}:")
        for row_idx in [even_row, odd_row]:
            try:
                value = cudf_df['lo_orderpriority'].iloc[row_idx]
                print(f"  行{row_idx}({'偶数' if row_idx % 2 == 0 else '奇数'}): {repr(value)}")
            except Exception as e:
                print(f"  行{row_idx}: エラー - {e}")
    
except Exception as e:
    print(f"DataFrame作成エラー: {e}")
    import traceback
    traceback.print_exc()

# メモリ使用状況
print("\n=== GPUメモリ使用状況 ===")
try:
    mempool = cp.get_default_memory_pool()
    used_bytes = mempool.used_bytes()
    total_bytes = mempool.total_bytes()
    print(f"使用中: {used_bytes / (1024**3):.2f} GB")
    print(f"合計: {total_bytes / (1024**3):.2f} GB")
except:
    pass

print("\n=== 調査完了 ===")
print("上記のダンプから、奇数行での文字列破損がどの段階で発生しているかを特定できます。")