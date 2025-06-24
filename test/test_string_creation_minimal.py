"""最小限の文字列作成テスト"""

import cupy as cp
import cudf
import rmm
import pylibcudf as plc

print("=== 最小限の文字列作成テスト ===")

# テストデータ
test_strings = [
    b"1-URGENT       ",  # 15 bytes
    b"2-HIGH         ",  # 15 bytes
    b"3-MEDIUM       ",  # 15 bytes
    b"4-NOT SPECI    ",  # 15 bytes
    b"5-LOW          ",  # 15 bytes
] * 10  # 50 strings total

rows = len(test_strings)

# オフセット計算
offsets_host = cp.zeros(rows + 1, dtype=cp.int32)
total_size = 0
for i, s in enumerate(test_strings):
    offsets_host[i] = total_size
    total_size += len(s)
offsets_host[rows] = total_size

# データ作成
data_host = cp.zeros(total_size, dtype=cp.uint8)
for i, s in enumerate(test_strings):
    start = offsets_host[i]
    end = offsets_host[i + 1]
    data_host[start:end] = cp.frombuffer(s, dtype=cp.uint8)

print(f"rows: {rows}, total_size: {total_size}")
print(f"offsets: {offsets_host[:10]}...")

# RMM DeviceBufferに変換
offsets_bytes = offsets_host.get().tobytes()
data_bytes = data_host.get().tobytes()

offsets_buffer = rmm.DeviceBuffer.to_device(offsets_bytes)
data_buffer = rmm.DeviceBuffer.to_device(data_bytes)

print(f"\nRMM DeviceBuffer作成完了")
print(f"offsets_buffer.size: {offsets_buffer.size}")
print(f"data_buffer.size: {data_buffer.size}")

# 方法1: 直接gpumemoryview（直接抽出版の新しい方法）
try:
    print("\n=== 方法1: 直接gpumemoryview ===")
    
    # オフセット子カラム
    offsets_mv = plc.gpumemoryview(offsets_buffer)
    offsets_col = plc.column.Column(
        plc.types.DataType(plc.types.TypeId.INT32),
        rows + 1,
        offsets_mv,
        None, 0, 0, []
    )
    
    # STRING親カラム
    chars_mv = plc.gpumemoryview(data_buffer)
    parent = plc.column.Column(
        plc.types.DataType(plc.types.TypeId.STRING),
        rows,
        chars_mv,
        None, 0, 0,
        [offsets_col]
    )
    
    # Series作成
    series1 = cudf.Series.from_pylibcudf(parent)
    print(f"成功！最初の5行:")
    for i in range(min(5, len(series1))):
        print(f"  {i}: {repr(series1.iloc[i])}")
        
except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()

# 方法2: ホスト転送経由（基本ベンチマークの方法）
try:
    print("\n=== 方法2: ホスト転送経由 ===")
    
    # すでにホスト転送済みなので、そのまま使用
    offsets_mv2 = plc.gpumemoryview(offsets_buffer)
    offsets_col2 = plc.column.Column(
        plc.types.DataType(plc.types.TypeId.INT32),
        rows + 1,
        offsets_mv2,
        None, 0, 0, []
    )
    
    chars_mv2 = plc.gpumemoryview(data_buffer)
    parent2 = plc.column.Column(
        plc.types.DataType(plc.types.TypeId.STRING),
        rows,
        chars_mv2,
        None, 0, 0,
        [offsets_col2]
    )
    
    series2 = cudf.Series.from_pylibcudf(parent2)
    print(f"成功！最初の5行:")
    for i in range(min(5, len(series2))):
        print(f"  {i}: {repr(series2.iloc[i])}")
        
except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n=== テスト完了 ===")