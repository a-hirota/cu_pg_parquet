"""文字列破損パターンの特定

どの行数・どのチャンクで破損が発生するかを特定
"""

import os
import numpy as np
from numba import cuda
from src.cuda_kernels.postgres_binary_parser import parse_binary_chunk_gpu_ultra_fast_v2, detect_pg_header_size
from src.metadata import fetch_column_meta
from src.main_postgres_to_parquet import ZeroCopyProcessor
from src.direct_column_extractor import DirectColumnExtractor
import psycopg
import cupy as cp

print("=== 文字列破損パターン特定ツール ===")

# メタデータ取得
dsn = os.environ.get("GPUPASER_PG_DSN")
conn = psycopg.connect(dsn)
columns = fetch_column_meta(conn, "SELECT * FROM lineorder")
conn.close()

# 全チャンクを調査
all_chunks = []
for i in range(8):
    chunk_file = f"/dev/shm/chunk_{i}.bin"
    if os.path.exists(chunk_file):
        all_chunks.append((i, chunk_file))

print(f"発見されたチャンク: {len(all_chunks)}個")

def check_chunk_for_corruption(chunk_id, chunk_file, sample_size=100000):
    """チャンクの破損をチェック"""
    print(f"\n=== チャンク{chunk_id} 調査 ===")
    
    try:
        # チャンク読み込み
        with open(chunk_file, 'rb') as f:
            data = f.read()
        
        print(f"チャンクサイズ: {len(data) / (1024**3):.2f} GB")
        
        raw_host = np.frombuffer(data, dtype=np.uint8)
        raw_dev = cuda.to_device(raw_host)
        
        # ヘッダーサイズ検出
        header_sample = raw_dev[:128].copy_to_host()
        header_size = detect_pg_header_size(header_sample)
        
        # GPUパース
        field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu_ultra_fast_v2(
            raw_dev, columns, header_size=header_size
        )
        
        rows = field_offsets_dev.shape[0]
        print(f"検出行数: {rows:,}")
        
        # 文字列バッファ作成
        processor = ZeroCopyProcessor()
        string_buffers = processor.create_string_buffers(
            columns, rows, raw_dev, field_offsets_dev, field_lengths_dev
        )
        
        # cuDF DataFrame作成
        extractor = DirectColumnExtractor()
        cudf_df = extractor.extract_columns_direct(
            raw_dev, field_offsets_dev, field_lengths_dev,
            columns, string_buffers
        )
        
        # 破損パターンを調査
        corruption_info = {
            'total_rows': rows,
            'even_errors': 0,
            'odd_errors': 0,
            'first_error_row': None,
            'error_pattern': {}
        }
        
        # サンプリング調査
        check_rows = min(sample_size, rows)
        
        # 最初の部分
        for i in range(min(10000, check_rows)):
            try:
                value = cudf_df['lo_orderpriority'].iloc[i]
                expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                is_valid = any(value.startswith(p) for p in expected_patterns)
                
                if not is_valid:
                    if i % 2 == 0:
                        corruption_info['even_errors'] += 1
                    else:
                        corruption_info['odd_errors'] += 1
                    
                    if corruption_info['first_error_row'] is None:
                        corruption_info['first_error_row'] = i
                        print(f"  最初のエラー: 行{i} ({'偶数' if i % 2 == 0 else '奇数'}): {repr(value)}")
                    
                    # エラーパターンを記録
                    if value not in corruption_info['error_pattern']:
                        corruption_info['error_pattern'][value] = 0
                    corruption_info['error_pattern'][value] += 1
                    
            except Exception as e:
                pass
        
        # 中間部分（もし行数が多い場合）
        if rows > 1000000:
            middle_start = rows // 2
            for i in range(middle_start, middle_start + 10000):
                if i >= rows:
                    break
                try:
                    value = cudf_df['lo_orderpriority'].iloc[i]
                    expected_patterns = ['1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECI', '5-LOW']
                    is_valid = any(value.startswith(p) for p in expected_patterns)
                    
                    if not is_valid:
                        if i % 2 == 0:
                            corruption_info['even_errors'] += 1
                        else:
                            corruption_info['odd_errors'] += 1
                except:
                    pass
        
        # 結果表示
        print(f"\n破損統計:")
        print(f"  偶数行エラー: {corruption_info['even_errors']}")
        print(f"  奇数行エラー: {corruption_info['odd_errors']}")
        
        if corruption_info['odd_errors'] > corruption_info['even_errors'] * 2:
            print("  ⚠️ 奇数行に顕著な破損パターン！")
        
        if corruption_info['error_pattern']:
            print(f"\nエラーパターン（上位5個）:")
            sorted_patterns = sorted(corruption_info['error_pattern'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for pattern, count in sorted_patterns:
                print(f"  {repr(pattern)}: {count}回")
        
        # メモリクリーンアップ
        del cudf_df
        del raw_dev
        cuda.synchronize()
        
        return corruption_info
        
    except Exception as e:
        print(f"チャンク{chunk_id}処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

# 各チャンクを調査
results = {}
for chunk_id, chunk_file in all_chunks[:4]:  # 最初の4チャンクのみ
    result = check_chunk_for_corruption(chunk_id, chunk_file)
    if result:
        results[chunk_id] = result
    
    # GPUメモリをクリア
    try:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
    except:
        pass

# 結果サマリー
print("\n=== 調査結果サマリー ===")
for chunk_id, info in results.items():
    print(f"\nチャンク{chunk_id}:")
    print(f"  総行数: {info['total_rows']:,}")
    print(f"  偶数行エラー: {info['even_errors']}")
    print(f"  奇数行エラー: {info['odd_errors']}")
    if info['first_error_row'] is not None:
        print(f"  最初のエラー行: {info['first_error_row']}")

# パターン分析
print("\n=== パターン分析 ===")
has_corruption = False
for chunk_id, info in results.items():
    if info['odd_errors'] > 0 or info['even_errors'] > 0:
        has_corruption = True
        print(f"チャンク{chunk_id}で破損を検出")

if not has_corruption:
    print("調査したチャンクでは破損は検出されませんでした。")
    print("より大規模な処理条件（並列処理、複数チャンク同時処理など）で発生する可能性があります。")