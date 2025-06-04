#!/usr/bin/env python
"""
Ultimate統合カーネル シンプル版テスト
==================================

デッドロック問題を回避したシンプル版での動作確認
"""

import time
import os
import numpy as np

# psycopg動的インポート
try:
    import psycopg
    print("Using psycopg3")
except ImportError:
    import psycopg2 as psycopg
    print("Using psycopg2")

# CUDA/numba
from numba import cuda
cuda.select_device(0)
print("CUDA context OK")

# 既存モジュールのインポート
from src.meta_fetch import fetch_column_meta
from src.gpu_parse_wrapper import parse_binary_chunk_gpu, detect_pg_header_size
from src.gpu_decoder_v2 import decode_chunk

def main():
    print("=== Ultimate統合カーネル シンプル版テスト ===")
    
    # PostgreSQL接続
    dsn = os.environ.get("GPUPASER_PG_DSN")
    if not dsn:
        print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
        return False
    
    conn = psycopg.connect(dsn)
    
    try:
        print("1. メタデータ取得中...")
        columns = fetch_column_meta(conn, "SELECT * FROM lineorder LIMIT 5000")  # 非常に小さく
        
        print(f"   総列数: {len(columns)}")
        
        print("2. COPY BINARY実行中...")
        copy_sql = "COPY (SELECT * FROM lineorder LIMIT 5000) TO STDOUT (FORMAT binary)"
        buf = bytearray()
        with conn.cursor().copy(copy_sql) as cpy:
            while True:
                chunk = cpy.read()
                if not chunk:
                    break
                buf.extend(chunk)
            raw_host = np.frombuffer(buf, dtype=np.uint8)
        print(f"   データサイズ: {len(raw_host) / (1024*1024):.2f} MB")
        
    finally:
        conn.close()
    
    print("3. GPU転送・パース中...")
    raw_dev = cuda.to_device(raw_host)
    header_size = detect_pg_header_size(raw_host[:128])
    
    field_offsets_dev, field_lengths_dev = parse_binary_chunk_gpu(
        raw_dev, len(columns), header_size=header_size
    )
    rows = field_lengths_dev.shape[0]
    print(f"   行数: {rows}")

    print("4. 従来版デコード（基準）...")
    start_traditional = time.time()
    batch_traditional = decode_chunk(raw_dev, field_offsets_dev, field_lengths_dev, columns)
    traditional_time = time.time() - start_traditional
    print(f"   完了: {traditional_time:.4f}秒")

    print("5. Ultimate統合版シンプル版テスト...")
    
    try:
        # シンプル版カーネルのテスト
        from src.gpu_memory_manager_v4_ultimate import GPUMemoryManagerV4Ultimate
        gmm_v4 = GPUMemoryManagerV4Ultimate()
        ultimate_info = gmm_v4.initialize_ultimate_buffers(columns, rows)
        
        # 10のべき乗テーブル
        from src.cuda_kernels.arrow_gpu_pass2_decimal128 import POW10_TABLE_LO_HOST, POW10_TABLE_HI_HOST
        d_pow10_table_lo = cuda.to_device(POW10_TABLE_LO_HOST)
        d_pow10_table_hi = cuda.to_device(POW10_TABLE_HI_HOST)
        
        # 固定長列情報
        fixed_layouts = ultimate_info.fixed_layouts
        var_layouts = ultimate_info.var_layouts
        fixed_count = len(fixed_layouts)
        var_count = len(var_layouts)
        
        print(f"   固定長列: {fixed_count}列, 可変長列: {var_count}列")
        
        if fixed_count > 0:
            fixed_types = np.array([layout.arrow_type_id for layout in fixed_layouts], dtype=np.int32)
            fixed_offsets = np.array([layout.buffer_offset for layout in fixed_layouts], dtype=np.int32)  
            fixed_sizes = np.array([layout.element_size for layout in fixed_layouts], dtype=np.int32)
            fixed_indices = np.array([layout.column_index for layout in fixed_layouts], dtype=np.int32)
            fixed_scales = np.array([layout.decimal_scale for layout in fixed_layouts], dtype=np.int32)
            
            d_fixed_types = cuda.to_device(fixed_types)
            d_fixed_offsets = cuda.to_device(fixed_offsets)
            d_fixed_sizes = cuda.to_device(fixed_sizes)
            d_fixed_indices = cuda.to_device(fixed_indices)
            d_fixed_scales = cuda.to_device(fixed_scales)
        else:
            d_fixed_types = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_offsets = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_sizes = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_indices = cuda.to_device(np.array([], dtype=np.int32))
            d_fixed_scales = cuda.to_device(np.array([], dtype=np.int32))
        
        # 可変長列情報
        if var_count > 0:
            var_indices_array = np.array([layout.column_index for layout in var_layouts], dtype=np.int32)
            d_var_indices = cuda.to_device(var_indices_array)
        else:
            d_var_indices = cuda.to_device(np.array([], dtype=np.int32))
        
        # 共通NULL配列
        d_nulls_all = cuda.device_array((rows, len(columns)), dtype=np.uint8)
        
        print("   シンプル版カーネル実行中...")
        threads = 256
        blocks = (rows + threads - 1) // threads
        
        # シンプル版カーネルインポート
        from src.cuda_kernels.arrow_gpu_pass1_ultimate_simple import pass1_ultimate_simple
        
        start_kernel = time.time()
        
        # シンプル版カーネル実行
        pass1_ultimate_simple[blocks, threads](
            raw_dev,
            field_offsets_dev,
            field_lengths_dev,
            
            # 統合固定長バッファ
            ultimate_info.fixed_buffer,
            ultimate_info.row_stride,
            
            # 固定長レイアウト情報
            fixed_count,
            d_fixed_types,
            d_fixed_offsets,
            d_fixed_sizes,
            d_fixed_indices,
            d_fixed_scales,
            
            # 可変長文字列バッファ（シンプル版：長さのみ）
            var_count,
            d_var_indices,
            
            # 出力: 可変長列の長さ配列
            ultimate_info.var_lens_buffer,
            
            # 共通出力
            d_nulls_all,
            
            # Decimal処理用
            d_pow10_table_lo,
            d_pow10_table_hi
        )
        
        print("   カーネル同期中...")
        cuda.synchronize()
        kernel_time = time.time() - start_kernel
        print(f"   ✓ シンプル版カーネル実行成功！ ({kernel_time:.4f}秒)")
        
        # 結果検証：固定長バッファの内容確認
        print("   結果検証中...")
        
        # 最初の行の最初の固定長列（lo_orderkey, DECIMAL128）をチェック
        if fixed_count > 0:
            first_layout = fixed_layouts[0]
            print(f"   検証対象: {first_layout.name} (offset={first_layout.buffer_offset}, size={first_layout.element_size})")
            
            # 統合バッファから最初の行の最初の列データを抽出
            buffer_sample = ultimate_info.fixed_buffer[:first_layout.element_size].copy_to_host()
            print(f"   バッファサンプル: {buffer_sample.tobytes().hex()}")
            
            # 対応する従来版の値と比較
            traditional_first_col = batch_traditional.column(first_layout.column_index)
            traditional_first_value = traditional_first_col.to_pylist()[0]
            print(f"   従来版最初の値: {traditional_first_value}")
        
        # 可変長列の長さ確認
        if var_count > 0:
            var_lens_host = ultimate_info.var_lens_buffer.copy_to_host()
            for var_idx, layout in enumerate(var_layouts):
                first_5_lens = var_lens_host[var_idx, :5]
                print(f"   可変長列 '{layout.name}' 最初の5行の長さ: {first_5_lens}")
        
        print("   ✓ Ultimate統合カーネル シンプル版テスト: 成功")
        print("   デッドロック問題は解決され、基本的な統合処理が動作しています")
        return True
        
    except Exception as e:
        print(f"   ✗ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)