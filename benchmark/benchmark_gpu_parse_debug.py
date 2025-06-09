#!/usr/bin/env python3
"""
GPU Parser Debug Benchmark - whileループ終了原因分析版
----------------------------------------------------
* detect_rows_optimized_debugを使用してwhileループの終了原因を詳細分析
* 未検出部分の原因特定と境界問題の診断
* 見逃し位置の担当スレッド特定機能
"""

import os
import sys
import time
import argparse
import numpy as np
from numba import cuda

# パスの設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='GPU Parser Debug Benchmark')
    parser.add_argument('--rows', type=int, default=100000, help='行数制限')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    args = parser.parse_args()

    print("CUDA context OK")
    print("=== GPUパース whileループデバッグ版ベンチマーク ===")
    print(f"テーブル: lineorder")
    print(f"行数制限: {args.rows:,}")
    print(f"Debug版 Parser: 有効")
    print("----------------------------------------")

    try:
        # PostgreSQL接続とデータ取得（既存の実装を使用）
        import psycopg
        from src.metadata import fetch_column_meta
        from src.build_buf_from_postgres import detect_pg_header_size
        
        dsn = os.environ.get("GPUPASER_PG_DSN")
        if not dsn:
            print("エラー: 環境変数 GPUPASER_PG_DSN が設定されていません。")
            return 1

        prefix = os.environ.get("PG_TABLE_PREFIX", "")
        tbl = f"{prefix}lineorder" if prefix else "lineorder"
        
        print("1. メタデータ取得中...")
        meta_start = time.time()
        
        conn = psycopg.connect(dsn)
        try:
            columns = fetch_column_meta(conn, f"SELECT * FROM {tbl}")
            meta_time = time.time() - meta_start
            print(f"   完了: {meta_time:.4f}秒")
            
            ncols = len(columns)

            print("2. COPY BINARY実行中...")
            copy_start = time.time()
            copy_sql = f"COPY (SELECT * FROM {tbl} LIMIT {args.rows}) TO STDOUT (FORMAT binary)"
            buf = bytearray()
            with conn.cursor().copy(copy_sql) as cpy:
                while True:
                    chunk = cpy.read()
                    if not chunk:
                        break
                    buf.extend(chunk)
                raw_host = np.frombuffer(buf, dtype=np.uint8)
            copy_time = time.time() - copy_start
            data_size_mb = len(raw_host) / (1024 * 1024)
            print(f"   完了: {copy_time:.4f}秒, サイズ: {data_size_mb:.2f} MB")
        finally:
            conn.close()
        
        # GPU転送
        print("3. GPU転送中...")
        transfer_start = time.time()
        raw_dev = cuda.to_device(raw_host)
        transfer_time = time.time() - transfer_start
        
        # ヘッダーサイズ検出
        header_sample = raw_dev[:min(128, raw_dev.shape[0])].copy_to_host()
        header_size = detect_pg_header_size(header_sample)
        print(f"   完了: {transfer_time:.4f}秒")
        print(f"   ヘッダーサイズ: {header_size} バイト")
        
        # 従来版パーサーテスト（比較用）
        print("\n4. 従来版パーサーテスト...")
        from src.build_buf_from_postgres import parse_binary_chunk_gpu
        
        conv_start = time.time()
        field_offsets_conv, field_lengths_conv = parse_binary_chunk_gpu(
            raw_dev, ncols, threads_per_block=256, header_size=header_size
        )
        conv_time = time.time() - conv_start
        
        # 従来版の行数取得
        conv_rows = field_offsets_conv.shape[0] if field_offsets_conv.size > 0 else 0
        print(f"   従来版完了: {conv_time:.4f}秒, 行数: {conv_rows:,}")
        
        # 従来版の行位置抽出
        if args.debug and conv_rows > 0:
            print("[DEBUG] 従来版行位置を抽出中...")
            conv_field_offsets_host = field_offsets_conv.copy_to_host()
            conv_row_positions = []
            
            raw_host_data = raw_dev.copy_to_host()
            for row_idx in range(conv_rows):
                # 最初のフィールドオフセットから行ヘッダ位置を逆算
                first_field_offset = conv_field_offsets_host[row_idx, 0]
                if first_field_offset > 6:  # 2B(フィールド数) + 4B(フィールド長)
                    row_start = first_field_offset - 6
                    conv_row_positions.append(row_start)
            
            conv_row_positions = np.array(conv_row_positions, dtype=np.int32)
            print(f"[DEBUG] 従来版行位置: {len(conv_row_positions)}個抽出")
        
        # Debug版パーサーテスト
        print("\n5. Debug版パーサーテスト...")
        from src.cuda_kernels.ultra_fast_parser_debug import detect_rows_optimized_debug, analyze_loop_exit_reasons
        from src.cuda_kernels.ultra_fast_parser import (
            estimate_row_size_from_columns,
            calculate_optimal_grid_sm_aware
        )
        # extract_fieldsは内部で定義またはスキップ
        from src.types import PG_OID_TO_BINARY_SIZE
        
        # ColumnMetaから固定長フィールド情報を動的抽出
        fixed_field_lengths = np.full(ncols, -1, dtype=np.int32)
        for i, column in enumerate(columns):
            # NUMERIC固定長制約緩和: OID 1700は可変長として扱う
            if column.pg_oid == 1700:  # NUMERIC
                fixed_field_lengths[i] = -1
            else:
                pg_binary_size = PG_OID_TO_BINARY_SIZE.get(column.pg_oid)
                if pg_binary_size is not None:
                    fixed_field_lengths[i] = pg_binary_size
        
        fixed_field_lengths_dev = cuda.to_device(fixed_field_lengths)
        
        # 設定値計算
        estimated_row_size = estimate_row_size_from_columns(columns)
        data_size = raw_dev.size - header_size
        
        # グリッド配置計算
        blocks_x, blocks_y, threads_per_block = calculate_optimal_grid_sm_aware(
            data_size, estimated_row_size
        )
        
        actual_threads = blocks_x * blocks_y * threads_per_block
        thread_stride = (data_size + actual_threads - 1) // actual_threads
        if thread_stride < estimated_row_size:
            thread_stride = estimated_row_size
        
        print(f"[DEBUG] Debug版設定: threads={actual_threads}, thread_stride={thread_stride}")
        print(f"[DEBUG] First column: {columns[0].name}, pg_oid={columns[0].pg_oid}, elem_size={columns[0].elem_size}")
        print(f"[DEBUG] ★NUMERIC固定長制約を緩和: OID 1700 → 可変長(-1)")
        print(f"[DEBUG] ★境界オーバーラップ強化: end_pos += estimated_row_size")
        
        # Debug版カーネル実行
        max_rows = min(2_000_000, (data_size // estimated_row_size) * 2)
        debug_info_size = 800  # 100スレッド×4要素=400 + 基本領域400
        
        # デバイス配列準備
        row_positions_host = np.full(max_rows, -1, dtype=np.int32)
        row_positions = cuda.to_device(row_positions_host)
        row_count_host = np.zeros(1, dtype=np.int32)
        row_count = cuda.to_device(row_count_host)
        debug_array_host = np.full(debug_info_size, -1, dtype=np.int32)
        debug_array = cuda.to_device(debug_array_host)
        
        debug_start = time.time()
        grid_2d = (blocks_x, blocks_y)
        detect_rows_optimized_debug[grid_2d, threads_per_block](
            raw_dev, header_size, thread_stride, estimated_row_size, ncols,
            row_positions, row_count, max_rows, fixed_field_lengths_dev, debug_array
        )
        cuda.synchronize()
        debug_time = time.time() - debug_start
        
        # 結果取得
        nrow = int(row_count.copy_to_host()[0])
        debug_results = debug_array.copy_to_host()
        
        if nrow > 0:
            debug_positions = row_positions[:nrow].copy_to_host()
            valid_positions = debug_positions[debug_positions >= 0]
            debug_row_positions = np.sort(valid_positions)
            
            # フィールド抽出は従来版の結果を使用（デバッグ目的なので）
            field_offsets_debug = field_offsets_conv
            field_lengths_debug = field_lengths_conv
        else:
            debug_row_positions = np.array([], dtype=np.int32)
            field_offsets_debug = cuda.device_array((0, ncols), np.int32)
            field_lengths_debug = cuda.device_array((0, ncols), np.int32)
        
        print(f"   Debug版完了: {debug_time:.4f}秒, 行数: {len(debug_row_positions):,}")
        
        # whileループ終了原因の詳細分析
        if args.debug:
            analyze_loop_exit_reasons(debug_results, actual_threads, header_size, 
                                    thread_stride, estimated_row_size)
        
        # 行位置比較分析
        if args.debug and conv_rows > 0 and len(debug_row_positions) > 0:
            print(f"\n[DEBUG] ★行位置比較分析開始...")
            print(f"[DEBUG] ★従来版位置数: {len(conv_row_positions)}")
            print(f"[DEBUG] ★Debug版位置数: {len(debug_row_positions)}")
            
            # 見逃し行の特定
            conv_set = set(conv_row_positions)
            debug_set = set(debug_row_positions)
            
            missing_positions = conv_set - debug_set
            extra_positions = debug_set - conv_set
            
            print(f"[DEBUG] ★見逃し行数: {len(missing_positions)}")
            print(f"[DEBUG] ★余分検出数: {len(extra_positions)}")
            
            if missing_positions:
                missing_list = sorted(list(missing_positions))
                print(f"[DEBUG] ★見逃し位置（最初の10個）: {missing_list[:10]}")
                
                # 見逃し位置の分布分析
                if len(missing_list) > 1:
                    data_span = raw_dev.size - header_size
                    bins = np.linspace(header_size, raw_dev.size, 11)
                    hist, _ = np.histogram(missing_list, bins=bins)
                    print(f"[DEBUG] ★見逃し分布（10分割）: {hist.tolist()}")
                
                # 見逃し位置の詳細分析
                print(f"\n[DEBUG] ★見逃し位置詳細分析...")
                raw_host_data = raw_dev.copy_to_host()
                
                for i, pos in enumerate(missing_list[:2]):  # 最初の2個のみ詳細分析
                    print(f"\n見逃し位置 {i+1}: {pos}")
                    
                    # バイナリダンプ
                    if pos + 32 < len(raw_host_data):
                        chunk = raw_host_data[pos:pos+32]
                        hex_dump = ' '.join(f'{b:02x}' for b in chunk)
                        ascii_dump = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
                        print(f"  バイナリ: {hex_dump}")
                        print(f"  ASCII:   {ascii_dump}")
                        
                        # ヘッダ値確認
                        if pos + 1 < len(raw_host_data):
                            header_val = (raw_host_data[pos] << 8) | raw_host_data[pos + 1]
                            print(f"  ヘッダ値: {header_val} ({ncols}フィールド期待)")
                    
                    # 担当スレッド計算
                    responsible_tid = (pos - header_size) // thread_stride
                    thread_start = header_size + responsible_tid * thread_stride
                    thread_end = header_size + (responsible_tid + 1) * thread_stride + estimated_row_size
                    
                    print(f"  Thread {responsible_tid}: 担当 {thread_start}-{thread_end}")
                    
                    # 範囲判定
                    if thread_start <= pos < thread_end:
                        print(f"  範囲判定: 担当領域内 → Debug版がカウントすべき")
                        
                        # 15Bステップでの検出可能性計算
                        step_offset = (pos - thread_start) % 15
                        step_num = (pos - thread_start) // 15
                        read_start = thread_start + step_num * 15
                        read_end = read_start + 16
                        
                        print(f"  15Bステップ進行: step#{step_num}, pos={read_start}")
                        print(f"  16B読み込み範囲: {read_start}-{read_end} (pos={pos})")
                        
                        if read_start <= pos < read_end:
                            print(f"  捕捉判定: 読み込み範囲内で検出可能")
                        else:
                            print(f"  捕捉判定: 読み込み範囲外（隙間問題）")
                    else:
                        print(f"  範囲判定: 担当領域外（境界問題）")
        
        # 性能比較
        print(f"\n=== 性能比較 ===")
        print(f"従来版:      {conv_time:.4f}秒")
        print(f"Debug版:     {debug_time:.4f}秒")
        if conv_time > 0:
            speedup = conv_time / debug_time
            print(f"高速化倍率:  {speedup:.2f}x")
        
        # 行数比較
        if conv_rows == len(debug_row_positions):
            print(f"✅ 行数一致: 従来版={conv_rows:,}, Debug版={len(debug_row_positions):,}")
        else:
            print(f"❌ 行数不一致: 従来版={conv_rows:,}, Debug版={len(debug_row_positions):,}")
        
        # 総合結果
        total_time = meta_time + copy_time + transfer_time + debug_time
        throughput = data_size_mb / total_time if total_time > 0 else 0
        
        print(f"\n=== 総合結果 ===")
        print(f"メタデータ取得: {meta_time:.4f}秒")
        print(f"COPY BINARY:   {copy_time:.4f}秒")
        print(f"GPU転送:       {transfer_time:.4f}秒")
        print(f"GPUパース:     {debug_time:.4f}秒")
        print(f"総時間:        {total_time:.4f}秒")
        print(f"処理データ:    {data_size_mb:.2f} MB")
        print(f"スループット:  {throughput:.2f} MB/秒")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())