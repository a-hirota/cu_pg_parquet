#!/usr/bin/env python3
"""
GPU vs CPUソートの性能比較テスト
================================

統合パーサーでのGPUソート最適化効果を測定します。
"""

import pytest
import numpy as np
import time
from numba import cuda
import cupy as cp

def test_gpu_sort_vs_cpu_sort():
    """GPUソート vs CPUソートの性能比較"""
    
    # テストデータサイズ（適応的ソート閾値を考慮）
    test_sizes = [10000, 100000, 1000000]  # 小規模、大規模、超大規模
    
    for data_size in test_sizes:
        print(f"\n=== データサイズ: {data_size:,}行 ===")
        
        # ランダムな行位置データを生成（PostgreSQLバイナリ位置を模擬）
        np.random.seed(42)  # 再現性のため
        row_positions_host = np.random.randint(0, data_size * 100, size=data_size, dtype=np.int32)
        row_positions_gpu = cuda.to_device(row_positions_host)
        
        # フィールドデータも生成（統合パーサーの実際の使用ケース）
        field_offsets_host = np.random.randint(0, 1000000, size=(data_size, 17), dtype=np.int32)
        field_lengths_host = np.random.randint(1, 100, size=(data_size, 17), dtype=np.int32)
        field_offsets_gpu = cuda.to_device(field_offsets_host)
        field_lengths_gpu = cuda.to_device(field_lengths_host)
        
        # === CPU従来方式の測定 ===
        start_time = time.perf_counter()
        
        # GPU→CPU転送
        row_pos_host = row_positions_gpu.copy_to_host()
        field_off_host = field_offsets_gpu.copy_to_host()
        field_len_host = field_lengths_gpu.copy_to_host()
        
        # CPUソート
        sort_indices_cpu = np.argsort(row_pos_host)
        field_offsets_sorted_cpu = field_off_host[sort_indices_cpu]
        field_lengths_sorted_cpu = field_len_host[sort_indices_cpu]
        
        # CPU→GPU転送
        result_offsets_cpu = cuda.to_device(field_offsets_sorted_cpu)
        result_lengths_cpu = cuda.to_device(field_lengths_sorted_cpu)
        
        cpu_time = time.perf_counter() - start_time
        
        # === GPU新方式の測定 ===
        start_time = time.perf_counter()
        
        # GPU上で直接処理
        row_positions_cupy = cp.asarray(row_positions_gpu)
        field_offsets_cupy = cp.asarray(field_offsets_gpu)
        field_lengths_cupy = cp.asarray(field_lengths_gpu)
        
        # GPUソート
        sort_indices_gpu = cp.argsort(row_positions_cupy)
        field_offsets_sorted_gpu = field_offsets_cupy[sort_indices_gpu]
        field_lengths_sorted_gpu = field_lengths_cupy[sort_indices_gpu]
        
        # CuPy→Numba変換
        result_offsets_gpu = cuda.as_cuda_array(field_offsets_sorted_gpu)
        result_lengths_gpu = cuda.as_cuda_array(field_lengths_sorted_gpu)
        
        gpu_time = time.perf_counter() - start_time
        
        # === 結果検証 ===
        # ソート結果が同じであることを確認
        cpu_result = result_offsets_cpu.copy_to_host()
        gpu_result = result_offsets_gpu.copy_to_host()
        
        # ソートインデックスが同じであることを確認（より確実な検証）
        sort_indices_cpu_host = sort_indices_cpu
        sort_indices_gpu_host = sort_indices_gpu.get()
        
        # デバッグ情報出力
        if not np.array_equal(sort_indices_cpu_host, sort_indices_gpu_host):
            print(f"[DEBUG] CPU sort indices: {sort_indices_cpu_host[:10]}")
            print(f"[DEBUG] GPU sort indices: {sort_indices_gpu_host[:10]}")
            print(f"[DEBUG] 元データ: {row_pos_host[:10]}")
            
            # 重複値の処理方法に違いがある可能性があるので、ソート後の実際の値を比較
            sorted_positions_cpu = row_pos_host[sort_indices_cpu_host]
            sorted_positions_gpu = row_positions_cupy[sort_indices_gpu].get()
            
            if np.array_equal(sorted_positions_cpu, sorted_positions_gpu):
                print("[DEBUG] ソート後の位置は一致（インデックス順序の違いは許容）")
            else:
                assert False, "ソート後の位置が不一致"
        else:
            assert np.array_equal(cpu_result, gpu_result), "CPU/GPUソート結果が不一致"
        
        # === 性能結果表示 ===
        speedup = cpu_time / gpu_time
        print(f"CPU方式: {cpu_time*1000:.2f}ms")
        print(f"GPU方式: {gpu_time*1000:.2f}ms")
        print(f"高速化率: {speedup:.2f}x")
        print(f"削減時間: {(cpu_time - gpu_time)*1000:.2f}ms")
        
        # 大規模データでのみGPU方式が高速であることを確認
        if data_size >= 100000:  # 大規模データのみチェック
            assert gpu_time < cpu_time, f"大規模データでGPU方式が遅い: {gpu_time:.4f}s vs {cpu_time:.4f}s"
            assert speedup >= 1.5, f"期待される高速化率に達していません: {speedup:.2f}x < 1.5x"
        else:
            # 小規模データではCPU方式が高速（期待通り）
            print(f"小規模データ（{data_size}行）ではCPU方式が高速: {speedup:.2f}x（期待通り）")

def test_gpu_sort_integration_with_parser():
    """統合パーサーでのGPUソート動作テスト"""
    
    try:
        from src.cuda_kernels.integrated_parser_lite import parse_binary_chunk_gpu_ultra_fast_v2_lite
        from src.types import ColumnMeta, INT32, UTF8
    except ImportError:
        pytest.skip("統合パーサーモジュールが利用できません")
    
    # シンプルなテストデータ
    columns = [
        ColumnMeta(name="id", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="name", pg_oid=25, pg_typmod=-1, arrow_id=UTF8, elem_size=-1)
    ]
    
    # PostgreSQL形式のテストバイナリデータ（簡易版）
    test_data = bytearray()
    test_data.extend(b'\x00' * 19)  # ヘッダ
    
    # 3行のテストデータ
    for i in range(3):
        test_data.extend((2).to_bytes(2, 'big'))  # フィールド数
        test_data.extend((4).to_bytes(4, 'big'))  # ID長
        test_data.extend((i + 1).to_bytes(4, 'big'))  # ID値
        test_data.extend((5).to_bytes(4, 'big'))  # 名前長
        test_data.extend(f"user{i}".encode('utf-8'))  # 名前
    
    raw_dev = cuda.to_device(np.frombuffer(test_data, dtype=np.uint8))
    
    # 統合パーサー実行（GPUソート使用）
    field_offsets, field_lengths = parse_binary_chunk_gpu_ultra_fast_v2_lite(
        raw_dev, columns, debug=True
    )
    
    # 結果確認
    assert field_offsets.shape[0] > 0, "行が検出されませんでした"
    assert field_offsets.shape[1] == len(columns), "列数が不正です"
    
    print(f"GPUソート統合テスト成功: {field_offsets.shape[0]}行検出")

if __name__ == "__main__":
    print("GPU vs CPUソート性能比較テスト開始...")
    
    test_gpu_sort_vs_cpu_sort()
    print("\n✓ 性能比較テスト完了")
    
    test_gpu_sort_integration_with_parser()
    print("✓ 統合パーサーテスト完了")
    
    print("\n🚀 GPUソート最適化が正常に動作しています！")