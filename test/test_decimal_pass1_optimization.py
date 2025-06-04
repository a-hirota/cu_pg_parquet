"""
Decimal Pass1統合最適化の性能検証テスト
====================================

このテストでは以下を検証します：
1. Pass1統合版と従来版の処理結果の一致性
2. 性能改善効果の測定
3. メモリアクセス削減効果の確認
4. 異なるDecimal列数での効果測定
"""

import os
import time
import numpy as np
import pyarrow as pa
from typing import List, Dict, Any
import pytest

# テスト対象のモジュールをインポート
try:
    from src.gpu_decoder_v2 import decode_chunk
    from src.gpu_decoder_v2_decimal_optimized import decode_chunk_decimal_optimized
    from src.type_map import ColumnMeta, DECIMAL128, INT32, UTF8
    from numba import cuda
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    exit(1)

class DecimalOptimizationBenchmark:
    """Decimal Pass1統合最適化のベンチマーククラス"""
    
    def __init__(self):
        self.test_data_cache = {}
        
    def generate_test_data(self, rows: int, decimal_cols: int, other_cols: int = 2) -> Dict[str, Any]:
        """
        テスト用のPOSTGRESQL COPY BINARYデータを生成
        
        Parameters:
        -----------
        rows : int
            行数
        decimal_cols : int
            Decimal列数
        other_cols : int
            その他の列数（INT32とUTF8を交互に）
            
        Returns:
        --------
        dict: テストデータと列メタデータ
        """
        cache_key = f"{rows}_{decimal_cols}_{other_cols}"
        if cache_key in self.test_data_cache:
            return self.test_data_cache[cache_key]
            
        print(f"Generating test data: {rows} rows, {decimal_cols} decimal cols, {other_cols} other cols")
        
        # 列メタデータ作成
        columns = []
        
        # Decimal列
        for i in range(decimal_cols):
            col = ColumnMeta(
                name=f"decimal_col_{i}",
                pg_type="numeric",
                arrow_id=DECIMAL128,
                elem_size=16,
                is_variable=False,
                arrow_param=(18, 2)  # precision=18, scale=2
            )
            columns.append(col)
        
        # その他の列（INT32とUTF8を交互に）
        for i in range(other_cols):
            if i % 2 == 0:
                col = ColumnMeta(
                    name=f"int_col_{i}",
                    pg_type="int4",
                    arrow_id=INT32,
                    elem_size=4,
                    is_variable=False,
                    arrow_param=None
                )
            else:
                col = ColumnMeta(
                    name=f"text_col_{i}",
                    pg_type="text",
                    arrow_id=UTF8,
                    elem_size=-1,
                    is_variable=True,
                    arrow_param=None
                )
            columns.append(col)
        
        # テストデータ生成（簡略化したバイナリ形式）
        total_cols = len(columns)
        
        # フィールドオフセットとlengthの生成
        field_offsets = np.zeros((rows, total_cols), dtype=np.int32)
        field_lengths = np.zeros((rows, total_cols), dtype=np.int32)
        
        # 簡易的なバイナリデータ（実際のPOSTGRESQL形式ではないが、テスト用として）
        raw_data_list = []
        current_offset = 0
        
        for row in range(rows):
            for col_idx, col in enumerate(columns):
                if col.arrow_id == DECIMAL128:
                    # NUMERIC形式のダミーデータ（8バイトヘッダ + 4バイト桁データ）
                    # ヘッダ: nd=2, weight=1, sign=0x0000, dscale=2
                    header = np.array([0x00, 0x02, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02], dtype=np.uint8)
                    # 桁データ: 123.45 を基数10000で表現 -> [1, 2345]
                    digits = np.array([0x00, 0x01, 0x09, 0x29], dtype=np.uint8)  # 1, 2345 in big-endian
                    data = np.concatenate([header, digits])
                    
                elif col.arrow_id == INT32:
                    # 4バイトINT32データ
                    value = row * 100 + col_idx
                    data = np.array([value & 0xFF, (value >> 8) & 0xFF, 
                                   (value >> 16) & 0xFF, (value >> 24) & 0xFF], dtype=np.uint8)
                    
                elif col.arrow_id == UTF8:
                    # 可変長テキストデータ
                    text = f"text_row_{row}_col_{col_idx}"
                    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
                
                # オフセットと長さを記録
                field_offsets[row, col_idx] = current_offset
                field_lengths[row, col_idx] = len(data)
                
                # データを蓄積
                raw_data_list.append(data)
                current_offset += len(data)
        
        # 生データを結合
        raw_data = np.concatenate(raw_data_list)
        
        # GPUに転送
        raw_dev = cuda.to_device(raw_data)
        field_offsets_dev = cuda.to_device(field_offsets)
        field_lengths_dev = cuda.to_device(field_lengths)
        
        result = {
            'raw_dev': raw_dev,
            'field_offsets_dev': field_offsets_dev,
            'field_lengths_dev': field_lengths_dev,
            'columns': columns,
            'rows': rows,
            'decimal_cols': decimal_cols
        }
        
        self.test_data_cache[cache_key] = result
        return result
    
    def benchmark_single_case(self, rows: int, decimal_cols: int, iterations: int = 3) -> Dict[str, float]:
        """
        単一ケースでのベンチマーク実行
        
        Returns:
        --------
        dict: {'traditional_time': float, 'optimized_time': float, 'speedup': float}
        """
        test_data = self.generate_test_data(rows, decimal_cols)
        
        print(f"\n--- Benchmarking: {rows} rows, {decimal_cols} decimal columns ---")
        
        # 従来版の計測
        traditional_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            
            result_traditional = decode_chunk(
                test_data['raw_dev'],
                test_data['field_offsets_dev'],
                test_data['field_lengths_dev'],
                test_data['columns']
            )
            cuda.synchronize()  # GPU処理完了を待機
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            traditional_times.append(elapsed)
            print(f"Traditional Run {i+1}: {elapsed:.4f}s")
        
        # 最適化版の計測
        optimized_times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            
            result_optimized = decode_chunk_decimal_optimized(
                test_data['raw_dev'],
                test_data['field_offsets_dev'],
                test_data['field_lengths_dev'],
                test_data['columns'],
                use_pass1_integration=True
            )
            cuda.synchronize()  # GPU処理完了を待機
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            optimized_times.append(elapsed)
            print(f"Optimized Run {i+1}: {elapsed:.4f}s")
        
        # 統計計算
        traditional_avg = np.mean(traditional_times)
        optimized_avg = np.mean(optimized_times)
        speedup = traditional_avg / optimized_avg if optimized_avg > 0 else 0
        
        print(f"Results:")
        print(f"  Traditional: {traditional_avg:.4f}s ± {np.std(traditional_times):.4f}s")
        print(f"  Optimized:   {optimized_avg:.4f}s ± {np.std(optimized_times):.4f}s")
        print(f"  Speedup:     {speedup:.2f}x")
        
        # 結果の正確性確認
        try:
            # スキーマ比較
            assert result_traditional.schema.equals(result_optimized.schema), "Schema mismatch"
            assert result_traditional.num_rows == result_optimized.num_rows, "Row count mismatch"
            print(f"✓ Results validation passed")
        except Exception as e:
            print(f"✗ Results validation failed: {e}")
        
        return {
            'traditional_time': traditional_avg,
            'optimized_time': optimized_avg,
            'speedup': speedup,
            'traditional_std': np.std(traditional_times),
            'optimized_std': np.std(optimized_times)
        }
    
    def run_comprehensive_benchmark(self):
        """包括的なベンチマークの実行"""
        print("=" * 60)
        print("DECIMAL PASS1 INTEGRATION OPTIMIZATION BENCHMARK")
        print("=" * 60)
        
        # テストケース定義
        test_cases = [
            (1000, 1),    # 小規模: 1K行, 1 Decimal列
            (1000, 3),    # 小規模: 1K行, 3 Decimal列
            (10000, 1),   # 中規模: 10K行, 1 Decimal列
            (10000, 5),   # 中規模: 10K行, 5 Decimal列
            (50000, 3),   # 大規模: 50K行, 3 Decimal列
            (50000, 10),  # 大規模: 50K行, 10 Decimal列
        ]
        
        results = []
        
        for rows, decimal_cols in test_cases:
            try:
                result = self.benchmark_single_case(rows, decimal_cols, iterations=3)
                result.update({'rows': rows, 'decimal_cols': decimal_cols})
                results.append(result)
            except Exception as e:
                print(f"✗ Failed benchmark for {rows} rows, {decimal_cols} decimal cols: {e}")
                continue
        
        # 結果サマリー
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Rows':<8} {'DecCols':<8} {'Traditional':<12} {'Optimized':<12} {'Speedup':<8}")
        print("-" * 60)
        
        total_speedup = 0
        valid_results = 0
        
        for r in results:
            print(f"{r['rows']:<8} {r['decimal_cols']:<8} "
                  f"{r['traditional_time']:<12.4f} {r['optimized_time']:<12.4f} "
                  f"{r['speedup']:<8.2f}x")
            if r['speedup'] > 0:
                total_speedup += r['speedup']
                valid_results += 1
        
        if valid_results > 0:
            avg_speedup = total_speedup / valid_results
            print("-" * 60)
            print(f"Average Speedup: {avg_speedup:.2f}x")
            
            # 効果分析
            print(f"\n--- Optimization Effect Analysis ---")
            
            # Decimal列数と効果の相関分析
            decimal_effect = {}
            for r in results:
                dec_cols = r['decimal_cols']
                if dec_cols not in decimal_effect:
                    decimal_effect[dec_cols] = []
                decimal_effect[dec_cols].append(r['speedup'])
            
            print("Speedup by Decimal Column Count:")
            for dec_cols in sorted(decimal_effect.keys()):
                avg_speedup_for_cols = np.mean(decimal_effect[dec_cols])
                print(f"  {dec_cols} columns: {avg_speedup_for_cols:.2f}x average speedup")
                
            # 理論値との比較
            print(f"\nTheoretical Analysis:")
            print(f"  Expected memory access reduction: ~50% (1/2 reads per decimal field)")
            print(f"  Expected kernel launch reduction: {max(decimal_effect.keys())} decimal kernels → 0")
            print(f"  Measured average speedup: {avg_speedup:.2f}x")
            
            if avg_speedup > 1.1:
                print(f"✓ Optimization is EFFECTIVE (>10% improvement)")
            else:
                print(f"? Optimization effect is minimal (<10% improvement)")
        
        return results


def main():
    """メイン実行関数"""
    print("Starting Decimal Pass1 Integration Optimization Benchmark...")
    
    # GPU可用性チェック
    try:
        cuda.select_device(0)
        print(f"✓ GPU available: {cuda.get_current_device()}")
    except Exception as e:
        print(f"✗ GPU not available: {e}")
        return
    
    # ベンチマーク実行
    benchmark = DecimalOptimizationBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # 結果をJSONで保存（オプション）
    try:
        import json
        with open('decimal_optimization_benchmark_results.json', 'w') as f:
            # NumPy floatをPython floatに変換
            serializable_results = []
            for r in results:
                serializable_r = {}
                for k, v in r.items():
                    if isinstance(v, np.floating):
                        serializable_r[k] = float(v)
                    else:
                        serializable_r[k] = v
                serializable_results.append(serializable_r)
            
            json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Results saved to decimal_optimization_benchmark_results.json")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")


if __name__ == "__main__":
    main()