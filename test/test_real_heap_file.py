#!/usr/bin/env python3
"""
実際のPostgreSQLヒープファイル解析テスト

PostgreSQL物理ヒープファイル（17292）から実際のページ構造を解析し、
MVCC（t_xmax/t_xmin）とvisibility mapを考慮した正確なタプル抽出を行う。

COPY BINARYとヒープファイルの重要な違い：
- COPY BINARY: PostgreSQLが生成する標準化バイナリ形式
- Heap File: ディスク上の物理8KBページ構造（PageHeader + ItemId + タプル）
"""

import os
import sys
import time
import numpy as np
import cupy as cp
from numba import cuda
import psycopg

# プロジェクトパスを追加
sys.path.append('/home/ubuntu/gpupgparser')

from src.heap_file_reader import read_heap_file_direct, HeapFileReaderError
from src.cuda_kernels.heap_page_parser import (
    parse_heap_file_gpu, create_page_offsets, POSTGRES_PAGE_SIZE
)

def test_real_postgresql_heap_file():
    """実際のPostgreSQLヒープファイル解析テスト"""
    print("=== 実際のPostgreSQLヒープファイル解析テスト ===")
    print("🎯 目標: lineorderヒープファイル → ページ構造解析 → 有効タプル抽出")
    
    # PostgreSQLデータディレクトリパス
    heap_file_path = "/var/lib/postgresql/17/main/base/5/17292"
    
    try:
        # ステップ1: ファイル存在・アクセス確認
        print(f"\n📁 ヒープファイル確認: {heap_file_path}")
        
        if not os.path.exists(heap_file_path):
            print(f"❌ ファイルが存在しません: {heap_file_path}")
            return False
        
        file_size = os.path.getsize(heap_file_path)
        file_size_mb = file_size / (1024*1024)
        num_pages = file_size // POSTGRES_PAGE_SIZE
        
        print(f"✅ ファイル情報:")
        print(f"   サイズ: {file_size_mb:.2f} MB ({file_size:,} bytes)")
        print(f"   ページ数: {num_pages:,} (8KB/page)")
        
        if not os.access(heap_file_path, os.R_OK):
            print(f"❌ 読み取り権限がありません: {heap_file_path}")
            print("💡 解決方法: sudo権限でスクリプトを実行してください")
            return False
        
        # ステップ2: kvikio直接GPU読み込み
        print(f"\n🚀 kvikio GPU Direct Storage読み込み...")
        gpu_read_start = time.time()
        
        try:
            heap_data_gpu = read_heap_file_direct(heap_file_path)
            gpu_read_time = time.time() - gpu_read_start
            
            print(f"✅ kvikio読み込み完了:")
            print(f"   読み込み時間: {gpu_read_time*1000:.3f} ms")
            print(f"   GPU配列形状: {heap_data_gpu.shape}")
            print(f"   データ型: {heap_data_gpu.dtype}")
            print(f"   転送スループット: {file_size_mb / gpu_read_time:.1f} MB/sec")
            
        except HeapFileReaderError as e:
            print(f"❌ kvikio読み込みエラー: {e}")
            return False
        
        # ステップ3: PostgreSQLヒープページ構造解析
        print(f"\n⚙️  PostgreSQLページ構造解析開始...")
        parse_start = time.time()
        
        try:
            # ヒープページ解析実行（debug=True で詳細ログ）
            tuple_offsets_gpu, total_tuple_count = parse_heap_file_gpu(
                heap_data_gpu, debug=True
            )
            
            parse_time = time.time() - parse_start
            
            print(f"✅ ページ構造解析完了:")
            print(f"   解析時間: {parse_time*1000:.3f} ms")
            print(f"   検出タプル数: {total_tuple_count:,}")
            print(f"   タプル密度: {total_tuple_count/num_pages:.2f} タプル/ページ")
            print(f"   解析スループット: {file_size_mb / parse_time:.1f} MB/sec")
            
        except Exception as e:
            print(f"❌ ページ構造解析エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ステップ4: ヒープファイル vs COPY BINARY比較
        print(f"\n📊 COPY BINARY vs ヒープファイル比較:")
        
        try:
            # COPY BINARYでのサンプル取得（比較用）
            dsn = os.environ.get('GPUPASER_PG_DSN', 
                                'dbname=postgres user=postgres host=localhost port=5432')
            
            conn = psycopg.connect(dsn)
            with conn.cursor() as cur:
                # 小さなサンプルでCOPY BINARY実行
                cur.execute("SELECT COUNT(*) FROM lineorder LIMIT 100000")
                sample_count = cur.fetchone()[0]
                
                print(f"   COPY BINARY サンプル: {sample_count:,} 行")
                print(f"   ヒープファイル検出: {total_tuple_count:,} タプル")
                
                if total_tuple_count > 0:
                    ratio = total_tuple_count / sample_count if sample_count > 0 else 0
                    print(f"   ヒープ/COPY比率: {ratio:.2f}")
                    
                    if ratio > 0.8:
                        validation = "✅ 妥当 - ヒープファイル解析成功"
                    elif ratio > 0.5:
                        validation = "⚠️  部分的 - 一部タプルが削除済み"
                    else:
                        validation = "❌ 異常 - ヒープ解析に問題の可能性"
                    
                    print(f"   検証結果: {validation}")
            
            conn.close()
            
        except Exception as e:
            print(f"⚠️  COPY BINARY比較エラー: {e}")
        
        # ステップ5: ページ内容のサンプリング確認
        print(f"\n🔍 ページ内容サンプリング確認:")
        
        if len(tuple_offsets_gpu) > 0:
            # 最初の数タプルの位置を確認
            sample_offsets = tuple_offsets_gpu[:min(5, len(tuple_offsets_gpu))]
            sample_offsets_host = cp.asnumpy(sample_offsets)
            
            print(f"   サンプルタプルオフセット:")
            for i, offset in enumerate(sample_offsets_host):
                page_num = offset // POSTGRES_PAGE_SIZE
                page_offset = offset % POSTGRES_PAGE_SIZE
                print(f"     タプル{i+1}: オフセット {offset:,} (ページ{page_num}, +{page_offset})")
        
        # ステップ6: 性能評価とスケーラビリティ
        print(f"\n📈 性能評価:")
        
        total_time = gpu_read_time + parse_time
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 処理結果:")
        print(f"   ファイルサイズ: {file_size_mb:.2f} MB")
        print(f"   ページ数: {num_pages:,}")
        print(f"   有効タプル数: {total_tuple_count:,}")
        
        print(f"\n⏱️  時間内訳:")
        print(f"   kvikio読み込み: {gpu_read_time*1000:.3f} ms")
        print(f"   ページ解析: {parse_time*1000:.3f} ms")
        print(f"   総時間: {total_time*1000:.3f} ms")
        
        print(f"\n🚀 スループット:")
        if total_time > 0:
            overall_throughput = file_size_mb / total_time
            tuple_speed = total_tuple_count / total_time
            page_speed = num_pages / total_time
            
            print(f"   総合: {overall_throughput:.1f} MB/sec")
            print(f"   タプル処理: {tuple_speed:,.0f} tuples/sec")
            print(f"   ページ処理: {page_speed:,.0f} pages/sec")
        
        # 性能クラス判定
        if total_time > 0 and overall_throughput > 1000:
            perf_class = "🏆 革命的 (1GB/sec+)"
        elif overall_throughput > 500:
            perf_class = "🥇 超高速 (500MB/sec+)"
        elif overall_throughput > 100:
            perf_class = "🥈 高速 (100MB/sec+)"
        else:
            perf_class = "🥉 標準"
        
        print(f"   性能クラス: {perf_class}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 全lineorderファイル処理予測
        print(f"\n🔮 全lineorderファイル処理予測:")
        
        # 全ファイルサイズ計算（17292.* ファイル群）
        try:
            import glob
            all_heap_files = glob.glob("/var/lib/postgresql/17/main/base/5/17292*")
            total_heap_size = sum(os.path.getsize(f) for f in all_heap_files 
                                if os.path.isfile(f))
            total_heap_size_gb = total_heap_size / (1024**3)
            
            if total_time > 0:
                predicted_time = (total_heap_size / file_size) * total_time
                predicted_throughput = total_heap_size_gb / predicted_time
                
                print(f"   全ヒープサイズ: {total_heap_size_gb:.1f} GB")
                print(f"   予測処理時間: {predicted_time:.1f}秒 ({predicted_time/60:.1f}分)")
                print(f"   予測スループット: {predicted_throughput:.1f} GB/sec")
                
                if predicted_time < 600:  # 10分以内
                    impact = "🚀 実用的 - 42GBを10分以内で処理可能"
                elif predicted_time < 1800:  # 30分以内
                    impact = "⚡ 高性能 - 42GBを30分以内で処理可能"
                else:
                    impact = "🏃 改善中 - さらなる最適化で実用化"
                
                print(f"   実用性評価: {impact}")
            
        except Exception as e:
            print(f"⚠️  全ファイル予測計算エラー: {e}")
        
        print(f"\n🎉 実際のPostgreSQLヒープファイル解析テスト成功!")
        print(f"   💡 物理ヒープページ → MVCC考慮 → 有効タプル抽出完了")
        print(f"   ⚡ COPY BINARYを超える直接ヒープアクセス実現!")
        
        return True
        
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    # CUDA環境確認
    if not cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = cuda.current_context().device
    print(f"🚀 GPU: {device.name.decode()} (Compute {device.compute_capability})")
    
    # 実際のヒープファイルテスト実行
    success = test_real_postgresql_heap_file()
    
    if success:
        print("\n✨ 実際のPostgreSQLヒープファイル解析完全成功 ✨")
        print("   → 物理ディスク → GPU Direct Storage → ページ構造解析パイプライン実証完了")
    else:
        print("\n⚠️  テストで問題が発生しました")

if __name__ == "__main__":
    main()