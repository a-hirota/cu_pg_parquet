#!/usr/bin/env python3
"""
実際のPostgreSQLデータでのGPGPU処理テスト

PostgreSQL COPY BINARY形式でデータを取得し、既存のGPGPUパーサーで処理する。
kvikio/cuDF不足環境での実際データテスト。
"""

import os
import io
import numpy as np
import cupy as cp
from numba import cuda
import psycopg

print("=== 実際のPostgreSQLデータ GPGPU処理テスト ===")

def fetch_postgresql_binary_data(table_name='gpuparser_test', limit=100):
    """PostgreSQL COPY BINARY形式でデータを取得"""
    print(f"📊 PostgreSQL COPY BINARYデータ取得: {table_name}")
    
    try:
        dsn = os.environ.get('GPUPASER_PG_DSN')
        conn = psycopg.connect(dsn)
        
        # COPY BINARY形式でデータ取得
        query = f"COPY (SELECT * FROM {table_name} LIMIT {limit}) TO STDOUT (FORMAT binary)"
        
        with conn.cursor() as cur:
            # バイナリデータをメモリバッファに取得
            buffer = io.BytesIO()
            with cur.copy(query) as copy:
                for data in copy:
                    buffer.write(data)
            
            binary_data = buffer.getvalue()
            buffer.close()
        
        conn.close()
        
        print(f"✓ バイナリデータ取得完了: {len(binary_data)} bytes")
        print(f"✓ 先頭20バイト: {binary_data[:20].hex()}")
        
        return binary_data
        
    except Exception as e:
        print(f"❌ PostgreSQLデータ取得エラー: {e}")
        raise

def detect_pg_binary_header_gpu(binary_data):
    """PostgreSQL BINARYヘッダー検出（GPU版）"""
    print("\n🔍 PostgreSQL BINARYヘッダー解析")
    
    # バイナリデータをGPU配列に転送
    data_host = np.frombuffer(binary_data, dtype=np.uint8)
    data_gpu = cuda.to_device(data_host)
    
    print(f"✓ GPU転送完了: {data_gpu.shape} shape")
    
    # ヘッダー解析カーネル
    @cuda.jit
    def analyze_binary_header(data, header_info_out):
        """PostgreSQL BINARYヘッダー解析カーネル"""
        idx = cuda.threadIdx.x
        
        if idx == 0 and data.size >= 19:
            # PostgreSQL BINARYヘッダー（19バイト）
            # - Magic: "PGCOPY\\n\\377\\r\\n\\0" (11バイト)
            # - Flags: 4バイト
            # - Extension: 4バイト
            
            # Magic署名確認（配列を使用せずに直接比較）
            magic_ok = (
                data[0] == 0x50 and data[1] == 0x47 and data[2] == 0x43 and data[3] == 0x4F and
                data[4] == 0x50 and data[5] == 0x59 and data[6] == 0x0A and data[7] == 0xFF and
                data[8] == 0x0D and data[9] == 0x0A and data[10] == 0x00
            )
            
            header_info_out[0] = 1 if magic_ok else 0  # Magic OK
            
            if data.size >= 19:
                # Flags (4バイト, ビッグエンディアン)
                flags = (data[11] << 24) | (data[12] << 16) | (data[13] << 8) | data[14]
                header_info_out[1] = flags
                
                # Extension (4バイト, ビッグエンディアン)  
                extension = (data[15] << 24) | (data[16] << 16) | (data[17] << 8) | data[18]
                header_info_out[2] = extension
    
    # ヘッダー情報出力配列
    header_info = cuda.device_array(3, dtype=np.uint32)
    
    # カーネル実行
    analyze_binary_header[1, 1](data_gpu, header_info)
    cuda.synchronize()
    
    # 結果確認
    header_result = header_info.copy_to_host()
    magic_ok = header_result[0] == 1
    flags = header_result[1]
    extension = header_result[2]
    
    print(f"✓ Magic署名: {'OK' if magic_ok else 'NG'}")
    print(f"✓ Flags: 0x{flags:08X}")
    print(f"✓ Extension: {extension}")
    
    if magic_ok:
        return 19  # ヘッダーサイズ
    else:
        print("⚠️  PostgreSQL BINARYヘッダーが検出されませんでした")
        return 0

def parse_binary_rows_gpu(binary_data, header_size):
    """PostgreSQL BINARYデータの行解析（GPU版）"""
    print(f"\n⚙️  PostgreSQL BINARY行解析 (header_size={header_size})")
    
    # データ部のみ抽出
    data_part = binary_data[header_size:]
    data_host = np.frombuffer(data_part, dtype=np.uint8)
    data_gpu = cuda.to_device(data_host)
    
    print(f"✓ データ部GPU転送: {data_gpu.shape} shape")
    
    # 行解析カーネル
    @cuda.jit
    def parse_binary_rows(data, row_info_out):
        """PostgreSQL BINARY行解析カーネル"""
        idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        
        if idx == 0:
            # 簡易行カウント（フィールド数を数える）
            offset = 0
            row_count = 0
            
            while offset + 2 < data.size and row_count < 1000:
                # フィールド数（2バイト, ビッグエンディアン）
                if offset + 1 < data.size:
                    field_count = (data[offset] << 8) | data[offset + 1]
                    offset += 2
                    
                    if field_count == 0xFFFF:  # 終了マーカー
                        break
                    
                    if field_count > 0 and field_count <= 100:  # 妥当なフィールド数
                        # 各フィールドの長さを読み飛ばし
                        for field in range(field_count):
                            if offset + 4 <= data.size:
                                # フィールド長（4バイト, ビッグエンディアン）
                                field_len = ((data[offset] << 24) | 
                                           (data[offset + 1] << 16) | 
                                           (data[offset + 2] << 8) | 
                                           data[offset + 3])
                                offset += 4
                                
                                if field_len == 0xFFFFFFFF:  # NULL
                                    continue
                                elif field_len > 0 and field_len < 10000:
                                    offset += field_len  # フィールドデータをスキップ
                                else:
                                    # 不正なフィールド長
                                    break
                            else:
                                break
                        
                        row_count += 1
                    else:
                        break
                else:
                    break
            
            row_info_out[0] = row_count
            row_info_out[1] = offset  # 処理したバイト数
    
    # 行情報出力配列
    row_info = cuda.device_array(2, dtype=np.uint32)
    
    # カーネル実行
    threads_per_block = 256
    blocks = 1
    
    import time
    start_time = time.time()
    parse_binary_rows[blocks, threads_per_block](data_gpu, row_info)
    cuda.synchronize()
    parse_time = time.time() - start_time
    
    # 結果確認
    row_result = row_info.copy_to_host()
    row_count = row_result[0]
    processed_bytes = row_result[1]
    
    print(f"✓ 解析完了時間: {parse_time*1000:.3f} ms")
    print(f"✓ 検出行数: {row_count}")
    print(f"✓ 処理バイト数: {processed_bytes:,}")
    print(f"✓ 解析スループット: {(len(data_part) / (1024*1024)) / parse_time:.1f} MB/sec")
    
    return row_count, processed_bytes

def test_real_postgresql_gpgpu():
    """実際のPostgreSQLデータでのGPGPU処理統合テスト"""
    print("\n🚀 実際のPostgreSQLデータ GPGPU統合テスト")
    
    try:
        # ステップ1: PostgreSQLからバイナリデータ取得
        binary_data = fetch_postgresql_binary_data('gpuparser_test', limit=500)
        
        # ステップ2: GPUヘッダー解析
        header_size = detect_pg_binary_header_gpu(binary_data)
        
        if header_size == 0:
            print("❌ PostgreSQL BINARYヘッダー解析失敗")
            return False
        
        # ステップ3: GPU行解析
        row_count, processed_bytes = parse_binary_rows_gpu(binary_data, header_size)
        
        # ステップ4: 結果検証
        print(f"\n📊 総合結果:")
        print(f"  バイナリデータサイズ: {len(binary_data):,} bytes")
        print(f"  ヘッダーサイズ: {header_size} bytes") 
        print(f"  データ部サイズ: {len(binary_data) - header_size:,} bytes")
        print(f"  検出行数: {row_count}")
        print(f"  処理バイト数: {processed_bytes:,}")
        
        # 期待値確認（500行取得したので、それに近い値を期待）
        if 400 <= row_count <= 600:
            print("✅ 結果検証: 期待範囲内の行数を検出")
            performance_ok = True
        else:
            print(f"⚠️  結果異常: 期待範囲 400-600, 実際 {row_count}")
            performance_ok = False
        
        # パフォーマンス評価
        if processed_bytes > 0:
            efficiency = (processed_bytes / len(binary_data)) * 100
            print(f"  データ処理効率: {efficiency:.1f}%")
            
            if efficiency > 80:
                print("  🏆 高効率: GPGPU処理最適化成功")
            elif efficiency > 50:
                print("  🥇 中効率: 良好なGPU利用")
            else:
                print("  🥈 低効率: 改善余地あり")
        
        return performance_ok
        
    except Exception as e:
        print(f"❌ 統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    try:
        # CUDA初期化確認
        if not cuda.is_available():
            print("❌ CUDA not available")
            return
        
        device = cuda.current_context().device
        print(f"🚀 GPU: {device.name.decode()} (Compute {device.compute_capability})")
        
        # 実際のPostgreSQLデータテスト実行
        success = test_real_postgresql_gpgpu()
        
        if success:
            print("\n🎉 実際のPostgreSQLデータ GPGPU処理テスト完全成功!")
            print("   → PostgreSQL→GPU直接処理パイプライン動作確認完了")
        else:
            print("\n⚠️  一部テストで問題が発生しました")
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()