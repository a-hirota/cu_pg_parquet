#!/usr/bin/env python3
"""
全スレッドメモリダンプ分析ツール
===============================

目的: 46,862行 vs 50,000行の差異の根本原因解析
方法: 全スレッドの処理状況、メモリアクセスパターン、境界条件を詳細ダンプ
"""

import os
import sys
import numpy as np
from numba import cuda

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.types import ColumnMeta, INT32, UTF8

def create_debug_test_data():
    """デバッグ用小規模テストデータ（100行）"""
    
    # ヘッダ（19バイト）
    header = bytearray(19)
    header[:11] = b"PGCOPY\n\xff\r\n\x00"  # COPY signature
    header[11:15] = (0).to_bytes(4, 'big')  # flags
    header[15:19] = (0).to_bytes(4, 'big')  # header extension length
    
    # データ部（正確に100行生成）
    data = bytearray()
    ncols = 3  # シンプルな3列
    
    for row_id in range(100):
        # 行ヘッダ: フィールド数（2バイト）
        data.extend(ncols.to_bytes(2, 'big'))
        
        # フィールド1: INT32（固定4バイト）
        data.extend((4).to_bytes(4, 'big'))
        data.extend(row_id.to_bytes(4, 'big'))
        
        # フィールド2: INT32（固定4バイト）
        data.extend((4).to_bytes(4, 'big'))
        data.extend((row_id * 2).to_bytes(4, 'big'))
        
        # フィールド3: 文字列（可変長）
        if row_id % 10 == 9:  # 10行に1回NULL
            # NULL値
            data.extend((0xFFFFFFFF).to_bytes(4, 'big'))
        else:
            # 通常の文字列
            field_data = f"ROW{row_id:03d}".encode('utf-8')
            data.extend(len(field_data).to_bytes(4, 'big'))
            data.extend(field_data)
    
    # PostgreSQL終端マーカー追加
    data.extend((0xFFFF).to_bytes(2, 'big'))
    
    return bytes(header + data)

def create_debug_columns():
    """デバッグ用3列定義"""
    return [
        ColumnMeta(name="id", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="value", pg_oid=23, pg_typmod=0, arrow_id=INT32, elem_size=4),
        ColumnMeta(name="text", pg_oid=25, pg_typmod=-1, arrow_id=UTF8, elem_size=-1),
    ]

@cuda.jit
def debug_memory_dump_kernel(
    raw_data,
    header_size,
    ncols,
    
    # デバッグ出力配列
    thread_info,        # int32[max_threads, 10] - スレッド情報
    memory_access_log,  # int32[max_threads, 1000] - メモリアクセスログ
    row_detection_log,  # int32[max_threads, 500] - 行検出ログ
    
    # 設定
    thread_stride,
    max_threads
):
    """
    全スレッドメモリダンプカーネル
    
    各スレッドの処理状況を詳細記録:
    - 担当範囲
    - メモリアクセスパターン
    - 行検出状況
    - 境界条件
    """
    
    # スレッド・ブロック情報
    tid = cuda.blockIdx.x * cuda.gridDim.y * cuda.blockDim.x + \
          cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    
    if tid >= max_threads:
        return
    
    # 担当範囲計算
    start_pos = header_size + tid * thread_stride
    end_pos = header_size + (tid + 1) * thread_stride
    
    # スレッド情報記録
    thread_info[tid, 0] = tid              # スレッドID
    thread_info[tid, 1] = start_pos        # 開始位置
    thread_info[tid, 2] = end_pos          # 終了位置
    thread_info[tid, 3] = thread_stride    # ストライド
    thread_info[tid, 4] = raw_data.size    # データサイズ
    thread_info[tid, 5] = 0                # 検出行数（後で更新）
    thread_info[tid, 6] = 0                # メモリアクセス回数
    thread_info[tid, 7] = 0                # エラー回数
    thread_info[tid, 8] = 0                # 境界越え回数
    thread_info[tid, 9] = 0                # 状態フラグ
    
    if start_pos >= raw_data.size:
        thread_info[tid, 9] = -1  # 範囲外
        return
    
    memory_idx = 0
    row_idx = 0
    detected_rows = 0
    pos = start_pos
    
    # **詳細メモリアクセスループ**
    while pos < end_pos and memory_idx < 1000 and row_idx < 500:
        
        # メモリアクセス記録
        if memory_idx < 999:
            memory_access_log[tid, memory_idx] = pos
            memory_access_log[tid, memory_idx + 1] = -1  # 区切り
            memory_idx += 2
        
        thread_info[tid, 6] += 1  # アクセス回数
        
        # 境界チェック
        if pos + 1 >= raw_data.size:
            thread_info[tid, 8] += 1  # 境界越え
            break
        
        if pos + 1 >= end_pos:
            thread_info[tid, 8] += 1  # 担当範囲越え
            break
        
        # 行ヘッダ検出（簡易版）
        num_fields = (raw_data[pos] << 8) | raw_data[pos + 1]
        
        if num_fields == ncols:
            # 行ヘッダ候補発見
            if row_idx < 498:
                row_detection_log[tid, row_idx] = pos      # 候補位置
                row_detection_log[tid, row_idx + 1] = 1    # 候補フラグ
                row_idx += 2
            
            # 簡易検証（完全版は重いのでスキップ）
            if pos + 2 + (ncols * 8) < raw_data.size:  # 最小行サイズチェック
                detected_rows += 1
                if row_idx < 498:
                    row_detection_log[tid, row_idx] = pos      # 確定位置
                    row_detection_log[tid, row_idx + 1] = 2    # 確定フラグ
                    row_idx += 2
                
                # 次の行へジャンプ（推定）
                pos += 30  # 推定行サイズ
            else:
                thread_info[tid, 7] += 1  # エラー
                pos += 1
        else:
            # 行ヘッダでない
            pos += 1
    
    # 最終結果記録
    thread_info[tid, 5] = detected_rows
    thread_info[tid, 9] = 1  # 正常終了

def analyze_memory_dump(test_data):
    """メモリダンプ分析実行"""
    
    print("🔍 全スレッドメモリダンプ分析")
    print("="*60)
    
    # GPU利用可能性確認
    if not cuda.is_available():
        print("❌ CUDAが利用できません")
        return False
    
    print(f"🔧 GPU: {cuda.get_current_device().name}")
    print(f"📝 テストデータ: {len(test_data)}B (期待行数: 100行)")
    
    columns = create_debug_columns()
    ncols = len(columns)
    header_size = 19
    data_size = len(test_data) - header_size
    
    # GPU メモリにデータ転送
    raw_dev = cuda.to_device(np.frombuffer(test_data, dtype=np.uint8))
    
    # グリッド設定（小規模）
    threads_per_block = 32
    blocks_x = 4
    blocks_y = 1
    max_threads = blocks_x * blocks_y * threads_per_block
    
    thread_stride = (data_size + max_threads - 1) // max_threads
    if thread_stride < 30:  # 最小行サイズ
        thread_stride = 30
    
    print(f"🔧 グリッド設定: ({blocks_x}, {blocks_y}) × {threads_per_block} = {max_threads}スレッド")
    print(f"🔧 スレッドストライド: {thread_stride}B")
    
    # デバッグ出力配列
    thread_info = cuda.device_array((max_threads, 10), np.int32)
    memory_access_log = cuda.device_array((max_threads, 1000), np.int32)
    row_detection_log = cuda.device_array((max_threads, 500), np.int32)
    
    # 配列初期化
    thread_info[:] = -999
    memory_access_log[:] = -999
    row_detection_log[:] = -999
    
    # デバッグカーネル実行
    grid_2d = (blocks_x, blocks_y)
    debug_memory_dump_kernel[grid_2d, threads_per_block](
        raw_dev, header_size, ncols,
        thread_info, memory_access_log, row_detection_log,
        thread_stride, max_threads
    )
    cuda.synchronize()
    
    # 結果取得
    thread_info_host = thread_info.copy_to_host()
    memory_access_host = memory_access_log.copy_to_host()
    row_detection_host = row_detection_log.copy_to_host()
    
    print(f"\n📊 スレッド処理結果分析")
    print("="*60)
    
    total_detected = 0
    active_threads = 0
    
    for tid in range(max_threads):
        info = thread_info_host[tid]
        
        if info[0] == -999:  # 未実行
            continue
        
        active_threads += 1
        detected_rows = info[5]
        total_detected += detected_rows
        
        print(f"\nスレッド {info[0]:3d}:")
        print(f"  範囲: {info[1]:6d} - {info[2]:6d} (ストライド: {info[3]:4d})")
        print(f"  検出行数: {detected_rows:3d}")
        print(f"  メモリアクセス: {info[6]:4d}回")
        print(f"  エラー: {info[7]:3d}回")
        print(f"  境界越え: {info[8]:3d}回")
        print(f"  状態: {info[9]:2d} (1=正常, -1=範囲外)")
        
        # 行検出詳細
        row_log = row_detection_host[tid]
        candidates = 0
        confirmed = 0
        
        for i in range(0, 500, 2):
            if row_log[i] == -999:
                break
            if row_log[i+1] == 1:  # 候補
                candidates += 1
            elif row_log[i+1] == 2:  # 確定
                confirmed += 1
        
        if candidates > 0 or confirmed > 0:
            print(f"  行検出: 候補{candidates}個, 確定{confirmed}個")
    
    print(f"\n📈 総合結果:")
    print(f"  アクティブスレッド: {active_threads}/{max_threads}")
    print(f"  総検出行数: {total_detected}")
    print(f"  期待行数: 100")
    print(f"  検出率: {total_detected/100*100:.1f}%")
    
    # スレッド境界分析
    print(f"\n🔍 スレッド境界分析:")
    overlaps = 0
    gaps = 0
    
    for tid in range(max_threads - 1):
        current_end = thread_info_host[tid][2]
        next_start = thread_info_host[tid + 1][1]
        
        if next_start < current_end:
            overlaps += 1
            print(f"  オーバーラップ: スレッド{tid} - {tid+1} ({current_end - next_start}B重複)")
        elif next_start > current_end:
            gaps += 1
            print(f"  ギャップ: スレッド{tid} - {tid+1} ({next_start - current_end}B未処理)")
    
    print(f"  オーバーラップ: {overlaps}箇所")
    print(f"  ギャップ: {gaps}箇所")
    
    if total_detected < 100:
        print(f"\n❌ 行数不足の原因分析:")
        print(f"  1. スレッド境界でのギャップ: {gaps}箇所")
        print(f"  2. メモリアクセス境界エラー")
        print(f"  3. 行検証の失敗")
        
        # 詳細ギャップ分析
        total_gap = 0
        for tid in range(max_threads - 1):
            current_end = thread_info_host[tid][2]
            next_start = thread_info_host[tid + 1][1]
            if next_start > current_end:
                gap_size = next_start - current_end
                total_gap += gap_size
                
                # ギャップ内の推定行数
                estimated_rows_in_gap = gap_size // 30  # 30B/行と仮定
                print(f"    ギャップ{tid}: {gap_size}B (推定{estimated_rows_in_gap}行)")
        
        print(f"  総ギャップサイズ: {total_gap}B (推定{total_gap//30}行)")

def main():
    """メイン実行関数"""
    
    # デバッグ用小規模データで分析
    test_data = create_debug_test_data()
    analyze_memory_dump(test_data)

if __name__ == "__main__":
    main()