#!/usr/bin/env python3
"""
最後のスレッドのデバッグ情報を解析
"""

def analyze_last_threads_debug(debug_info):
    """最後のスレッドのデバッグ情報を解析"""
    
    if debug_info is None or len(debug_info) == 0:
        print("デバッグ情報なし")
        return
    
    print("\n=== 最後の10スレッドの処理状況 ===")
    print("（インデックス10-19の範囲に記録）")
    
    # デバッグ情報の解釈
    # debug_info[i, 10]: Thread ID
    # debug_info[i, 11]: 最後から何番目
    # debug_info[i, 12]: 開始位置
    # debug_info[i, 13]: 終了位置
    # debug_info[i, 14]: データサイズ
    # debug_info[i, 15]: thread_stride
    # debug_info[i, 16]: 処理行数
    # debug_info[i, 17]: 最後の行位置
    # debug_info[i, 18]: データ終端検出位置
    # debug_info[i, 19]: 総スレッド数
    
    for i in range(len(debug_info)):
        row = debug_info[i]
        
        # 最後のスレッド情報かチェック（Thread IDが10に記録されている）
        if row[10] > 0:  # Thread IDが記録されている
            tid = int(row[10])
            threads_from_end = int(row[11])
            start_pos = int(row[12])
            end_pos = int(row[13])
            data_size = int(row[14])
            thread_stride = int(row[15])
            processed_rows = int(row[16])
            last_row_pos = int(row[17]) if row[17] != -1 else None
            data_end_pos = int(row[18]) if row[18] != -1 else None
            total_threads = int(row[19])
            
            # 最後から10番目以内のスレッドのみ表示
            if threads_from_end >= 10:
                continue
            print(f"\nスレッド {tid} (最後から{threads_from_end}番目):")
            print(f"  担当範囲: {start_pos:,} - {end_pos:,} ({end_pos-start_pos:,} bytes)")
            print(f"  データサイズ: {data_size:,} bytes")
            print(f"  thread_stride: {thread_stride:,} bytes")
            print(f"  処理行数: {processed_rows}")
            
            if last_row_pos is not None:
                print(f"  最後の行位置: {last_row_pos:,}")
                distance_to_end = end_pos - last_row_pos
                print(f"  end_posまでの距離: {distance_to_end:,} bytes")
            
            if data_end_pos is not None:
                print(f"  データ終端検出位置: {data_end_pos:,}")
                remaining = data_size - data_end_pos
                print(f"  残りデータ: {remaining:,} bytes")
            
            # 問題の可能性をチェック
            if end_pos > data_size:
                print(f"  ⚠️ 担当範囲がデータサイズを超過")
            
            if processed_rows == 0 and start_pos < data_size:
                print(f"  ⚠️ データ範囲内だが行を処理していない")
            
            if last_row_pos and end_pos - last_row_pos > 500:
                print(f"  ⚠️ 最後の行からend_posまでが離れすぎ（{end_pos - last_row_pos}バイト）")

if __name__ == "__main__":
    # テスト用のダミーデータ
    import numpy as np
    test_data = np.zeros((10, 100), dtype=np.int64)
    
    # サンプルデータ
    test_data[0, 10] = 1679359  # Thread ID
    test_data[0, 11] = 0  # 最後から0番目
    test_data[0, 12] = 8589930000  # 開始位置
    test_data[0, 13] = 8589934592  # 終了位置
    test_data[0, 14] = 8589934592  # データサイズ
    test_data[0, 15] = 5116  # thread_stride
    test_data[0, 16] = 0  # 処理行数
    test_data[0, 17] = -1  # 最後の行位置
    test_data[0, 18] = 8589934590  # データ終端検出位置
    test_data[0, 19] = 1679360  # 総スレッド数
    
    analyze_last_threads_debug(test_data)