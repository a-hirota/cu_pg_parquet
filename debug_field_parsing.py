"""フィールド解析デバッグツール

大規模データで奇数行の文字列が破損する原因を調査
"""

import numpy as np
from numba import cuda, int32
import cupy as cp

@cuda.jit
def debug_extract_fields(raw, roff, foff, flen, debug_info, nrow, ncol, target_rows):
    """フィールド抽出のデバッグ版"""
    rid = cuda.grid(1)
    if rid >= nrow:
        return
    
    # デバッグ対象行のみ詳細情報を記録
    is_debug_row = False
    for i in range(len(target_rows)):
        if rid == target_rows[i]:
            is_debug_row = True
            break
    
    pos = int32(roff[rid])
    
    # フィールド数を読み取り（16ビット）
    if pos + 1 >= raw.size:
        if is_debug_row:
            debug_info[rid, 0] = -999  # エラーコード
        return
        
    nc = (raw[pos] << 8) | raw[pos+1]
    
    if is_debug_row:
        debug_info[rid, 0] = nc  # フィールド数
        debug_info[rid, 1] = pos  # 開始位置
    
    # 行終端チェック（0xFFFF = -1 as 16-bit signed）
    if nc == 0xFFFF:  # 65535 as unsigned, -1 as signed
        if is_debug_row:
            debug_info[rid, 2] = 0xFFFF  # 行終端マーカー検出
        return
    
    pos += 2
    
    for c in range(ncol):
        if c >= nc:
            flen[rid,c] = -1
            foff[rid,c] = 0
            continue
            
        # フィールド長を読み取り（32ビット）
        if pos + 3 >= raw.size:
            if is_debug_row and c < 10:  # 最初の10フィールドまで記録
                debug_info[rid, 3 + c*3] = -998  # エラーコード
            break
            
        # 各バイトを個別に読み取って検証
        b0 = int32(raw[pos])
        b1 = int32(raw[pos+1]) 
        b2 = int32(raw[pos+2])
        b3 = int32(raw[pos+3])
        
        ln = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        
        if is_debug_row and c < 10:  # 最初の10フィールドまで記録
            debug_info[rid, 3 + c*3] = ln      # フィールド長
            debug_info[rid, 3 + c*3 + 1] = pos  # 読み取り位置
            # 16ビット値として解釈した場合の値も記録
            as_16bit = (b2 << 8) | b3
            debug_info[rid, 3 + c*3 + 2] = as_16bit
        
        if ln == 0xFFFFFFFF:  # NULL
            flen[rid,c] = -1
            foff[rid,c] = 0
            pos += 4
        else:
            # 異常な値のチェック
            if ln < 0 or ln > 1000000:
                if is_debug_row:
                    debug_info[rid, 2] = ln  # 異常値を記録
            
            foff[rid,c] = pos + 4
            flen[rid,c] = ln
            pos += 4 + ln


def analyze_field_parsing(raw_dev, row_offsets, columns, sample_rows=100):
    """フィールド解析の問題を調査"""
    
    nrows = len(row_offsets)
    ncols = len(columns)
    
    # デバッグ対象行（最初の偶数行と奇数行）
    target_rows = []
    for i in range(min(sample_rows, nrows)):
        if i % 2 == 0:  # 偶数行
            target_rows.append(i)
        if i % 2 == 1 and len(target_rows) < sample_rows:  # 奇数行
            target_rows.append(i)
    
    target_rows_dev = cuda.to_device(np.array(target_rows, dtype=np.int32))
    
    # デバッグ情報格納用（行ごとに最大33要素）
    debug_info = cuda.device_array((nrows, 33), dtype=np.int32)
    field_offsets = cuda.device_array((nrows, ncols), dtype=np.int32)
    field_lengths = cuda.device_array((nrows, ncols), dtype=np.int32)
    
    # カーネル実行
    threads = 256
    blocks = (nrows + threads - 1) // threads
    
    debug_extract_fields[blocks, threads](
        raw_dev, row_offsets, field_offsets, field_lengths,
        debug_info, nrows, ncols, target_rows_dev
    )
    cuda.synchronize()
    
    # 結果分析
    debug_host = debug_info.copy_to_host()
    field_lengths_host = field_lengths.copy_to_host()
    
    print("=== フィールド解析デバッグ ===")
    
    # 偶数行と奇数行の比較
    even_rows = []
    odd_rows = []
    
    for i in target_rows[:20]:  # 最初の20行を詳細表示
        info = debug_host[i]
        if info[0] == 0:  # 未処理
            continue
            
        print(f"\n行{i} ({'偶数' if i % 2 == 0 else '奇数'}):")
        print(f"  フィールド数: {info[0]}")
        print(f"  開始位置: {info[1]}")
        
        if info[0] == 0xFFFF:
            print("  ⚠️ 行終端マーカー(0xFFFF)検出！")
            
        if info[2] != 0 and info[2] != 0xFFFF:
            print(f"  ⚠️ 異常値検出: {info[2]}")
        
        # 最初の3フィールドの詳細
        for f in range(min(3, ncols)):
            idx = 3 + f * 3
            if info[idx] != 0:
                print(f"  フィールド{f}:")
                print(f"    長さ(32bit): {info[idx]}")
                print(f"    位置: {info[idx+1]}")
                print(f"    16bit解釈: {info[idx+2]}")
                if info[idx+2] == 0xFFFF:
                    print("      ⚠️ 16ビットで0xFFFF！")
        
        # 文字列フィールドの長さを記録
        if i % 2 == 0:
            even_rows.append(field_lengths_host[i])
        else:
            odd_rows.append(field_lengths_host[i])
    
    # 統計
    print("\n=== 統計情報 ===")
    if even_rows and odd_rows:
        even_lens = [row[0] for row in even_rows if row[0] > 0]  # 最初の列
        odd_lens = [row[0] for row in odd_rows if row[0] > 0]
        
        if even_lens and odd_lens:
            print(f"偶数行の平均フィールド長: {np.mean(even_lens):.2f}")
            print(f"奇数行の平均フィールド長: {np.mean(odd_lens):.2f}")
            
            # 異常値チェック
            if max(odd_lens) > 1000:
                print(f"⚠️ 奇数行に異常な長さ: {max(odd_lens)}")
            if max(even_lens) > 1000:
                print(f"⚠️ 偶数行に異常な長さ: {max(even_lens)}")
    
    return debug_info, field_offsets, field_lengths


if __name__ == "__main__":
    # テスト用：実際のデータで実行する場合はここを変更
    print("このスクリプトは他のスクリプトからインポートして使用してください")
    print("使用例:")
    print("from debug_field_parsing import analyze_field_parsing")
    print("debug_info, field_offsets, field_lengths = analyze_field_parsing(raw_dev, row_offsets, columns)")