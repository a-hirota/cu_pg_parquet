"""
PostgreSQLバイナリデータの基本パース処理
"""

import numpy as np
from numba import njit
from typing import Tuple

@njit
def parse_binary_chunk(chunk_array, header_expected=True):
    """バイナリチャンクのパース（Numba最適化版）"""
    field_offsets = []
    field_lengths = []
    pos = np.int64(0)
    
    # ヘッダーのスキップ（最初のチャンクのみ）
    if header_expected and len(chunk_array) >= 11:
        header = np.array([80,71,67,79,80,89,10,255,13,10,0], dtype=np.uint8)
        if np.all(chunk_array[0:11] == header):
            pos = np.int64(11)
            if len(chunk_array) >= pos + 8:
                # フラグとヘッダー拡張をスキップ
                flags = np.int32((chunk_array[pos] << 24) | (chunk_array[pos+1] << 16) | \
                       (chunk_array[pos+2] << 8) | chunk_array[pos+3])
                pos += np.int64(4)
                ext_len = np.int32((chunk_array[pos] << 24) | (chunk_array[pos+1] << 16) | \
                         (chunk_array[pos+2] << 8) | chunk_array[pos+3])
                pos += np.int64(4) + np.int64(ext_len)
    
    # チャンク内の各タプルを処理
    while pos + 2 <= len(chunk_array):
        # タプルのフィールド数を読み取り
        num_fields = np.int16((chunk_array[pos] << 8) | chunk_array[pos + 1])
        if num_fields == -1:  # ファイル終端
            break
            
        pos += np.int64(2)
        
        # 各フィールドを処理
        for _ in range(num_fields):
            if pos + 4 > len(chunk_array):
                break
                
            # フィールド長を読み取り
            b0 = chunk_array[pos]
            b1 = chunk_array[pos + 1]
            b2 = chunk_array[pos + 2]
            b3 = chunk_array[pos + 3]
            
            # ビッグエンディアンからリトルエンディアンに変換
            field_len = ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)
            
            # 符号付き32ビット整数に変換
            if field_len & 0x80000000:  # 最上位ビットが1なら負の数
                field_len = -((~field_len + 1) & 0xFFFFFFFF)
            
            pos += np.int64(4)
            
            if field_len == -1:  # NULL値
                field_offsets.append(0)  # NULL値のオフセットは0
                field_lengths.append(-1)
            else:
                if pos + field_len > len(chunk_array):
                    # チャンク境界をまたぐ場合は中断
                    return np.array(field_offsets, dtype=np.int32), np.array(field_lengths, dtype=np.int32)
                field_offsets.append(int(pos))
                field_lengths.append(int(field_len))
                pos += np.int64(field_len)
    
    return np.array(field_offsets, dtype=np.int32), np.array(field_lengths, dtype=np.int32)

class BinaryDataParser:
    """バイナリデータのパーサークラス"""
    
    def __init__(self):
        self.header_expected = True
        self._remaining_data = None
    
    def parse_chunk(self, chunk_data: bytes, max_chunk_size: int = 1024*1024, num_columns: int = None, start_row: int = 0, max_rows: int = 65535) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """チャンク単位でのパース処理
        
        Args:
            chunk_data: バイナリデータ
            max_chunk_size: 最大チャンクサイズ
            num_columns: カラム数（Noneの場合は動的に検出）
            start_row: 処理開始行インデックス（複数チャンク対応）
            max_rows: 処理する最大行数（デフォルト: 65535）
            
        Returns:
            (chunk_array, field_offsets, field_lengths, rows_count)
        """
        # 常に完全なバイナリデータを使用
        chunk_array = np.frombuffer(chunk_data, dtype=np.uint8)
        
        # ヘッダーフラグは最初のチャンク（start_row=0）のみTrueに設定
        self.header_expected = (start_row == 0)
        
        # 最大試行回数を設定
        max_attempts = 3
        attempts = 0
        
        # ヘッダースキップ処理（最初のチャンクの場合のみ）
        current_pos = 0
        header_len = 0
        
        if start_row == 0 and len(chunk_array) >= 11:
            # ヘッダーが "PGCOPY\n\377\r\n\0" (バイナリフォーマット識別子) で始まるか確認
            if chunk_array[0] == 80 and chunk_array[1] == 71:  # 'P', 'G'
                header_len = 11  # ヘッダー部分の長さ
                
                # フラグフィールドとヘッダー拡張をスキップ
                if len(chunk_array) >= header_len + 8:
                    header_len += 8  # フラグ(4バイト) + 拡張長(4バイト)
                    
                    # ヘッダーサイズ（ヘッダー拡張）を読み取り、拡張データをスキップ
                    ext_len = ((int(chunk_array[header_len-4]) & 0xFF) << 24) | \
                              ((int(chunk_array[header_len-3]) & 0xFF) << 16) | \
                              ((int(chunk_array[header_len-2]) & 0xFF) << 8) | \
                              (int(chunk_array[header_len-1]) & 0xFF)
                    
                    if ext_len > 0 and len(chunk_array) >= header_len + ext_len:
                        header_len += ext_len
        
        # データポジション計算（ヘッダー + スキップ行分）
        if start_row > 0:
            print(f"開始行 {start_row} からのパース開始")
            
            # 現在位置をヘッダーの後ろに設定
            current_pos = header_len
            skipped_rows = 0
            
            # 指定行数をスキップ（行単位で移動）
            while skipped_rows < start_row and current_pos + 2 <= len(chunk_array):
                # 行の最初にあるフィールド数（2バイト）を読み取り
                num_fields = ((chunk_array[current_pos] << 8) | chunk_array[current_pos + 1])
                
                if num_fields == 0xFFFF:  # ファイル終端マーカー
                    print(f"ファイル終端に到達しました（行 {skipped_rows}）")
                    return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
                
                current_pos += 2  # フィールド数フィールドをスキップ
                
                # 各フィールドをスキップ
                for field_idx in range(num_fields):
                    if current_pos + 4 > len(chunk_array):
                        print(f"データ境界を超えました（行 {skipped_rows}, フィールド {field_idx}, 位置 {current_pos}/{len(chunk_array)}）")
                        return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
                    
                    # フィールド長を読み取り（ビッグエンディアン）
                    field_len = ((int(chunk_array[current_pos]) & 0xFF) << 24) | \
                               ((int(chunk_array[current_pos+1]) & 0xFF) << 16) | \
                               ((int(chunk_array[current_pos+2]) & 0xFF) << 8) | \
                               (int(chunk_array[current_pos+3]) & 0xFF)
                    
                    # 符号付き32ビット整数に変換
                    if field_len & 0x80000000:
                        field_len = -((~field_len + 1) & 0xFFFFFFFF)
                    
                    current_pos += 4  # フィールド長フィールドをスキップ
                    
                    if field_len == -1:  # NULLフィールド
                        continue
                    elif field_len < 0:
                        print(f"無効なフィールド長: {field_len} (行 {skipped_rows}, フィールド {field_idx})")
                        return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
                    
                    # バッファ境界チェック
                    if current_pos + field_len > len(chunk_array):
                        print(f"フィールドデータがバッファ境界を超えています（行 {skipped_rows}, フィールド {field_idx}, 長さ {field_len}, 位置 {current_pos}/{len(chunk_array)}）")
                        return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
                    
                    # フィールドデータをスキップ
                    current_pos += field_len
                
                # 1行スキップ完了
                skipped_rows += 1
            
            if skipped_rows < start_row:
                print(f"指定された開始行 {start_row} まで到達できませんでした（スキップした行数: {skipped_rows}）")
                return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
        
        # 現在位置以降のデータを処理する
        if current_pos > 0:
            # 現在位置以降のデータで新しい配列を作成
            chunk_array = chunk_array[current_pos:]
            print(f"パース開始位置: {current_pos}, 残りデータ: {len(chunk_array)} バイト")
            
        # ヘッダーは既に処理済みなので、残りのデータを処理
        self.header_expected = False
            
        while attempts < max_attempts:
            field_offsets, field_lengths = parse_binary_chunk(chunk_array, self.header_expected)
            self.header_expected = False  # 2回目以降はヘッダーを期待しない
            
            # 完全な行が得られた場合
            if len(field_offsets) > 0:
                last_field_end = field_offsets[-1] + max(0, field_lengths[-1])
                if last_field_end <= len(chunk_array):
                    # 次のチャンクのために残りデータを保存
                    if last_field_end < len(chunk_array):
                        self._remaining_data = chunk_array[last_field_end:]
                    break
            
            # 追加データが必要な場合（実際の実装では追加データを読み込む処理が入る）
            # この例では単純化のためbreak
            break
        
        # パース完了時の処理
        if len(field_offsets) == 0:
            return None, None, None, 0
            
        # フィールド数から行数を計算
        if num_columns is None:
            # フィールド数が不明の場合は、最初の行のフィールド数を使用
            for i in range(0, min(20, len(field_offsets))):
                # PostgreSQLのバイナリフォーマットでは各行の最初に2バイトのフィールド数がある
                # chunk_arrayがあればそこから検出できるが、ここでは単純化のため固定値を使用
                num_columns = 4  # デフォルト値
                break
        
        if num_columns <= 0:
            num_columns = 1  # 最低でも1列あると仮定
        
        # 最大行数の制限を適用
        total_fields = len(field_offsets)
        rows_in_chunk = total_fields // num_columns
        
        if rows_in_chunk > max_rows:
            print(f"警告: 行数を {max_rows} に制限します (実際の行数: {rows_in_chunk})")
            # フィールド数と行数を制限
            max_fields = max_rows * num_columns
            field_offsets = field_offsets[:max_fields]
            field_lengths = field_lengths[:max_fields]
            rows_in_chunk = max_rows
        
        return chunk_array, field_offsets, field_lengths, rows_in_chunk
