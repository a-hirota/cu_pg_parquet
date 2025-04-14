"""
PostgreSQLバイナリデータの基本パース処理
"""

import numpy as np
from numba import njit
from typing import Tuple
from gpupaser.gpu_decoder import GPUDecoder

# Numbaによる高速化を適用
@njit
def parse_binary_chunk(chunk_array, header_expected=True):
    """バイナリチャンクのパース"""
    field_offsets = []
    field_lengths = []
    pos = np.int64(0)
    
    # デバッグ出力の追加
    print(f"パース開始: chunk_size={len(chunk_array)}, header_expected={header_expected}")
    
    # ヘッダーのスキップ（最初のチャンクのみ）
    if header_expected and len(chunk_array) >= 11:
        header_pattern = np.array([80,71,67,79,80,89,10,255,13,10,0], dtype=np.int32)
        header_match = True
        
        # ヘッダーパターンの比較
        for i in range(min(11, len(chunk_array))):
            if chunk_array[i] != header_pattern[i]:
                header_match = False
                break
        
        if header_match:
            print(f"ヘッダーパターンを検出: PGCOPY\\n\\377\\r\\n\\0")
            pos = np.int64(11)
            
            if len(chunk_array) >= pos + 8:
                # フラグとヘッダー拡張をスキップ
                flags = np.int32((chunk_array[pos] << 24) | (chunk_array[pos+1] << 16) | \
                       (chunk_array[pos+2] << 8) | chunk_array[pos+3])
                pos += np.int64(4)
                ext_len = np.int32((chunk_array[pos] << 24) | (chunk_array[pos+1] << 16) | \
                         (chunk_array[pos+2] << 8) | chunk_array[pos+3])
                pos += np.int64(4)
                
                print(f"ヘッダー情報: フラグ={flags}, 拡張長={ext_len}")
                
                if ext_len > 0 and len(chunk_array) >= pos + ext_len:
                    pos += np.int64(ext_len)
    
    print(f"パースポジション: {pos}/{len(chunk_array)}")
    
    # チャンク内の各タプルを処理
    row_count = 0
    while pos + 2 <= len(chunk_array):
        # タプルのフィールド数を読み取り
        num_fields = np.int16((chunk_array[pos] << 8) | chunk_array[pos + 1])
        
        # 値のチェックと詳細デバッグ（Numba対応のため書式指定子を削除）
        if pos < 100:  # 先頭部分のみ詳細表示
            print(f"位置 {pos}: フィールド数={num_fields}, バイト値=[{chunk_array[pos]},{chunk_array[pos+1]}]")
        
        if num_fields == 0xFFFF:  # ファイル終端 (-1)
            print(f"ファイル終端マーカーを検出: pos={pos}")
            break
            
        if num_fields <= 0 or num_fields > 100:  # 異常値チェック
            print(f"警告: 異常なフィールド数 {num_fields} (位置={pos})")
            break
            
        row_count += 1
        pos += np.int64(2)
        
        # 各フィールドを処理
        field_count_in_row = 0
        for field_idx in range(num_fields):
            if pos + 4 > len(chunk_array):
                print(f"警告: フィールド長読み取り時にバッファ終端に到達 (pos={pos}, len={len(chunk_array)})")
                break
                
            # フィールド長を読み取り
            b0 = chunk_array[pos]
            b1 = chunk_array[pos + 1]
            b2 = chunk_array[pos + 2]
            b3 = chunk_array[pos + 3]
            
            # ビッグエンディアンからリトルエンディアンに変換
            # 整数オーバーフローを防ぐため、int32を使用
            field_len = np.int32(0)
            field_len = field_len | np.int32(b0 & 0xFF) << 24
            field_len = field_len | np.int32(b1 & 0xFF) << 16
            field_len = field_len | np.int32(b2 & 0xFF) << 8
            field_len = field_len | np.int32(b3 & 0xFF)
            
            # field_lenは既にnp.int32なので、符号は自動的に処理される
            
            if field_idx < 2 and row_count <= 2:  # 最初の行の詳細のみ表示
                print(f"行 {row_count} フィールド {field_idx}: 長さ={field_len}, バイト値=[{b0},{b1},{b2},{b3}]")
            
            pos += np.int64(4)
            field_count_in_row += 1
            
            if field_len == -1:  # NULL値
                field_offsets.append(0)  # NULL値のオフセットは0
                field_lengths.append(-1)
            elif field_len < -1:  # 異常値
                print(f"警告: 無効なフィールド長 {field_len} (行={row_count}, フィールド={field_idx})")
                break
            else:
                if pos + field_len > len(chunk_array):
                    print(f"警告: フィールドデータがバッファ終端を超える (pos={pos}, len={field_len}, buffer_size={len(chunk_array)})")
                    # チャンク境界をまたぐ場合は中断、ここまでのデータを返す
                    return np.array(field_offsets, dtype=np.int32), np.array(field_lengths, dtype=np.int32)
                
                # 正常な値フィールド
                field_offsets.append(int(pos))
                field_lengths.append(int(field_len))
                pos += np.int64(field_len)
        
        # 1行の処理が完了
        if field_count_in_row != num_fields:
            print(f"警告: 期待されるフィールド数と実際のフィールド数が一致しません: 期待={num_fields}, 実際={field_count_in_row}")
            break
    
    print(f"パース完了: 処理行数={row_count}, 取得フィールド数={len(field_offsets)}")
    
    return np.array(field_offsets, dtype=np.int32), np.array(field_lengths, dtype=np.int32)

class BinaryDataParser:
    """バイナリデータのパーサークラス"""
    
    def __init__(self, use_gpu=True):
        """初期化
        
        Args:
            use_gpu: GPUパーサーを使用するかどうか
        """
        self.header_expected = True
        self._remaining_data = None
        self.use_gpu = use_gpu
        self.gpu_decoder = GPUDecoder() if use_gpu else None
    
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
        
        # バージョンエラー防止のため、カラム数をnp.int32型に統一
        if num_columns is not None:
            num_columns = np.int32(num_columns)
        
        # フィールド数が不明の場合は検出を試みる
        if num_columns is None and len(chunk_array) >= 2:
            # 型安全なキャスト
            b0 = np.int32(chunk_array[0])
            b1 = np.int32(chunk_array[1])
            detected_columns = np.int32((b0 << 8) | b1)
            
            if 0 < detected_columns < 100:  # 妥当なカラム数範囲チェック
                num_columns = detected_columns
                print(f"検出したカラム数: {num_columns}")
            else:
                print(f"警告: 検出したカラム数 {detected_columns} が範囲外です。デフォルト値を使用します。")
                num_columns = np.int32(8)  # customerテーブルのデフォルト値
        elif num_columns is None:
            print(f"警告: カラム数を検出できません。デフォルト値を使用します。")
            num_columns = np.int32(8)  # customerテーブルのデフォルト値
        
        # 開始行と最大行数をnp.int32に変換
        start_row = np.int32(start_row)
        max_rows = np.int32(max_rows)
        
        # GPU実装を使用する場合、直接GPUでパース
        if self.use_gpu and self.gpu_decoder is not None:
            try:
                print(f"GPUを使用してバイナリデータをパース（{len(chunk_array)}バイト、予想行数:{max_rows}行）")
                # 予測行数を計算（バイト数と列数から概算）
                # このnp.int32キャストは重要なので削除しないでください
                est_rows_val = min(int(max_rows), len(chunk_array) // (int(num_columns) * 20))  # 1フィールド平均20バイトと仮定
                est_rows_val = max(est_rows_val, 1000)  # 最低1000行は確保
                est_rows = np.int32(est_rows_val)
                
                # GPUパーサーを呼び出し
                field_offsets, field_lengths, rows_in_chunk = self.gpu_decoder.parse_binary_data(
                    chunk_array, est_rows, num_columns
                )
                
                # 結果の検証
                if rows_in_chunk > 0:
                    print(f"GPU処理成功: {rows_in_chunk}行解析")
                    # 最大行数の制限
                    if rows_in_chunk > max_rows:
                        print(f"警告: 行数を {max_rows} に制限します (実際の行数: {rows_in_chunk})")
                        max_fields = int(max_rows) * int(num_columns)
                        field_offsets = field_offsets[:max_fields]
                        field_lengths = field_lengths[:max_fields]
                        rows_in_chunk = np.int32(max_rows)
                    
                    return chunk_array, field_offsets, field_lengths, rows_in_chunk
                else:
                    print("GPU処理が0行を返却。CPUパースにフォールバック")
            except Exception as e:
                print(f"GPUパース中にエラー発生、CPUにフォールバック: {e}")
                import traceback
                traceback.print_exc()
        
        # GPU処理が失敗した場合やGPUを使用しない場合はCPU処理
        print("CPUを使用してバイナリデータをパース")
        
        # ヘッダーフラグは最初のチャンク（start_row=0）のみTrueに設定
        self.header_expected = (start_row == 0)
        
        # 最大試行回数を設定
        max_attempts = np.int32(3)
        attempts = np.int32(0)
        
        # ヘッダースキップ処理（最初のチャンクの場合のみ）
        current_pos = np.int32(0)
        header_len = np.int32(0)
        
        if start_row == 0 and len(chunk_array) >= 11:
            # ヘッダーが "PGCOPY\n\377\r\n\0" (バイナリフォーマット識別子) で始まるか確認
            if np.int32(chunk_array[0]) == 80 and np.int32(chunk_array[1]) == 71:  # 'P', 'G'
                header_len = np.int32(11)  # ヘッダー部分の長さ
                
                # フラグフィールドとヘッダー拡張をスキップ
                if len(chunk_array) >= header_len + 8:
                    header_len += np.int32(8)  # フラグ(4バイト) + 拡張長(4バイト)
                    
                    # ヘッダーサイズ（ヘッダー拡張）を読み取り、拡張データをスキップ
                    b0 = np.int32(chunk_array[header_len-4])
                    b1 = np.int32(chunk_array[header_len-3])
                    b2 = np.int32(chunk_array[header_len-2])
                    b3 = np.int32(chunk_array[header_len-1])
                    
                    ext_len = np.int32(0)
                    ext_len = ext_len | ((b0 & 0xFF) << 24)
                    ext_len = ext_len | ((b1 & 0xFF) << 16)
                    ext_len = ext_len | ((b2 & 0xFF) << 8)
                    ext_len = ext_len | (b3 & 0xFF)
                    
                    if ext_len > 0 and len(chunk_array) >= header_len + ext_len:
                        header_len += ext_len
                
                print(f"ヘッダーパターン検出: 長さ={header_len}バイト")
        
        # データポジション計算（ヘッダー + スキップ行分）
        if start_row > 0:
            print(f"開始行 {start_row} からのパース開始: ヘッダー長={header_len}バイト")
            
            # 現在位置をヘッダーの後ろに設定
            current_pos = np.int32(header_len)
            skipped_rows = np.int32(0)
            row_positions = []  # 各行の開始位置を記録
            
            # パフォーマンス向上のため、行位置を事前スキャン
            print(f"行位置のスキャンを開始: 開始位置={current_pos}")
            
            # 指定行数をスキップ（行単位で移動）
            while skipped_rows < start_row and current_pos + 2 <= len(chunk_array):
                # 行の開始位置を記録
                row_positions.append(current_pos)
                
                # 行の最初にあるフィールド数（2バイト）を読み取り
                b0 = np.int32(chunk_array[current_pos])
                b1 = np.int32(chunk_array[current_pos + 1])
                num_fields = np.int32((b0 << 8) | b1)
                
                if num_fields == 0xFFFF:  # ファイル終端マーカー
                    print(f"ファイル終端に到達しました（行 {skipped_rows}）")
                    return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
                
                # フィールド数の妥当性チェック
                if num_fields <= 0 or num_fields > 1000:
                    print(f"警告: 異常なフィールド数 {num_fields} (位置={current_pos})")
                    break
                
                current_pos += np.int32(2)  # フィールド数フィールドをスキップ
                
                # 各フィールドをスキップ
                valid_row = True
                for field_idx in range(num_fields):
                    if current_pos + 4 > len(chunk_array):
                        print(f"データ境界を超えました（行 {skipped_rows}, フィールド {field_idx}, 位置 {current_pos}/{len(chunk_array)}）")
                        valid_row = False
                        break
                    
                    # フィールド長を読み取り（ビッグエンディアン）
                    # 整数オーバーフローを防ぐため、np.int32を使用
                    b0 = np.int32(chunk_array[current_pos])
                    b1 = np.int32(chunk_array[current_pos+1])
                    b2 = np.int32(chunk_array[current_pos+2])
                    b3 = np.int32(chunk_array[current_pos+3])
                    
                    field_len = np.int32(0)
                    field_len = field_len | ((b0 & 0xFF) << 24)
                    field_len = field_len | ((b1 & 0xFF) << 16)
                    field_len = field_len | ((b2 & 0xFF) << 8)
                    field_len = field_len | (b3 & 0xFF)
                    
                    current_pos += np.int32(4)  # フィールド長フィールドをスキップ
                    
                    if field_len == -1:  # NULLフィールド
                        continue
                    elif field_len < 0:
                        print(f"無効なフィールド長: {field_len} (行 {skipped_rows}, フィールド {field_idx})")
                        valid_row = False
                        break
                    
                    # バッファ境界チェック
                    if current_pos + field_len > len(chunk_array):
                        print(f"フィールドデータがバッファ境界を超えています（行 {skipped_rows}, フィールド {field_idx}, 長さ {field_len}, 位置 {current_pos}/{len(chunk_array)}）")
                        valid_row = False
                        break
                    
                    # フィールドデータをスキップ
                    current_pos += field_len
                
                # 行が正常に処理できた場合のみカウント
                if valid_row:
                    skipped_rows += 1
                else:
                    # 無効な行があった場合は処理を中止
                    break
            
            if skipped_rows < start_row:
                print(f"指定された開始行 {start_row} まで到達できませんでした（スキップした行数: {skipped_rows}）")
                return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
            
            # 開始行が見つかった場合、そこからデータを切り出す
            if len(row_positions) >= start_row:
                start_pos = row_positions[start_row - 1] if start_row > 0 else header_len
                print(f"行 {start_row} の開始位置: {start_pos}")
                
                # 現在位置以降のデータで新しい配列を作成
                if start_pos > 0:
                    chunk_array = chunk_array[start_pos:]
                    print(f"パース開始位置: {start_pos}, 残りデータ: {len(chunk_array)} バイト")
        
        # ヘッダー処理フラグの設定（start_row=0の時はヘッダーを期待する）
        self.header_expected = (start_row == 0)
        
        # デバッグ情報の出力
        print(f"ヘッダー処理フラグ: {self.header_expected}, 開始位置: {current_pos}")
            
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
            # フィールド数が不明の場合は、検出を試みる
            # チャンク配列の先頭から最初の行のフィールド数を読み込む
            if len(chunk_array) >= 2:
                detected_columns = ((chunk_array[0] << 8) | chunk_array[1])
                if 0 < detected_columns < 100:  # 妥当なカラム数範囲チェック
                    num_columns = detected_columns
                    print(f"検出したカラム数: {num_columns}")
                else:
                    print(f"警告: 検出したカラム数 {detected_columns} が範囲外です。デフォルト値を使用します。")
                    num_columns = 8  # customerテーブルのデフォルト値
            else:
                print(f"警告: カラム数を検出できません。デフォルト値を使用します。")
                num_columns = 8  # customerテーブルのデフォルト値
        
        if num_columns <= 0:
            num_columns = 1  # 最低でも1列あると仮定
        
        # 最大行数の制限を適用
        total_fields = len(field_offsets)
        
        # カラム数と総フィールド数から行数を計算
        if total_fields == 0:
            print("警告: フィールドが検出されませんでした")
            return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
            
        if total_fields % num_columns != 0:
            print(f"警告: 総フィールド数 {total_fields} がカラム数 {num_columns} の倍数ではありません")
            # 完全な行のみを処理するために、あふれたフィールドを除外
            rows_in_chunk = total_fields // num_columns
            total_fields = rows_in_chunk * num_columns
            field_offsets = field_offsets[:total_fields]
            field_lengths = field_lengths[:total_fields]
        else:
            rows_in_chunk = total_fields // num_columns
        
        print(f"検出した行数: {rows_in_chunk} (フィールド数: {total_fields}, カラム数: {num_columns})")
        
        if rows_in_chunk > max_rows:
            print(f"警告: 行数を {max_rows} に制限します (実際の行数: {rows_in_chunk})")
            # フィールド数と行数を制限
            max_fields = max_rows * num_columns
            field_offsets = field_offsets[:max_fields]
            field_lengths = field_lengths[:max_fields]
            rows_in_chunk = max_rows
        
        if rows_in_chunk > 0:
            print(f"処理完了: {rows_in_chunk}行 ({total_fields}フィールド)")
            return chunk_array, field_offsets, field_lengths, rows_in_chunk
        else:
            print("警告: 有効な行が検出されませんでした")
            return np.zeros(0, dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0
