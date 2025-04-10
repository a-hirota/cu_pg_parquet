
import numpy as np
from numba import cuda, njit
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.driver import CudaAPIError
import cupy as cp
import psycopg2
import struct
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time

# CUDA初期化
try:
    cuda.select_device(0)
    print("CUDA device initialized")
except Exception as e:
    print(f"Failed to initialize CUDA device: {e}")
    raise

def hex_dump(data: bytes, limit=256):
    """
    data (bytes or np.uint8 array) の先頭 limit バイトを
    16進文字列に変換して表示する
    """
    # もし data が np.ndarray の場合は bytes に変換
    if hasattr(data, 'tobytes'):
        data = data.tobytes()
    dump_len = min(limit, len(data))
    hex_str = ' '.join(f"{data[i]:02X}" for i in range(dump_len))
    print(hex_str)
    if len(data) > limit:
        print(f"... (total {len(data)} bytes)")



@dataclass
class ColumnInfo:
    """カラム情報を保持するクラス"""
    name: str
    type: str
    length: Optional[int] = None

def check_table_exists(conn, table_name: str) -> bool:
    """テーブルの存在確認"""
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        )
    """, (table_name,))
    exists = cur.fetchone()[0]
    cur.close()
    return exists

def get_table_info(conn, table_name: str) -> List[ColumnInfo]:
    """テーブル情報の取得"""
    cur = conn.cursor()
    cur.execute("""
        SELECT column_name, data_type, 
               CASE WHEN character_maximum_length IS NOT NULL 
                    THEN character_maximum_length 
                    ELSE NULL 
               END as max_length
        FROM information_schema.columns 
        WHERE table_name = %s 
        ORDER BY ordinal_position
    """, (table_name,))
    
    columns = []
    for name, type_, length in cur.fetchall():
        print(f"Column: {name}, Type: {type_}, Length: {length}")  # デバッグ出力
        columns.append(ColumnInfo(name, type_, length))
    
    cur.close()
    return columns

def get_table_row_count(conn, table_name: str) -> int:
    """テーブルの行数取得"""
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cur.fetchone()[0]
    cur.close()
    print(f"Table {table_name} has {row_count} rows")  # デバッグ出力
    return row_count

def get_column_type(type_name: str) -> int:
    """カラムの型を数値に変換"""
    if type_name == 'integer':
        return 0  # 整数型
    elif type_name in ('numeric', 'decimal'):
        return 1  # 数値型
    elif type_name.startswith(('character', 'text')):
        return 2  # 文字列型
    else:
        raise ValueError(f"Unsupported column type: {type_name}")

def get_column_length(type_name: str, length: Optional[int]) -> int:
    """カラムの長さを取得"""
    if type_name == 'integer':
        return 4  # 32-bit整数
    elif type_name in ('numeric', 'decimal'):
        return 8  # 64-bit数値
    elif type_name.startswith('character'):
        return int(length) if length else 256
    elif type_name == 'text':
        return 1024  # テキスト型のデフォルト長
    else:
        raise ValueError(f"Unsupported column type: {type_name}")

class ChunkConfig:
    def __init__(self, total_rows=6_000_000):
        # チャンクサイズを行数に基づいて調整
        self.rows_per_chunk = min(4096, total_rows)  # 最大4096行に制限
        self.num_chunks = (total_rows + self.rows_per_chunk - 1) // self.rows_per_chunk
        self.threads_per_block = 256  # スレッド数を増加
        self.max_blocks = 65535  # CUDA制限
        
    def get_grid_size(self, chunk_size):
        return min(
            self.max_blocks,
            (chunk_size + self.threads_per_block - 1) // self.threads_per_block
        )

# デコード用の補助関数
@cuda.jit(device=True)
def check_bounds(data, pos, size):
    """境界チェック"""
    return pos >= 0 and pos + size <= len(data)

@cuda.jit(device=True)
def decode_int16(data, pos):
    """2バイト整数のデコード（ビッグエンディアン）"""
    if not check_bounds(data, pos, 2):
        return 0
    
    # バイトを取得
    b0 = data[pos]
    b1 = data[pos + 1]
    
    # ビッグエンディアンからリトルエンディアンに変換
    val = ((b0 & 0xFF) << 8) | (b1 & 0xFF)
    
    # 符号付き16ビット整数に変換
    if val & 0x8000:  # 最上位ビットが1なら負の数
        val = -(((~val) + 1) & 0xFFFF)
    
    return val

@cuda.jit(device=True)
def decode_int32(data, pos):
    """4バイト整数のデコード（ビッグエンディアン）"""
    if not check_bounds(data, pos, 4):
        return 0
    
    # バイトを取得
    b0 = data[pos]
    b1 = data[pos + 1]
    b2 = data[pos + 2]
    b3 = data[pos + 3]
    
    # ビッグエンディアンからリトルエンディアンに変換
    val = ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)
    
    # 符号付き32ビット整数に変換
    if val & 0x80000000:  # 最上位ビットが1なら負の数
        val = -(((~val) + 1) & 0xFFFFFFFF)
    
    return val

@cuda.jit(device=True)
def decode_numeric_postgres(data, pos, hi_out, lo_out, scale_out, row_idx):
    """PostgreSQLのnumeric型を128ビット固定小数点数に変換"""
    if not check_bounds(data, pos, 8):  # 少なくともヘッダー部分があるか
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # ヘッダー情報の取得
    ndigits = decode_int16(data, pos)
    weight = decode_int16(data, pos + 2)
    sign = decode_int16(data, pos + 4)
    dscale = decode_int16(data, pos + 6)
    
    # データの妥当性チェック
    if ndigits < 0 or ndigits > 100 or dscale < 0 or dscale > 100:
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # 必要なバイト数をチェック
    if not check_bounds(data, pos + 8, ndigits * 2):
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # 128ビット整数に変換
    hi = 0
    lo = 0
    
    # 各桁を処理
    digit_pos = pos + 8
    scale = 0
    
    for i in range(ndigits):
        digit = decode_int16(data, digit_pos + i * 2)
        if digit < 0 or digit > 9999:  # 不正な桁
            continue
            
        # 既存の値を10000倍して新しい桁を加算
        hi_old = hi
        lo_old = lo
        
        # 10000倍
        lo = lo_old * 10000
        hi = hi_old * 10000 + (lo_old >> 32) * 10000
        
        # 桁を加算
        lo += digit
        if lo < lo_old:  # 桁上がり
            hi += 1
        
        # スケールの更新
        scale = max(scale, dscale)
    
    # 符号の適用
    if sign == 0x4000:  # 負の数
        if lo == 0:
            hi = -hi
        else:
            lo = ~lo + 1
            hi = ~hi
            if lo == 0:
                hi += 1
    
    # 結果の格納
    scale_out[0] = scale
    hi_out[row_idx] = hi
    lo_out[row_idx] = lo

@cuda.jit(device=True)
def bulk_copy_64bytes(src, src_pos, dst, dst_pos, size):
    """64バイト単位でのバルクコピー"""
    if size > 64:
        size = 64
    
    # 8バイトずつコピー
    for i in range(0, size, 8):
        if i + 8 <= size:
            # 8バイトを一度に読み書き
            val = 0
            for j in range(8):
                val = (val << 8) | src[src_pos + i + j]
            
            # 8バイトを一度に書き込み
            for j in range(8):
                dst[dst_pos + i + j] = (val >> ((7-j) * 8)) & 0xFF
        else:
            # 残りのバイトを1バイトずつコピー
            for j in range(size - i):
                dst[dst_pos + i + j] = src[src_pos + i + j]

@cuda.jit
def decode_all_columns_kernel(raw_data, field_offsets, field_lengths,
                            int_outputs, num_hi_output, num_lo_output, num_scale_output,
                            str_outputs, str_null_pos,
                            col_types, col_lengths, chunk_size, num_cols):
    """全カラムを一度に処理する統合カーネル"""
    # スレッドインデックスの計算を改善
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    grid_size = cuda.gridDim.x
    
    # グリッド内の絶対位置を計算
    row = block_id * block_size + thread_id
    
    # ストライド処理を追加
    stride = block_size * grid_size
    while row < chunk_size:
        for col in range(num_cols):
            # フィールドオフセットとデータ長を取得
            field_idx = row * num_cols + col
            if field_idx >= len(field_offsets):
                return
                
            pos = field_offsets[field_idx]
            length = field_lengths[field_idx]
            
            if col_types[col] <= 1:  # 数値型
                if length == -1:  # NULL値
                    int_outputs[col * chunk_size + row] = 0
                else:
                    if col_types[col] == 0:  # integer
                        val = decode_int32(raw_data, pos)
                        int_outputs[col * chunk_size + row] = val
                    else:  # numeric/decimal
                        # PostgreSQLのnumeric型を128ビット固定小数点数として処理
                        decode_numeric_postgres(raw_data, pos,
                                             num_hi_output[col * chunk_size:],
                                             num_lo_output[col * chunk_size:],
                                             num_scale_output[col:col + 1],
                                             row)
            else:  # 文字列型
                max_length = col_lengths[col]
                dst_pos = (col * chunk_size + row) * max_length
                
                if length == -1:  # NULL値
                    str_null_pos[col * chunk_size + row] = 0
                    continue
                    
                # 文字列データのコピー
                valid_length = min(length, max_length)
                
                # バルクコピーを使用
                for i in range(0, valid_length, 64):
                    copy_size = min(64, valid_length - i)
                    bulk_copy_64bytes(raw_data, pos + i, str_outputs, dst_pos + i, copy_size)
                
                # 文字列の有効範囲を探す
                valid_start = 0
                valid_end = valid_length
                
                # # ヌルバイトを探す
                # for i in range(valid_length):
                #     if str_outputs[dst_pos + i] == 0:
                #         valid_end = i
                #         break
                
                # 前後の空白を除去
                while valid_start < valid_end and str_outputs[dst_pos + valid_start] <= 32:
                    valid_start += 1
                
                while valid_end > valid_start and str_outputs[dst_pos + valid_end - 1] <= 32:
                    valid_end -= 1
                
                # 文字列の範囲を設定
                if valid_end > valid_start:
                    # 文字列の範囲を保存（上位16ビットにstart、下位16ビットにend）
                    str_null_pos[col * chunk_size + row] = (valid_start << 16) | valid_end
                else:
                    str_null_pos[col * chunk_size + row] = 0
        
        # 次の行へ
        row += stride

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

class PgGpuStreamProcessor:
    def __init__(self, conn, chunk_config):
        self.conn = conn
        self.config = chunk_config
        self.header_expected = True
        
    def _initialize_device_buffers(self, columns, chunk_size):
        """GPUバッファの初期化"""
        # カラム情報の収集
        col_types = []  # 0: integer, 1: numeric, 2: string
        col_lengths = []
        max_str_length = 0
        num_int_cols = 0
        num_str_cols = 0
        
        for col in columns:
            col_type = get_column_type(col.type)
            col_types.append(col_type)
            
            if col_type <= 1:  # 数値型（integer or numeric）
                num_int_cols += 1
                col_lengths.append(get_column_length(col.type, col.length))
            else:  # 文字列型
                num_str_cols += 1
                length = get_column_length(col.type, col.length)
                max_str_length = max(max_str_length, length)
                col_lengths.append(length)
        
        # バッファの確保
        try:
            # バッファサイズの計算
            int_buffer_size = chunk_size * num_int_cols
            str_buffer_size = chunk_size * num_str_cols * max_str_length
            str_null_pos_size = chunk_size * num_str_cols
            
            print(f"Allocating buffers: int={int_buffer_size}, str={str_buffer_size}, null={str_null_pos_size}")
            
            # バッファの確保と初期化
            int_buffer = None
            num_hi_output = None
            num_lo_output = None
            num_scale_output = None
            str_buffer = None
            str_null_pos = None
            
            # メモリ割り当ての順序を調整
            if num_int_cols > 0:
                int_buffer = cuda.to_device(np.zeros(int_buffer_size, dtype=np.int32))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
                # numeric型用のバッファ
                num_hi_output = cuda.device_array(chunk_size * num_int_cols, dtype=np.int64)
                num_lo_output = cuda.device_array(chunk_size * num_int_cols, dtype=np.int64)
                num_scale_output = cuda.device_array(num_int_cols, dtype=np.int32)
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
            if num_str_cols > 0:
                # 文字列バッファを一括で確保
                str_buffer = cuda.to_device(np.zeros(str_buffer_size, dtype=np.uint8))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
                
                str_null_pos = cuda.to_device(np.zeros(str_null_pos_size, dtype=np.int32))
                cuda.synchronize()  # メモリ割り当ての完了を待つ
            
            return int_buffer, num_hi_output, num_lo_output, num_scale_output, \
                   str_buffer, str_null_pos, \
                   np.array(col_types, dtype=np.int32), np.array(col_lengths, dtype=np.int32)
        except CudaAPIError as e:
            print(f"Failed to allocate GPU memory: {e}")
            # クリーンアップ
            if int_buffer is not None:
                del int_buffer
            if num_hi_output is not None:
                del num_hi_output
            if num_lo_output is not None:
                del num_lo_output
            if num_scale_output is not None:
                del num_scale_output
            if str_buffer is not None:
                del str_buffer
            if str_null_pos is not None:
                del str_null_pos
            cuda.synchronize()
            raise

    def _read_and_parse_chunk(self, copy_data, chunk_size):
        """チャンクデータの読み込みとパース（Numba最適化版）"""
        # 前のチャンクの残りデータを使用
        if hasattr(self, '_remaining_data'):
            chunk_array = self._remaining_data
            del self._remaining_data
        else:
            # バッファの現在位置を保存
            current_pos = copy_data.tell()
            chunk = copy_data.read(chunk_size)
            if not chunk:
                if current_pos > 0:
                    # バッファを先頭に戻して再試行
                    copy_data.seek(0)
                    chunk = copy_data.read(chunk_size)
                    if not chunk:
                        return None
                else:
                    return None
            chunk_array = np.frombuffer(chunk, dtype=np.uint8)
        
        # 最大試行回数を設定
        max_attempts = 3
        attempts = 0
        
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
            
            # 追加データの読み込み
            chunk = copy_data.read(chunk_size)
            if not chunk:
                break
            
            # 配列の結合
            new_array = np.frombuffer(chunk, dtype=np.uint8)
            chunk_array = np.concatenate([chunk_array, new_array])
            attempts += 1
        
        if len(field_offsets) == 0:
            return None
            
        try:
            # 行数の計算
            rows_in_chunk = len(field_offsets) // len(self.columns)
            rows_in_chunk = min(rows_in_chunk, self.config.rows_per_chunk)  # 最大行数に制限
            print(f"Parsed {rows_in_chunk} rows from chunk")  # デバッグ出力
            
            # フィールドの制限
            field_offsets = field_offsets[:rows_in_chunk * len(self.columns)]
            field_lengths = field_lengths[:rows_in_chunk * len(self.columns)]
            
            # デバイスメモリの確保と転送
            d_chunk = cuda.to_device(chunk_array)
            cuda.synchronize()  # メモリ転送の完了を待つ
            
            d_offsets = cuda.to_device(field_offsets)
            cuda.synchronize()  # メモリ転送の完了を待つ
            
            d_lengths = cuda.to_device(field_lengths)
            cuda.synchronize()  # メモリ転送の完了を待つ
            
            return d_chunk, d_offsets, d_lengths, rows_in_chunk
        except Exception as e:
            print(f"Error during chunk processing: {e}")
            raise
        except CudaAPIError as e:
            print(f"Failed to transfer data to GPU: {e}")
            raise

    def process_table(self, table_name, columns):
        """テーブル全体の処理"""
        self.columns = columns
        cur = self.conn.cursor()
        
        # バイナリデータを一時的にメモリに保存
        import io
        buffer = io.BytesIO()
        cur.copy_expert(f"COPY (SELECT * FROM {table_name} LIMIT 2000)  TO STDOUT WITH (FORMAT binary)", buffer)
        
        # バッファをメモリに固定
        buffer_data = buffer.getvalue()

        print("=== Raw binary data (first 256 bytes) ===")
        hex_dump(buffer_data, 256)
        # バイナリファイルに書き込む
        with open('output_debug.bin', 'wb') as f:
            f.write(buffer_data)

        print("バイナリファイルに書き込みました。")


        buffer = io.BytesIO(buffer_data)
        
        # バッファサイズの確認
        total_size = buffer.getbuffer().nbytes
        print(f"Total binary data size: {total_size} bytes")  # デバッグ出力
        
        # 全データを一度に処理
        chunk_array = np.frombuffer(buffer_data, dtype=np.uint8)
        print(list(chunk_array[:300]))
        print(f"Processing entire table data")  # デバッグ出力
        
        # buffer_dataはここで使い終わったので削除
        del buffer_data
        
        # バッファの初期化
        int_buffer, num_hi_output, num_lo_output, num_scale_output, \
        str_buffer, str_null_pos, col_types, col_lengths = \
            self._initialize_device_buffers(columns, self.config.rows_per_chunk)
        
        # デバイスに転送
        d_col_types = cuda.to_device(col_types)
        d_col_lengths = cuda.to_device(col_lengths)
        
        # 結果格納用の辞書
        results = {col.name: [] for col in columns}
        
        try:
            # 全データを一度にパース
            start_time = time.time()
            field_offsets, field_lengths = parse_binary_chunk(chunk_array, True)
            
            # 行数の計算
            rows_in_chunk = len(field_offsets) // len(self.columns)
            print(f"Processing {rows_in_chunk} rows")  # デバッグ出力
            
            # デバイスメモリの確保と転送
            d_chunk = cuda.to_device(chunk_array)
            d_offsets = cuda.to_device(field_offsets)
            d_lengths = cuda.to_device(field_lengths)
            cuda.synchronize()  # メモリ転送の完了を待つ
            
            total_rows = rows_in_chunk
            print(f"Processing {rows_in_chunk} rows")  # デバッグ出力
            
            # スレッド数とブロック数の調整
            threads_per_block = 256  # 固定スレッド数
            blocks = min(
                1024,  # ブロック数の制限を緩和
                max(128, (rows_in_chunk + threads_per_block - 1) // threads_per_block)  # 最小128ブロック
            )
            print(f"Using {blocks} blocks with {threads_per_block} threads per block")  # デバッグ出力
                
            # 統合カーネルの起動
            decode_all_columns_kernel[blocks, threads_per_block](
                d_chunk, d_offsets, d_lengths,
                int_buffer, num_hi_output, num_lo_output, num_scale_output,
                str_buffer, str_null_pos,
                d_col_types, d_col_lengths,
                rows_in_chunk, len(columns)
            )
                
            # 結果の回収
            int_col_idx = 0
            str_col_idx = 0
            
            for i, col in enumerate(columns):
                col_type = get_column_type(col.type)
                if col_type <= 1:  # 数値型（integer or numeric）
                    if col_type == 0:  # integer
                        if int_buffer is not None:
                            host_array = np.empty(rows_in_chunk, dtype=np.int32)
                            int_buffer[int_col_idx * rows_in_chunk:(int_col_idx + 1) * rows_in_chunk].copy_to_host(host_array)
                            results[col.name].append(host_array)
                            int_col_idx += 1
                    else:  # numeric
                        if num_hi_output is not None and num_lo_output is not None:
                            host_hi = np.empty(rows_in_chunk, dtype=np.int64)
                            host_lo = np.empty(rows_in_chunk, dtype=np.int64)
                            host_scale = np.empty(1, dtype=np.int32)
                            
                            num_hi_output[int_col_idx * rows_in_chunk:(int_col_idx + 1) * rows_in_chunk].copy_to_host(host_hi)
                            num_lo_output[int_col_idx * rows_in_chunk:(int_col_idx + 1) * rows_in_chunk].copy_to_host(host_lo)
                            num_scale_output[int_col_idx:int_col_idx + 1].copy_to_host(host_scale)
                            
                            values = []
                            for j in range(rows_in_chunk):
                                hi = host_hi[j]
                                lo = host_lo[j]
                                scale = host_scale[0]
                                
                                if hi == 0 and lo >= 0:
                                    value = float(lo) / (10 ** scale) if scale > 0 else float(lo)
                                elif hi == -1 and lo < 0:
                                    value = float(lo) / (10 ** scale) if scale > 0 else float(lo)
                                else:
                                    value = f"[{hi},{lo}]@{scale}"
                                
                                values.append(value)
                            
                            results[col.name].append(values)
                            int_col_idx += 1
                else:  # 文字列型
                    if str_buffer is not None and str_null_pos is not None:
                        length = get_column_length(col.type, col.length)

                        # 文字列データの取得
                        str_start = str_col_idx * rows_in_chunk * length
                        str_end   = (str_col_idx + 1) * rows_in_chunk * length
                        host_array = np.empty(rows_in_chunk * length, dtype=np.uint8)
                        str_buffer[str_start:str_end].copy_to_host(host_array)

                        null_positions = np.empty(rows_in_chunk, dtype=np.int32)
                        str_null_pos[str_col_idx * rows_in_chunk:(str_col_idx + 1) * rows_in_chunk].copy_to_host(null_positions)

                        data = host_array.reshape(-1, length)
                        strings = []

                        for row in range(rows_in_chunk):
                            offset = null_positions[row]
                            if offset == 0:
                                # NULL or 長さ0
                                strings.append('')
                            else:
                                start_pos = (offset >> 16) & 0xFFFF
                                end_pos   = offset & 0xFFFF
                                
                                if end_pos > start_pos:
                                    row_data = data[row, start_pos:end_pos]
                                    try:
                                        s = row_data.tobytes().decode('utf-8', errors='replace')
                                    except:
                                        s = ''
                                    strings.append(s)
                                else:
                                    # 万一start_pos >= end_posなら空文字
                                    strings.append('')

                        results[col.name].append(strings)
                        str_col_idx += 1

                        
                        # # 文字列データの取得
                        # str_start = str_col_idx * rows_in_chunk * length
                        # str_end = (str_col_idx + 1) * rows_in_chunk * length
                        # host_array = np.empty(rows_in_chunk * length, dtype=np.uint8)
                        # str_buffer[str_start:str_end].copy_to_host(host_array)
                            
                        # # ヌルバイト位置の取得
                        # null_positions = np.empty(rows_in_chunk, dtype=np.int32)
                        # str_null_pos[str_col_idx * rows_in_chunk:(str_col_idx + 1) * rows_in_chunk].copy_to_host(null_positions)
                        
                        # # 文字列の変換（ベクトル化）
                        # strings = []
                        # data = host_array.reshape(-1, length)
                        # for row in range(rows_in_chunk):
                        #     end_pos = null_positions[row]
                        #     if end_pos == 0:  # NULL値
                        #         strings.append('')
                        #     else:
                        #         # 文字列データの取り出しとデコード
                        #         # if end_pos > 0:
                        #         #     # 文字列の範囲を取得
                        #         #     start_pos = (end_pos >> 16) & 0xFFFF
                        #         #     end_pos = end_pos & 0xFFFF
                                    
                        #         #     if end_pos > start_pos:
                        #         #         # 文字列全体を取得
                        #         #         row_data = data[row, start_pos:end_pos]
                        #         #         # バイナリデータをクリーンアップ
                        #         #         valid_data = []
                        #         #         for b in row_data:
                        #         #             if b >= 32 and b <= 126:  # 印字可能なASCII文字のみ
                        #         #                 valid_data.append(b)
                        #         if end_pos > 0:
                        #             # end_posはフィールド長そのもの
                        #             row_data = data[row, :end_pos]
                        #             try:
                        #                 s = bytes(row_data).decode('utf-8', errors='replace')
                        #                 strings.append(s)
                        #             except:
                        #                 strings.append('')  # デコードエラーの場合
                                        
                        #             # if valid_data:
                        #             #     try:
                        #             #         # 文字列をデコード
                        #             #         s = bytes(valid_data).decode('utf-8', errors='replace')
                        #             #         s = s.strip()  # 前後の空白を除去
                        #             #         if s:  # 空文字列でない場合のみ追加
                        #             #             strings.append(s)
                        #             #             continue
                        #             #     except:
                        #             #         pass  # デコードエラーの場合は空文字列を追加
                        #         strings.append('')  # NULL値または空文字列の場合
                        # results[col.name].append(strings)
                        # str_col_idx += 1
                
            # 一時的なGPUメモリの解放
            cuda.synchronize()  # 処理完了を待つ
            del d_chunk
            del d_offsets
            del d_lengths
            cuda.synchronize()  # メモリ解放完了を待つ
            
            print(f"Processing completed in {time.time() - start_time:.3f}s")
            
            print(f"Total rows processed: {total_rows}")  # デバッグ出力
        
        finally:
            # リソースのクリーンアップ
            cuda.synchronize()  # すべての操作が完了するのを待つ
            
            del d_col_types
            del d_col_lengths
            if int_buffer is not None:
                del int_buffer
            if str_buffer is not None:
                del str_buffer
            if str_null_pos is not None:
                del str_null_pos
            
            cuda.synchronize()  # メモリ解放が完了するのを待つ
        
        # 結果の結合
        final_results = {}
        for col_name, chunks in results.items():
            if chunks:
                if isinstance(chunks[0], np.ndarray):
                    final_results[col_name] = np.concatenate(chunks)
                else:
                    final_results[col_name] = [item for chunk in chunks for item in chunk]
        
        return final_results

def load_table_optimized(table_name: str):
    """最適化されたGPU実装でテーブルを読み込む"""
    # PostgreSQLに接続
    conn = psycopg2.connect(
        dbname='postgres',
        user='postgres',
        password='postgres',
        host='localhost'
    )
    
    try:
        # テーブルの存在確認
        if not check_table_exists(conn, table_name):
            raise ValueError(f"Table {table_name} does not exist")
            
        # テーブル情報の取得
        columns = get_table_info(conn, table_name)
        if not columns:
            raise ValueError(f"No columns found in table {table_name}")
            
        # 行数の取得
        row_count = get_table_row_count(conn, table_name)
        
        # 処理の設定
        chunk_config = ChunkConfig(row_count)
        processor = PgGpuStreamProcessor(conn, chunk_config)
        
        # GPUでデコード
        results = processor.process_table(table_name, columns)
        
        return results
    finally:
        conn.close()

if __name__ == "__main__":
    import time
    
    # date1テーブルのテスト
    print("=== date1テーブル ===")
    print("\n[最適化GPU実装]")
    start_time = time.time()
    try:
        results_date1 = load_table_optimized('date1')
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
        print("\n最初の5行:")
        for col_name, data in results_date1.items():
            print(f"{col_name}: {data[:5]}")
    except Exception as e:
        print(f"Error processing date1 table: {e}")
    
    # customerテーブルのテスト
    print("\n=== customerテーブル ===")
    print("\n[最適化GPU実装]")
    start_time = time.time()
    try:
        results_customer = load_table_optimized('customer')
        gpu_time = time.time() - start_time
        print(f"処理時間: {gpu_time:.3f}秒")
        print("\n最初の5行:")
        for col_name, data in results_customer.items():
            print(f"{col_name}: {data[:5]}")
    except Exception as e:
        print(f"Error processing customer table: {e}")
