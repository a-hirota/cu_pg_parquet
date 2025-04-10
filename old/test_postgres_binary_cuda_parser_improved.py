import numpy as np
from numba import cuda, njit
from numba.cuda.cudadrv import driver
from numba.cuda.cudadrv.driver import CudaAPIError
import cupy as cp
import psycopg2
import struct
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import io

# CUDA初期化
try:
    cuda.select_device(0)
    print("CUDA device initialized")
except Exception as e:
    print(f"Failed to initialize CUDA device: {e}")
    raise

@dataclass
class ColumnInfo:
    """カラム情報を保持するクラス"""
    name: str
    type: str
    length: Optional[int] = None
    
    def get_type_id(self) -> int:
        """カラムの型をID値に変換"""
        if self.type == 'integer':
            return 0  # 整数型
        elif self.type in ('numeric', 'decimal'):
            return 1  # 数値型
        elif self.type.startswith(('character', 'text')):
            return 2  # 文字列型
        else:
            raise ValueError(f"Unsupported column type: {self.type}")
    
    def get_element_size(self) -> int:
        """データ型の要素サイズを取得"""
        if self.type == 'integer':
            return 4  # 32-bit整数
        elif self.type in ('numeric', 'decimal'):
            return 16  # 128-bit数値型（PG-Strom方式）
        elif self.type.startswith('character'):
            return 8  # オフセット+長さ（固定バイト数）
        elif self.type == 'text':
            return 8  # オフセット+長さ（固定バイト数）
        else:
            raise ValueError(f"Unsupported column type: {self.type}")

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

class GpuBufferManager:
    """GPUバッファ管理クラス"""
    def __init__(self):
        self.fixed_buffers = {}  # 固定長データ用
        self.var_data = {}      # 可変長データ用
        self.var_offsets = {}   # 可変長データのオフセット配列
        
    def allocate_buffers(self, columns: List[ColumnInfo], row_count: int):
        """全カラムのバッファを一括確保"""
        print(f"Allocating GPU buffers for {len(columns)} columns, {row_count} rows")
        
        for col in columns:
            type_id = col.get_type_id()
            
            if type_id <= 1:  # int/numeric
                # 固定長バッファ
                if type_id == 0:  # integer
                    self.fixed_buffers[col.name] = cuda.device_array(row_count, dtype=np.int32)
                else:  # numeric
                    # numeric型は128ビット固定小数点で表現（hi:64bit, lo:64bit）
                    # CUDAでは構造体配列として扱えないため、2つの配列で管理
                    self.fixed_buffers[f"{col.name}_hi"] = cuda.device_array(row_count, dtype=np.int64)
                    self.fixed_buffers[f"{col.name}_lo"] = cuda.device_array(row_count, dtype=np.int64)
                    # scale値も保持（すべての行で共通）
                    self.fixed_buffers[f"{col.name}_scale"] = cuda.device_array(1, dtype=np.int32)
                
            else:  # text/varchar
                # 可変長データ用のオフセット配列
                self.var_offsets[col.name] = cuda.device_array(row_count + 1, dtype=np.int32)
                # デフォルトで0埋め
                self.var_offsets[col.name].copy_to_device(np.zeros(row_count + 1, dtype=np.int32))
                
                # 可変長データ用の初期バッファ
                # 最初は平均64バイト/行で確保（後で必要に応じて拡張）
                self.var_data[col.name] = cuda.device_array(row_count * 64, dtype=np.uint8)
    
    def get_fixed_buffer(self, col_name: str) -> cuda.devicearray.DeviceNDArray:
        """固定長バッファの取得"""
        return self.fixed_buffers.get(col_name)
    
    def get_var_buffers(self, col_name: str) -> Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """可変長バッファとオフセット配列の取得"""
        return self.var_data.get(col_name), self.var_offsets.get(col_name)
    
    def cleanup(self):
        """メモリの解放"""
        for buffer in list(self.fixed_buffers.values()):
            del buffer
        
        for buffer in list(self.var_data.values()):
            del buffer
            
        for buffer in list(self.var_offsets.values()):
            del buffer
        
        cuda.synchronize()
        
        self.fixed_buffers = {}
        self.var_data = {}
        self.var_offsets = {}

# バイナリデータ解析
@njit
def analyze_binary_format(raw_data):
    """バイナリデータの構造を解析"""
    pos = np.int64(0)
    
    # ヘッダーのスキップ
    if len(raw_data) >= 11:
        header = np.array([80,71,67,79,80,89,10,255,13,10,0], dtype=np.uint8)
        if np.all(raw_data[0:11] == header):
            pos = np.int64(11)
            if len(raw_data) >= pos + 8:
                # フラグとヘッダー拡張をスキップ
                flags = np.int32((raw_data[pos] << 24) | (raw_data[pos+1] << 16) | \
                       (raw_data[pos+2] << 8) | raw_data[pos+3])
                pos += np.int64(4)
                ext_len = np.int32((raw_data[pos] << 24) | (raw_data[pos+1] << 16) | \
                         (raw_data[pos+2] << 8) | raw_data[pos+3])
                pos += np.int64(4) + np.int64(ext_len)
    
    # フィールドオフセットを収集
    field_offsets = []
    field_lengths = []
    
    while pos + 2 <= len(raw_data):
        # タプルのフィールド数を読み取り
        num_fields = np.int16((raw_data[pos] << 8) | raw_data[pos + 1])
        if num_fields == -1:  # ファイル終端
            break
            
        pos += np.int64(2)
        
        # 各フィールドを処理
        for _ in range(num_fields):
            if pos + 4 > len(raw_data):
                break
                
            # フィールド長を読み取り
            b0 = raw_data[pos]
            b1 = raw_data[pos + 1]
            b2 = raw_data[pos + 2]
            b3 = raw_data[pos + 3]
            
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
                field_offsets.append(int(pos))
                field_lengths.append(int(field_len))
                pos += np.int64(field_len)
    
    return np.array(field_offsets, dtype=np.int32), np.array(field_lengths, dtype=np.int32)

# デコード用の補助関数
@cuda.jit(device=True)
def check_bounds(data, pos, size):
    """境界チェック"""
    return pos >= 0 and pos + size <= len(data)

@cuda.jit(device=True)
def decode_int32(data, pos):
    """4バイト整数のデコード（ビッグエンディアン）"""
    if not check_bounds(data, pos, 4):
        return 0
    
    # バイトを取得してエンディアン変換
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
def decode_int64(data, pos):
    """8バイト整数のデコード（ビッグエンディアン）"""
    if not check_bounds(data, pos, 8):
        return 0
    
    # 上位32ビットと下位32ビットに分けて処理
    hi = decode_int32(data, pos)
    lo = decode_int32(data, pos + 4)
    
    # 符号なし32ビット値として扱う
    hi_unsigned = hi & 0xFFFFFFFF
    lo_unsigned = lo & 0xFFFFFFFF
    
    # 64ビット整数に結合
    return (hi_unsigned << 32) | lo_unsigned

@cuda.jit(device=True)
def decode_int16(data, pos):
    """2バイト整数のデコード（ビッグエンディアン）"""
    if not check_bounds(data, pos, 2):
        return 0
    
    b0 = data[pos]
    b1 = data[pos + 1]
    
    # 符号付き16ビット整数として解釈
    val = ((b0 & 0xFF) << 8) | (b1 & 0xFF)
    if val & 0x8000:  # 最上位ビットが1なら負の数
        val = -(((~val) + 1) & 0xFFFF)
    
    return val

@cuda.jit(device=True)
def decode_numeric_postgres(data, pos, hi_out, lo_out, scale_out, row_idx):
    """PostgreSQLのnumeric型を128ビット固定小数点数に変換"""
    # PostgreSQLのnumeric型のバイナリ形式:
    # int16 ndigits - 数字の数
    # int16 weight - 先頭の数字の位置（10000進数）
    # int16 sign - 符号（0: 正, 0x4000: 負）
    # int16 dscale - 表示スケール
    # int16[ndigits] digits - 各桁の値（0-9999）
    
    if not check_bounds(data, pos, 8):  # 少なくともヘッダー部分があるか
        # NULL または無効なデータ
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
        # 異常値の場合はゼロを設定
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # 必要なバイト数をチェック
    if not check_bounds(data, pos + 8, ndigits * 2):
        # データ不足
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return
    
    # 128ビット整数に変換（PG-Strom方式）
    hi = 0
    lo = 0
    
    # 各桁（0-9999）を処理して128ビット整数に累積
    digit_pos = pos + 8  # 桁データの開始位置
    for i in range(ndigits):
        digit = decode_int16(data, digit_pos + i * 2)
        
        # 既存の値を10000倍して新しい桁を加算
        # 128ビット計算を64ビットx2で実装
        hi_old = hi
        lo_old = lo
        
        # lo * 10000
        lo = lo_old * 10000
        
        # hi * 10000 + (lo_old * 10000) >> 64
        hi = hi_old * 10000 + (lo_old >> 16) * (10000 >> 16)
        
        # digitを加算
        lo += digit
        if lo < lo_old:  # 桁上がり
            hi += 1
    
    # 符号の適用
    if sign != 0:
        # 負の数: 2の補数表現
        lo = ~lo + 1
        hi = ~hi
        if lo == 0:
            hi += 1
    
    # スケールの保存
    scale_out[0] = dscale
    
    # 結果の格納
    hi_out[row_idx] = hi
    lo_out[row_idx] = lo

@cuda.jit(device=True)
def copy_string_data(src, src_pos, dst, dst_pos, length):
    """文字列データのコピー"""
    if length <= 0:
        return
    
    # 1バイトずつコピー
    for i in range(length):
        if src_pos + i < len(src) and dst_pos + i < len(dst):
            dst[dst_pos + i] = src[src_pos + i]

@cuda.jit
def calculate_var_sizes_kernel(raw_data, field_offsets, field_lengths, 
                              var_col_indices, var_total_sizes, row_count, num_cols):
    """可変長データの総サイズを計算するカーネル"""
    row_idx = cuda.grid(1)
    if row_idx >= row_count:
        return
    
    for v_idx, col_idx in enumerate(var_col_indices):
        field_idx = row_idx * num_cols + col_idx
        length = field_lengths[field_idx]
        
        if length > 0:
            # 可変長データのサイズを累積
            cuda.atomic.add(var_total_sizes, v_idx, length)

@cuda.jit
def preprocess_var_offsets_kernel(var_offsets, var_col_indices, field_lengths, 
                                row_count, num_cols):
    """可変長データのオフセット配列を前処理するカーネル"""
    row_idx = cuda.grid(1)
    if row_idx >= row_count:
        return
    
    for v_idx, col_idx in enumerate(var_col_indices):
        field_idx = row_idx * num_cols + col_idx
        length = field_lengths[field_idx]
        
        if length > 0:
            # 各行の可変長データのサイズをオフセット配列に設定
            var_offsets[v_idx][row_idx + 1] = length

@cuda.jit
def exclusive_scan_kernel(input_array, output_array, length):
    """排他的スキャン（prefix sum）を計算するカーネル"""
    # 単純なCUDAカーネルでの実装（実際にはCUBやThrustを使うべき）
    tid = cuda.grid(1)
    if tid < length:
        if tid == 0:
            output_array[0] = 0
        else:
            output_array[tid] = output_array[tid - 1] + input_array[tid - 1]

@cuda.jit
def parse_binary_data_kernel(raw_data, field_offsets, field_lengths,
                           int_col_indices, num_col_indices, var_col_indices,
                           int_buffers, num_hi_buffers, num_lo_buffers, num_scale_buffers,
                           var_data_buffers, var_offsets_buffers,
                           row_count, num_cols):
    """バイナリデータをパースするメインカーネル"""
    row_idx = cuda.grid(1)
    if row_idx >= row_count:
        return
    
    # 整数カラムの処理
    for i_idx, col_idx in enumerate(int_col_indices):
        field_idx = row_idx * num_cols + col_idx
        if field_idx >= len(field_offsets):
            continue
            
        pos = field_offsets[field_idx]
        length = field_lengths[field_idx]
        
        if length == -1:  # NULL値
            int_buffers[i_idx][row_idx] = 0
        else:
            # 整数値のデコード
            val = decode_int32(raw_data, pos)
            int_buffers[i_idx][row_idx] = val
    
    # numeric カラムの処理
    for n_idx, col_idx in enumerate(num_col_indices):
        field_idx = row_idx * num_cols + col_idx
        if field_idx >= len(field_offsets):
            continue
            
        pos = field_offsets[field_idx]
        length = field_lengths[field_idx]
        
        if length == -1:  # NULL値
            num_hi_buffers[n_idx][row_idx] = 0
            num_lo_buffers[n_idx][row_idx] = 0
        else:
            # numeric値の128ビット固定小数点数への変換
            decode_numeric_postgres(raw_data, pos, 
                                  num_hi_buffers[n_idx], 
                                  num_lo_buffers[n_idx], 
                                  num_scale_buffers[n_idx],
                                  row_idx)
    
    # 可変長カラムの処理
    for v_idx, col_idx in enumerate(var_col_indices):
        field_idx = row_idx * num_cols + col_idx
        if field_idx >= len(field_offsets):
            continue
            
        pos = field_offsets[field_idx]
        length = field_lengths[field_idx]
        
        if length <= 0:  # NULL値または空文字列
            continue
        
        # オフセット配列から書き込み位置を取得
        dst_pos = var_offsets_buffers[v_idx][row_idx]
        
        # データをコピー
        copy_string_data(raw_data, pos, var_data_buffers[v_idx], dst_pos, length)

class PostgresGpuLoader:
    def __init__(self):
        self.buffer_manager = GpuBufferManager()
        
    def process_table(self, conn, table_name: str, columns: List[ColumnInfo]):
        """テーブルデータをGPUで処理"""
        results = {}
        try:
            # バイナリデータを一時的にメモリに保存（一括読み込み）
            print(f"Loading data from table {table_name}...")
            buffer = io.BytesIO()
            with conn.cursor() as cur:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH (FORMAT binary)", buffer)
            
            # バッファをメモリに固定
            buffer_data = buffer.getvalue()
            buffer = io.BytesIO(buffer_data)
            
            # バッファサイズの確認
            total_size = buffer.getbuffer().nbytes
            print(f"Total binary data size: {total_size} bytes")
            
            # バイナリデータを解析
            raw_data = np.frombuffer(buffer_data, dtype=np.uint8)
            field_offsets, field_lengths = analyze_binary_format(raw_data)
            
            # 行数の計算
            num_fields = len(field_offsets)
            if num_fields == 0:
                print("No data found in table")
                return {}
                
            num_cols = len(columns)
            row_count = num_fields // num_cols
            print(f"Detected {row_count} rows, {num_cols} columns")
            
            # GPUバッファの確保
            self.buffer_manager.allocate_buffers(columns, row_count)
            
            # カラムタイプごとのインデックスリスト
            int_col_indices = []
            num_col_indices = []
            var_col_indices = []
            
            for i, col in enumerate(columns):
                type_id = col.get_type_id()
                if type_id == 0:  # integer
                    int_col_indices.append(i)
                elif type_id == 1:  # numeric
                    num_col_indices.append(i)
                else:  # text/varchar
                    var_col_indices.append(i)
            
            # 各種バッファのリスト
            int_buffers = []
            num_hi_buffers = []
            num_lo_buffers = []
            num_scale_buffers = []
            var_data_buffers = []
            var_offsets_buffers = []
            
            # 整数カラムのバッファ収集
            for i, col_idx in enumerate(int_col_indices):
                col_name = columns[col_idx].name
                int_buffers.append(self.buffer_manager.get_fixed_buffer(col_name))
            
            # numeric カラムのバッファ収集
            for i, col_idx in enumerate(num_col_indices):
                col_name = columns[col_idx].name
                num_hi_buffers.append(self.buffer_manager.get_fixed_buffer(f"{col_name}_hi"))
                num_lo_buffers.append(self.buffer_manager.get_fixed_buffer(f"{col_name}_lo"))
                num_scale_buffers.append(self.buffer_manager.get_fixed_buffer(f"{col_name}_scale"))
            
            # 可変長カラムの処理用バッファ
            if var_col_indices:
                # 可変長データの総サイズを計算
                d_var_col_indices = cuda.to_device(np.array(var_col_indices, dtype=np.int32))
                d_var_total_sizes = cuda.to_device(np.zeros(len(var_col_indices), dtype=np.int32))
                
                # カーネル起動パラメータ
                threads_per_block = 256
                blocks = (row_count + threads_per_block - 1) // threads_per_block
                
                # 可変長データの総サイズを計算
                calculate_var_sizes_kernel[blocks, threads_per_block](
                    cuda.to_device(raw_data),
                    cuda.to_device(field_offsets),
                    cuda.to_device(field_lengths),
                    d_var_col_indices,
                    d_var_total_sizes,
                    row_count,
                    num_cols
                )
                
                # 結果を取得
                var_total_sizes = d_var_total_sizes.copy_to_host()
                
                # 可変長データのオフセット配列を準備
                for i, col_idx in enumerate(var_col_indices):
                    col_name = columns[col_idx].name
                    var_data, var_offsets = self.buffer_manager.get_var_buffers(col_name)
                    
                    # 必要に応じてデータバッファを拡張
                    total_size = var_total_sizes[i]
                    if total_size > var_data.size:
                        # 新しいバッファを確保して古いバッファを解放
                        print(f"Resizing variable-length buffer for {col_name} to {total_size} bytes")
                        new_var_data = cuda.device_array(total_size, dtype=np.uint8)
                        self.buffer_manager.var_data[col_name] = new_var_data
                        var_data = new_var_data
                    
                    # オフセットを計算するために各行のサイズを設定
                    preprocess_var_offsets_kernel[blocks, threads_per_block](
                        cuda.device_array([var_offsets]),
                        cuda.to_device(np.array([col_idx], dtype=np.int32)),
                        cuda.to_device(field_lengths),
                        row_count,
                        num_cols
                    )
                    
                    # 排他的スキャンでオフセットを累積
                    exclusive_scan_kernel[blocks, threads_per_block](
                        var_offsets,
                        var_offsets,
                        row_count + 1
                    )
                    
                    var_data_buffers.append(var_data)
                    var_offsets_buffers.append(var_offsets)
            
            # データのGPU転送
            d_raw_data = cuda.to_device(raw_data)
            d_field_offsets = cuda.to_device(field_offsets)
            d_field_lengths = cuda.to_device(field_lengths)
            
            # カラムタイプのインデックスをGPUに転送
            d_int_col_indices = cuda.to_device(np.array(int_col_indices, dtype=np.int32))
            d_num_col_indices = cuda.to_device(np.array(num_col_indices, dtype=np.int32))
            d_var_col_indices = cuda.to_device(np.array(var_col_indices, dtype=np.int32))
            
            # メインパースカーネルの起動
            threads_per_block = 256
            blocks = (row_count + threads_per_block - 1) // threads_per_block
            
            parse_binary_data_kernel[blocks, threads_per_block](
                d_raw_data, d_field_offsets, d_field_lengths,
                d_int_col_indices, d_num_col_indices, d_var_col_indices,
                int_buffers, num_hi_buffers, num_lo_buffers, num_scale_buffers,
                var_data_buffers, var_offsets_buffers,
                row_count, num_cols
            )
            
            # 結果の回収
            cuda.synchronize()
            
            # 結果の格納
            results = {}
            
            # 整数カラムの結果取得
            for i, col_idx in enumerate(int_col_indices):
                col_name = columns[col_idx].name
                host_array = np.empty(row_count, dtype=np.int32)
                int_buffers[i].copy_to_host(host_array)
                results[col_name] = host_array
            
            # numeric カラムの結果取得
            for i, col_idx in enumerate(num_col_indices):
                col_name = columns[col_idx].name
                
                # 上位64ビット、下位64ビット、スケール値を取得
                hi_array = np.
