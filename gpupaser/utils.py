"""
共通ユーティリティ関数とデータ構造
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ColumnInfo:
    """カラム情報を保持するクラス"""
    name: str
    type: str
    length: Optional[int] = None

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
    """チャンク処理の設定クラス"""
    def __init__(self, total_rows=6_000_000, rows_per_chunk=None):
        # チャンクサイズを行数に基づいて調整
        if rows_per_chunk is not None:
            # 明示的に行数が指定された場合はそれを使用
            self.rows_per_chunk = min(rows_per_chunk, total_rows)
        else:
            # デフォルトは最大65535行まで（CUDA制限）
            self.rows_per_chunk = min(65535, total_rows)
            
        self.num_chunks = (total_rows + self.rows_per_chunk - 1) // self.rows_per_chunk
        self.threads_per_block = 256  # スレッド数を増加
        self.max_blocks = 65535  # CUDA制限
        
    def get_grid_size(self, chunk_size):
        """グリッドサイズの計算"""
        return min(
            self.max_blocks,
            (chunk_size + self.threads_per_block - 1) // self.threads_per_block
        )
