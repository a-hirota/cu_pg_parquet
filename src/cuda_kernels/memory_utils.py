"""
CUDA用メモリ操作ユーティリティ関数
"""

import numpy as np
from numba import cuda

@cuda.jit(device=True)
def bulk_copy_64bytes(src, src_pos, dst, dst_pos, size):
    """
    64バイト単位でのバルクコピー
    
    GPU上での効率的なメモリコピー処理のための関数です。
    最大64バイトのデータを一度にコピーします。
    
    Args:
        src: コピー元データ配列
        src_pos: コピー元の開始位置
        dst: コピー先データ配列
        dst_pos: コピー先の開始位置
        size: コピーするバイト数（最大64）
    """
    # 型の一貫性を保証
    size_val = np.int32(size)
    if size_val > 64:
        size_val = np.int32(64)
    
    # 8バイトずつコピー
    for i in range(0, size_val, 8):
        if i + 8 <= size_val:
            # 8バイトを一度に読み書き
            val = np.int32(0)
            for j in range(8):
                val = np.int32((val << 8) | np.int32(src[src_pos + i + j]))
            
            # 8バイトを一度に書き込み
            for j in range(8):
                dst[dst_pos + i + j] = np.uint8((val >> ((7-j) * 8)) & 0xFF)
        else:
            # 残りのバイトを1バイトずつコピー
            for j in range(size_val - i):
                dst[dst_pos + i + j] = src[src_pos + i + j]
