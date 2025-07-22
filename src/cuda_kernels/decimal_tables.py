"""
Decimal定数テーブル
==================

Decimal128演算用の10の累乗テーブルを提供
"""

import numpy as np

# ==================================================
# 10^n 定数テーブル (ホスト側で準備)
# ==================================================
# Decimal128は最大38桁なので、10^0 から 10^38 まで対応
_POW10_TABLE_SIZE = 39
POW10_TABLE_LO_HOST = np.zeros(_POW10_TABLE_SIZE, dtype=np.uint64)
POW10_TABLE_HI_HOST = np.zeros(_POW10_TABLE_SIZE, dtype=np.uint64)

for i in range(_POW10_TABLE_SIZE):
    val = 10**i
    POW10_TABLE_LO_HOST[i] = val & 0xFFFFFFFFFFFFFFFF
    POW10_TABLE_HI_HOST[i] = (val >> 64) & 0xFFFFFFFFFFFFFFFF

__all__ = ["POW10_TABLE_LO_HOST", "POW10_TABLE_HI_HOST"]
