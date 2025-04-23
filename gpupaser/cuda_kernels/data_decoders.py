"""
PostgreSQLデータ型デコード用のCUDA関数（numeric簡易版実装含む）
"""

from numba import cuda

@cuda.jit(device=True)
def check_bounds(data, pos, size):
    # 範囲チェック
    return pos >= 0 and pos + size <= len(data)

@cuda.jit(device=True)
def decode_int16(data, pos):
    # 2バイト整数デコード（ビッグエンディアン）
    if not check_bounds(data, pos, 2):
        return 0
    b0 = data[pos]
    b1 = data[pos + 1]
    val = ((b0 & 0xFF) << 8) | (b1 & 0xFF)
    if val & 0x8000:
        val = -(((~val) + 1) & 0xFFFF)
    return val

@cuda.jit(device=True)
def decode_int32(data, pos):
    # 4バイト整数デコード（ビッグエンディアン）
    if not check_bounds(data, pos, 4):
        return 0
    b0 = data[pos]
    b1 = data[pos + 1]
    b2 = data[pos + 2]
    b3 = data[pos + 3]
    val = ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)
    if val & 0x80000000:
        val = -(((~val) + 1) & 0xFFFFFFFF)
    return val

@cuda.jit(device=True)
def decode_numeric_postgres(data, pos, hi_out, lo_out, scale_out, row_idx):
    """
    簡易版: numeric型をlo_outのみ正確に変換し、hi_outは0に固定、scale_outにdscaleを設定
    """
    # ヘッダー部分確認
    if not check_bounds(data, pos, 8):
        hi_out[row_idx] = 0
        lo_out[row_idx] = 0
        scale_out[0] = 0
        return

    # ヘッダー情報取得
    ndigits = decode_int16(data, pos)
    weight = decode_int16(data, pos + 2)
    sign = decode_int16(data, pos + 4)
    dscale = decode_int16(data, pos + 6)

    # hi_out固定
    hi_out[row_idx] = 0

    # lo_out累積計算
    lo_val = 0
    base = pos + 8
    for i in range(ndigits):
        digit = decode_int16(data, base + i*2)
        lo_val = lo_val * 10000 + digit

    # weight補正: exponent = weight - (ndigits - 1)
    exp = weight - (ndigits - 1)
    if exp != 0:
        pow_base = 1
        for _ in range(abs(exp)):
            pow_base *= 10000
        if exp > 0:
            lo_val *= pow_base
        else:
            lo_val //= pow_base

    # 符号適用
    if sign == 0x4000:
        lo_val = -lo_val

    lo_out[row_idx] = lo_val
    scale_out[0] = dscale
