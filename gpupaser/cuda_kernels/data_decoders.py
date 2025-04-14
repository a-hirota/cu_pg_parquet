"""
PostgreSQLデータ型デコード用のCUDA関数
"""

from numba import cuda

@cuda.jit(device=True)
def check_bounds(data, pos, size):
    """
    境界チェック - 指定された範囲がデータ配列の範囲内にあるかを確認
    
    Args:
        data: データ配列
        pos: 開始位置
        size: 読み取るサイズ
        
    Returns:
        範囲内であればTrue、範囲外ならFalse
    """
    return pos >= 0 and pos + size <= len(data)

@cuda.jit(device=True)
def decode_int16(data, pos):
    """
    2バイト整数のデコード（ビッグエンディアン）
    
    Args:
        data: バイナリデータ配列
        pos: 読み取り位置
        
    Returns:
        デコードされた16ビット整数値
    """
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
    """
    4バイト整数のデコード（ビッグエンディアン）
    
    Args:
        data: バイナリデータ配列
        pos: 読み取り位置
        
    Returns:
        デコードされた32ビット整数値
    """
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
    """
    PostgreSQLのnumeric型を128ビット固定小数点数に変換
    
    Args:
        data: バイナリデータ配列
        pos: 読み取り位置
        hi_out: 上位64ビット出力配列
        lo_out: 下位64ビット出力配列
        scale_out: スケール出力配列
        row_idx: 行インデックス
    """
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
