"""
GPU 固定長列 scatter-copy カーネル
-------------------------------------
1 カーネル呼び出しで「ある 1 つの固定長列」を処理します。

Parameters
----------
raw           : uint8[:]      COPYバイナリ全体
field_offsets : int32[:]      列ごとのフィールド先頭オフセット (rows,)
elem_size     : int32         列のバイト幅 (PG COPY BINARYの値サイズ) (2, 4, 8)
dst_buf       : uint8[:]      出力バッファ (rows * stride)
stride        : int32         出力バッファの1行あたりのバイト幅 (8 等。メモリ確保時の値)
"""

from numba import cuda

@cuda.jit # Remove debug=True
def pass2_scatter_fixed(raw, field_offsets, elem_size, dst_buf, stride):
    row = cuda.grid(1)
    if row >= field_offsets.size:
        return

    # Debug print for each thread (limit output in practice)
    # if row < 10: # Print only for first 10 rows
    #    print("pass2_fixed: row=", row, "field_offsets.size=", field_offsets.size)

    src = field_offsets[row]
    dst = row * stride # Calculate destination base offset

    # if row < 10: # Print only for first 10 rows
    #    print("pass2_fixed: row=", row, "src_offset=", src, "dst_base=", dst, "elem_size=", elem_size, "stride=", stride)

    # Check if the offset is 0, which indicates a NULL value
    # Note: parse_binary_format_kernel_one_row sets offset to 0 for NULL
    if src == 0:
        # if row < 10: print("pass2_fixed: row=", row, "NULL detected (src=0), zeroing dst")
        # For NULL values, zero out the destination buffer area for this row.
        # The null bitmap is handled separately in decode_chunk.
        for i in range(stride):
           dst_buf[dst + i] = 0
        return # Skip copy for NULL rows

    # Proceed only for non-NULL rows
    # if row < 10: print("pass2_fixed: row=", row, "Non-NULL, proceeding with copy")

    # PostgreSQL is Big Endian (network order), convert to Little Endian.
    # Write the elem_size bytes directly to the beginning of the stride slot.
    # No padding offset needed anymore as we handle type-specific widths later.
    for i in range(elem_size):
        # Read source bytes in reverse order for endian swap
        # Write target bytes in normal order to the start of the slot
        dst_buf[dst + i] = raw[src + elem_size - 1 - i]

    # Zero out the remaining bytes in the stride slot if elem_size < stride
    # This ensures consistency if the buffer is later read as a larger type (e.g., int64)
    # although the plan is to use correct types during Arrow assembly.
    for i in range(elem_size, stride):
        dst_buf[dst + i] = 0
