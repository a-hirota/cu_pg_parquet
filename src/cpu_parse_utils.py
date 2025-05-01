import numpy as np

# CPUで複数行の開始位置を計算するヘルパー関数 (クリーンアップ版)
def calculate_row_starts_cpu(raw_data, header_size, num_rows):
    """
    CPU上でCOPY BINARYデータの各行の開始位置を計算します。
    ヘッダー後の潜在的なパディング/フラグをスキップし、
    各行のフィールドを正しく読み進めて次の行の開始位置を特定します。
    """
    row_starts = np.full(num_rows, -1, dtype=np.int32) # Initialize with -1 (invalid)
    pos = header_size
    array_size = len(raw_data)
    current_row_index = 0

    while current_row_index < num_rows and pos < array_size:
        # --- Find the actual start of the row data ---
        found_start = False
        original_search_pos = pos
        while pos + 2 <= array_size:
            potential_num_fields = (raw_data[pos] << 8) | raw_data[pos + 1]

            if potential_num_fields == 0xFFFF: # EOF marker
                 pos = array_size # Stop processing
                 break

            # Check for a reasonable number of fields
            if potential_num_fields > 0 and potential_num_fields < 1000: # Adjust 1000 if needed
                 found_start = True
                 break # Found the likely start

            # If not a valid field count or EOF, advance by one byte and retry
            pos += 1

        if not found_start or pos >= array_size:
             break # Stop if no valid start found or reached end

        # --- Process the row starting at the found 'pos' ---
        row_starts[current_row_index] = pos

        # Read the confirmed number of fields
        if pos + 2 > array_size: # Should not happen if found_start is True, but safety check
            row_starts[current_row_index] = -1 # Mark as invalid
            break

        num_fields = (raw_data[pos] << 8) | raw_data[pos + 1]
        pos += 2 # Advance past num_fields

        # --- Inner loop to skip fields and find next row's start ---
        inner_loop_broken = False
        for field_idx in range(num_fields):
            # Check if we can read field length
            if pos + 4 > array_size:
                inner_loop_broken = True
                pos = array_size # Ensure outer loop terminates
                break # Break inner loop

            # Read field length
            field_len = int.from_bytes(raw_data[pos:pos+4], 'big', signed=True)
            pos += 4 # Advance past field_len

            # If field is not NULL, advance past data
            if field_len >= 0:
                # Check if we can read field data
                if pos + field_len > array_size:
                    inner_loop_broken = True
                    pos = array_size # Ensure outer loop terminates
                    break # Break inner loop
                pos += field_len # Advance past field data
            # else: # NULL case, pos already advanced by 4

        # Move to the next row index
        current_row_index += 1

        # If the inner loop was broken due to data ending, stop processing further rows
        if inner_loop_broken:
             break

    # Mark any remaining rows (if loop exited early) as invalid
    if current_row_index < num_rows:
        row_starts[current_row_index:] = -1

    return row_starts
