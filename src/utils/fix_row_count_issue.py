#!/usr/bin/env python3
"""
行数不一致問題の修正提案
"""

def analyze_issue():
    """問題の分析と修正提案"""
    
    print("=== 行数不一致問題の分析 ===\n")
    
    print("【問題の原因】")
    print("1. GPUパーサーがatomic操作で行を追加（順序保証なし）")
    print("2. ソートが無効化されている（feat/no-sortブランチ）")
    print("3. 複数スレッドが同じ行を重複検出する可能性")
    print("4. チャンク境界での行の分割処理")
    print()
    
    print("【具体的な問題箇所】")
    print("- postgres_binary_parser.py:422")
    print("  `base_idx = cuda.atomic.add(row_count, 0, local_count)`")
    print("  → 複数スレッドが競合して重複カウントの可能性")
    print()
    print("- postgres_binary_parser.py:825-829")
    print("  ソートが無効化されており、行の順序が保証されない")
    print()
    
    print("【修正案】")
    print()
    print("=== 修正案1: 重複検出の改善 ===")
    print("""
@cuda.jit
def parse_rows_and_fields_lite(
    raw_data, header_size, ncols,
    row_positions, field_offsets, field_lengths, row_count,
    thread_stride, max_rows, fixed_field_lengths
):
    # ... 既存のコード ...
    
    # === 修正: 担当範囲の厳密な管理 ===
    start_pos = uint64(header_size + tid * thread_stride)
    end_pos = uint64(header_size + (tid + 1) * thread_stride)
    
    # 最後のスレッドはデータ終端まで処理
    if tid == cuda.gridsize(1) - 1:
        end_pos = raw_data.size
    
    # === 修正: 行の重複検出を防ぐ ===
    while pos < end_pos:
        candidate_pos = read_uint16_simd16_lite(raw_data, pos, end_pos, ncols)
        
        if candidate_pos == UINT64_MAX:
            break
        
        # 行の開始位置が担当範囲内かチェック
        if candidate_pos < start_pos:
            # 前のスレッドの担当なのでスキップ
            pos = candidate_pos + 1
            continue
        
        # 行の検証と抽出
        is_valid, row_end = validate_and_extract_fields_lite(...)
        
        if is_valid:
            # 行の開始が担当範囲内なら処理
            # （行の終了が範囲外でもOK）
            if candidate_pos >= start_pos and candidate_pos < end_pos:
                if local_count < 256:
                    local_positions[local_count] = uint64(candidate_pos)
                    local_count += 1
            
            pos = row_end
        else:
            pos = candidate_pos + 1
""")
    
    print("\n=== 修正案2: ソートの再有効化 ===")
    print("""
# postgres_binary_parser.py:825付近
if valid_rows > 0:
    # GPUソートを再有効化
    sort_start = time.time()
    
    # CuPyを使用した高速ソート
    import cupy as cp
    
    # 有効な行のみ抽出
    row_positions_gpu = cp.asarray(row_positions[:nrow])
    valid_mask = row_positions_gpu != 0xFFFFFFFFFFFFFFFF
    valid_indices = cp.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        # 有効な行位置でソート
        valid_positions = row_positions_gpu[valid_indices]
        sort_indices = cp.argsort(valid_positions)
        sorted_indices = valid_indices[sort_indices]
        
        # フィールド情報を並べ替え
        field_offsets_sorted = field_offsets[sorted_indices.get()]
        field_lengths_sorted = field_lengths[sorted_indices.get()]
        
        sort_time = time.time() - sort_start
        print(f"GPUソート完了: {len(sorted_indices)}行, {sort_time:.3f}秒")
        
        return field_offsets_sorted, field_lengths_sorted
""")
    
    print("\n=== 修正案3: デバッグモードの追加 ===")
    print("""
# 環境変数でデバッグモードを有効化
export GPUPGPARSER_DEBUG_DUPLICATE=1

# GPUカーネル内で重複検出
@cuda.jit
def detect_duplicate_rows(row_positions, nrows, duplicates):
    tid = cuda.grid(1)
    if tid >= nrows - 1:
        return
    
    # 隣接する行位置をチェック
    if row_positions[tid] == row_positions[tid + 1]:
        cuda.atomic.add(duplicates, 0, 1)
""")
    
    print("\n=== 推奨される実装手順 ===")
    print("1. まず修正案3のデバッグモードで重複を確認")
    print("2. 重複が確認されたら修正案1を実装")
    print("3. ソートの再有効化（修正案2）を検討")
    print("4. 各チャンクで行数を検証")

if __name__ == "__main__":
    analyze_issue()