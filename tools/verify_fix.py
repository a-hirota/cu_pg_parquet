#!/usr/bin/env python3
"""修正が正しく適用されているか確認"""

import os

def check_fix_applied():
    """修正が適用されているか確認"""
    
    print("=== 修正の適用状況確認 ===\n")
    
    # postgres_to_cudf.pyの修正を確認
    file_path = "src/postgres_to_cudf.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 問題のあるパターンを検索
    bad_pattern = "if is_null or src_offset == 0:"
    good_pattern = "if is_null:"
    
    bad_count = content.count(bad_pattern)
    good_count = content.count(good_pattern)
    
    print(f"ファイル: {file_path}")
    print(f"  - 問題のあるパターン '{bad_pattern}': {bad_count}回")
    print(f"  - 修正後のパターン '{good_pattern}': {good_count}回")
    
    if bad_count > 0:
        print("\n❌ まだ修正が必要な箇所があります！")
        
        # 該当行を表示
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if bad_pattern in line:
                print(f"\n  行{i+1}: {line.strip()}")
    else:
        print("\n✅ 修正が正しく適用されています！")
    
    # postgres_binary_parser.pyの修正も確認
    parser_file = "src/cuda_kernels/postgres_binary_parser.py"
    
    print(f"\n\nファイル: {parser_file}")
    
    with open(parser_file, 'r') as f:
        parser_content = f.read()
    
    # NULLフィールドのオフセット処理を確認
    if "field_offsets_out[field_idx] = uint32(0)" in parser_content:
        print("  ❌ NULLフィールドのオフセットが0に設定されています（要修正）")
    elif "field_offsets_out[field_idx] = relative_offset" in parser_content and "if flen == 0xFFFFFFFF:" in parser_content:
        print("  ✅ NULLフィールドのオフセットが正しく設定されています")
    else:
        print("  ⚠️  NULLフィールドの処理を確認できませんでした")

if __name__ == "__main__":
    check_fix_applied()