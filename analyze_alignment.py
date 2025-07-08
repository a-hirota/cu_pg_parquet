#!/usr/bin/env python3
"""
アライメント修正が正しく動作しているか確認
"""
import subprocess
import json
import re

# テスト実行してワーカーメタデータを取得
print("テスト実行中...")
result = subprocess.run([
    "python", "cu_pg_parquet.py", 
    "--test", "--table", "customer", 
    "--parallel", "2", "--chunks", "1"
], capture_output=True, text=True, env={
    **subprocess.os.environ,
    "GPUPASER_PG_DSN": "host=localhost dbname=postgres user=postgres",
    "GPUPGPARSER_TEST_MODE": "1"
})

# JSON結果を抽出
json_match = re.search(r'===CHUNK_RESULT_JSON===\n(.*?)\n===END_CHUNK_RESULT_JSON===', 
                      result.stdout, re.DOTALL)
if json_match:
    chunk_result = json.loads(json_match.group(1))
    
    print("\nワーカーメタデータ:")
    print("="*60)
    for worker in chunk_result['workers']:
        print(f"Worker {worker['id']}:")
        print(f"  offset: {worker['offset']:,} (0x{worker['offset']:08X})")
        print(f"  size: {worker['size']:,}")
        if 'actual_size' in worker:
            print(f"  actual_size: {worker['actual_size']:,}")
        
        # 64MB境界チェック
        mb_64 = 64 * 1024 * 1024
        if worker['offset'] % mb_64 == 0:
            print(f"  ✅ 64MB境界にアライメント済み")
        else:
            offset_in_mb = worker['offset'] / (1024 * 1024)
            print(f"  ❌ アライメントされていない ({offset_in_mb:.2f}MB)")
        print()
else:
    print("JSONデータが見つかりません")
    print("stdout:", result.stdout[-1000:])  # 最後の1000文字を表示