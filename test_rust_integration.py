#!/usr/bin/env python3
"""Rust統合のテストスクリプト"""
import os
import sys
import cudf
import psycopg2
from src.rust_integration import PostgresGPUReader, RustStringBuilder
from src.types import ColumnMeta, UTF8, INT32

def setup_test_table():
    """テスト用テーブルの作成"""
    dsn = os.environ.get('GPUPASER_PG_DSN')
    if not dsn:
        print("ERROR: GPUPASER_PG_DSN環境変数が設定されていません")
        sys.exit(1)
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # テスト用テーブル作成
    cur.execute("DROP TABLE IF EXISTS rust_test")
    cur.execute("""
        CREATE TABLE rust_test (
            id INTEGER,
            name TEXT,
            description TEXT
        )
    """)
    
    # テストデータ挿入
    test_data = [
        (1, "Alice", "First test record"),
        (2, "Bob", "Second test record with longer text"),
        (3, "Charlie", "Third record"),
        (4, None, "Record with NULL name"),
        (5, "Eve", None),  # NULL description
    ]
    
    cur.executemany(
        "INSERT INTO rust_test (id, name, description) VALUES (%s, %s, %s)",
        test_data
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✓ テストテーブル作成完了")

def test_postgres_gpu_reader():
    """PostgresGPUReaderのテスト"""
    print("\n=== PostgresGPUReader テスト ===")
    
    reader = PostgresGPUReader()
    
    # 文字列カラムをGPUに転送
    query = "COPY (SELECT name FROM rust_test ORDER BY id) TO STDOUT WITH BINARY"
    
    try:
        gpu_buffers = reader.fetch_string_column_to_gpu(query, column_index=0)
        print(f"✓ GPUバッファ取得成功:")
        print(f"  - データポインタ: 0x{gpu_buffers['data_ptr']:x}")
        print(f"  - データサイズ: {gpu_buffers['data_size']} bytes")
        print(f"  - オフセットポインタ: 0x{gpu_buffers['offsets_ptr']:x}")
        print(f"  - 行数: {gpu_buffers['row_count']}")
        
        # cuDF Series作成
        series = reader.create_cudf_series_from_gpu_buffers(gpu_buffers)
        print(f"\n✓ cuDF Series作成成功:")
        print(series)
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()

def test_rust_string_builder():
    """RustStringBuilderのテスト"""
    print("\n=== RustStringBuilder テスト ===")
    
    # 手動でビルダーを使用
    builder = RustStringBuilder()
    
    test_strings = [
        b"Hello",
        b"World",
        b"GPU",
        b"Processing",
        b"",  # 空文字列
    ]
    
    for s in test_strings:
        builder.add_string(s)
    
    # GPUバッファ構築
    data_gpu, offsets_gpu = builder.build_gpu_buffers()
    print(f"✓ GPUバッファ構築成功:")
    print(f"  - データサイズ: {data_gpu.size} bytes")
    print(f"  - オフセット数: {offsets_gpu.size}")
    
    # cuDF Series構築
    try:
        series = builder.build_cudf_series()
        print(f"\n✓ cuDF Series構築成功:")
        print(series)
    except Exception as e:
        print(f"✗ cuDF Series構築エラー: {e}")

def test_numba_integration():
    """既存のNumba実装との統合テスト"""
    print("\n=== Numba統合テスト ===")
    
    # PostgreSQLからバイナリデータ取得（従来の方法）
    dsn = os.environ.get('GPUPASER_PG_DSN')
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    
    # COPY BINARYでデータ取得
    query = "COPY (SELECT * FROM rust_test ORDER BY id) TO STDOUT WITH BINARY"
    
    import io
    output = io.BytesIO()
    cur.copy_expert(query, output)
    binary_data = output.getvalue()
    output.close()
    
    cur.close()
    conn.close()
    
    print(f"✓ バイナリデータ取得: {len(binary_data)} bytes")
    
    # RustでGPUに転送
    reader = PostgresGPUReader()
    gpu_info = reader.transfer_binary_to_gpu(binary_data)
    
    print(f"✓ GPU転送成功:")
    print(f"  - デバイスポインタ: 0x{gpu_info['device_ptr']:x}")
    print(f"  - サイズ: {gpu_info['size']} bytes")
    
    # TODO: ここで既存のNumbaカーネルを呼び出す

def main():
    print("Rust統合テスト開始")
    print("=" * 50)
    
    # 環境確認
    print("環境確認:")
    print(f"  - Python: {sys.version}")
    
    try:
        import gpupgparser_rust
        print("  - ✓ gpupgparser_rust モジュール: インポート成功")
    except ImportError:
        print("  - ✗ gpupgparser_rust モジュール: 見つかりません")
        print("\nRustモジュールをビルドしてください:")
        print("  cd rust")
        print("  maturin develop")
        sys.exit(1)
    
    # テスト実行
    try:
        setup_test_table()
        test_postgres_gpu_reader()
        test_rust_string_builder()
        test_numba_integration()
        
        print("\n" + "=" * 50)
        print("✓ すべてのテスト完了")
        
    except Exception as e:
        print(f"\n✗ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()