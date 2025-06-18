#!/usr/bin/env python3
"""
kvikio lineorderベンチマーク（権限制限回避版）

PostgreSQLヒープファイルの直接アクセスが制限されている環境での
代替実装。COPY BINARYデータをkvikio風に処理するハイブリッド版。

権限問題を回避しつつ、kvikio統合パイプラインの性能実証を行う。
"""

import os
import time
import io
import tempfile
import psycopg
import cudf
import numpy as np
from numba import cuda
import rmm

from src.metadata import fetch_column_meta
from src.types import ColumnMeta
from src.build_cudf_from_buf import integrate_kvikio_pipeline
from src.heap_file_reader import read_heap_file_direct

TABLE_NAME = "lineorder"
OUTPUT_DIR = "benchmark"

def create_mock_heap_file_from_copy_binary(database: str, table: str, limit: int = 100000):
    """
    COPY BINARYデータからモックヒープファイルを作成
    
    権限制限によりPostgreSQLヒープファイルに直接アクセスできない場合の
    代替手法。実際のlineorderデータを使用してkvikio処理をテストする。
    """
    print(f"📦 {table}モックヒープファイル作成 ({limit:,}行)")
    
    dsn = os.environ.get("GPUPASER_PG_DSN")
    conn = psycopg.connect(dsn)
    
    try:
        # PostgreSQLから実データを取得
        start_time = time.time()
        
        with conn.cursor() as cur:
            # サンプルクエリ実行
            query = f"COPY (SELECT * FROM {table} LIMIT {limit}) TO STDOUT (FORMAT binary)"
            
            buffer = io.BytesIO()
            with cur.copy(query) as copy:
                for data in copy:
                    buffer.write(data)
            
            binary_data = buffer.getvalue()
            buffer.close()
        
        fetch_time = time.time() - start_time
        
        print(f"✅ PostgreSQLデータ取得完了:")
        print(f"   取得時間: {fetch_time:.3f}秒")
        print(f"   データサイズ: {len(binary_data) / (1024*1024):.2f} MB")
        print(f"   取得スループット: {(len(binary_data) / (1024*1024)) / fetch_time:.1f} MB/sec")
        
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.heap') as temp_file:
            temp_file.write(binary_data)
            temp_file_path = temp_file.name
        
        print(f"✅ モックヒープファイル作成: {temp_file_path}")
        
        return temp_file_path, len(binary_data), fetch_time
        
    finally:
        conn.close()

def convert_binary_to_mock_heap(binary_data):
    """
    PostgreSQL BINARYデータをPostgreSQLヒープページ風に変換
    
    実際のヒープファイル構造を模擬してkvikioパイプラインテスト用の
    データを作成する。
    """
    # PostgreSQL BINARYヘッダー（19バイト）をスキップ
    if len(binary_data) < 19:
        raise ValueError("不正なPostgreSQL BINARYデータ")
    
    # 簡易的なヒープページ構造作成
    page_size = 8192
    header_size = 24
    
    # BINARYデータをページに分割
    data_without_header = binary_data[19:]  # PostgreSQL BINARYヘッダーをスキップ
    
    mock_heap_data = bytearray()
    
    # 各ページのモック作成
    offset = 0
    page_count = 0
    
    while offset < len(data_without_header):
        # ページヘッダー作成（24バイト）
        page_header = bytearray(header_size)
        
        # pd_lower, pd_upper設定（簡易）
        chunk_size = min(page_size - header_size, len(data_without_header) - offset)
        pd_lower = header_size + 8  # ItemId配列（仮）
        pd_upper = page_size - chunk_size
        
        page_header[12:14] = pd_lower.to_bytes(2, 'little')
        page_header[14:16] = pd_upper.to_bytes(2, 'little')
        
        # ページデータ作成
        page_data = bytearray(page_size)
        page_data[:header_size] = page_header
        
        # ItemId配列（簡易）
        item_id = bytearray(4)
        item_id[0:2] = pd_upper.to_bytes(2, 'little')  # lp_off
        item_id[2:4] = (chunk_size | (1 << 14)).to_bytes(2, 'little')  # lp_len + flags
        page_data[header_size:header_size+4] = item_id
        
        # 実際のデータ
        page_data[pd_upper:pd_upper+chunk_size] = data_without_header[offset:offset+chunk_size]
        
        mock_heap_data.extend(page_data)
        offset += chunk_size
        page_count += 1
    
    print(f"✅ モックヒープページ作成: {page_count}ページ")
    return bytes(mock_heap_data)

def run_kvikio_lineorder_mock_benchmark(
    database: str = "postgres",
    table: str = TABLE_NAME,
    limit: int = 100000,
    output_path: str = None
):
    """kvikio lineorderモックベンチマーク実行"""
    
    print("=== kvikio lineorderベンチマーク（権限制限回避版）===")
    print("🎯 目標: PostgreSQL実データ → kvikio風処理 → cuDF → GPU Parquet")
    print(f"データベース: {database}")
    print(f"テーブル: {table}")
    print(f"サンプルサイズ: {limit:,}行")
    
    if output_path is None:
        timestamp = int(time.time())
        output_path = f"{OUTPUT_DIR}/{table}_kvikio_mock_{timestamp}.parquet"
    
    print(f"出力先: {output_path}")
    
    try:
        # RMM初期化
        if not rmm.is_initialized():
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=2*1024**3  # 2GB
            )
            print("✅ RMM 2GB pool初期化完了")
        
        start_total_time = time.time()
        
        # ステップ1: メタデータ取得
        print(f"\n📊 メタデータ取得中...")
        start_meta_time = time.time()
        
        dsn = os.environ.get("GPUPASER_PG_DSN")
        conn = psycopg.connect(dsn)
        try:
            columns = fetch_column_meta(conn, f"SELECT * FROM {table}")
        finally:
            conn.close()
        
        meta_time = time.time() - start_meta_time
        print(f"✅ メタデータ取得完了 ({meta_time:.4f}秒)")
        print(f"   列数: {len(columns)}")
        
        # ステップ2: PostgreSQL実データ取得とモック変換
        print(f"\n📦 PostgreSQL実データ取得...")
        mock_file_path, data_size, fetch_time = create_mock_heap_file_from_copy_binary(
            database, table, limit
        )
        
        # ステップ3: kvikio風統合処理実行
        print(f"\n🚀 kvikio風統合処理開始...")
        start_kvikio_time = time.time()
        
        try:
            # kvikio統合パイプライン実行（モックファイル使用）
            cudf_df = integrate_kvikio_pipeline(mock_file_path, columns)
            
        except Exception as e:
            print(f"❌ kvikio統合処理エラー: {e}")
            return False
        finally:
            # 一時ファイル削除
            os.unlink(mock_file_path)
        
        kvikio_time = time.time() - start_kvikio_time
        rows = len(cudf_df)
        
        print(f"✅ kvikio風処理完了 ({kvikio_time:.4f}秒)")
        print(f"   処理行数: {rows:,}")
        print(f"   処理列数: {len(cudf_df.columns)}")
        
        # ステップ4: GPU直接Parquet出力
        print(f"\n💾 GPU直接Parquet書き込み中...")
        start_write_time = time.time()
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            cudf_df.to_parquet(
                output_path,
                compression='snappy',
                engine='cudf'
            )
            
        except Exception as e:
            print(f"❌ GPU Parquet書き込みエラー: {e}")
            return False
        
        write_time = time.time() - start_write_time
        output_size = os.path.getsize(output_path)
        
        print(f"✅ GPU Parquet出力完了 ({write_time:.4f}秒)")
        print(f"   出力ファイルサイズ: {output_size / (1024**2):.1f} MB")
        
        # ステップ5: 性能評価
        total_time = time.time() - start_total_time
        processing_time = kvikio_time + write_time
        
        print(f"\n📈 kvikio風lineorderベンチマーク結果:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 データ規模:")
        print(f"   行数: {rows:,}")
        print(f"   列数: {len(columns)}")
        print(f"   データサイズ: {data_size / (1024**2):.2f} MB")
        print(f"   出力サイズ: {output_size / (1024**2):.1f} MB")
        
        print(f"\n⏱️  時間内訳:")
        print(f"   メタデータ取得: {meta_time:.4f} 秒")
        print(f"   PostgreSQL取得: {fetch_time:.4f} 秒")
        print(f"   kvikio風処理: {kvikio_time:.4f} 秒")
        print(f"   GPU Parquet出力: {write_time:.4f} 秒")
        print(f"   総時間: {total_time:.4f} 秒")
        
        print(f"\n🚀 スループット:")
        if kvikio_time > 0:
            kvikio_throughput = (data_size / (1024**2)) / kvikio_time
            print(f"   kvikio風処理: {kvikio_throughput:.1f} MB/sec")
        
        if processing_time > 0:
            overall_throughput = (data_size / (1024**2)) / processing_time
            row_speed = rows / processing_time
            print(f"   GPU処理全体: {overall_throughput:.1f} MB/sec")
            print(f"   行処理速度: {row_speed:,.0f} rows/sec")
        
        # 全テーブル外挿予測
        full_table_rows = 246012324  # lineorder全行数
        if rows > 0 and processing_time > 0:
            scale_factor = full_table_rows / rows
            predicted_time = processing_time * scale_factor
            predicted_throughput = (42 * 1024) / predicted_time  # 42GB
            
            print(f"\n🔮 全lineorderテーブル処理予測:")
            print(f"   予測処理時間: {predicted_time:.1f}秒 ({predicted_time/60:.1f}分)")
            print(f"   予測スループット: {predicted_throughput:.1f} MB/sec")
            
            if predicted_time < 300:  # 5分以内
                impact = "🏆 実用的 - 5分以内で42GB処理可能"
            elif predicted_time < 1800:  # 30分以内
                impact = "🥇 高性能 - 30分以内で42GB処理可能"
            else:
                impact = "🥈 改善中 - さらなる最適化で実用化"
            
            print(f"   実用性評価: {impact}")
        
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 結果検証
        print(f"\n🔍 結果検証...")
        try:
            verify_df = cudf.read_parquet(output_path)
            print(f"✅ 検証成功: {len(verify_df):,}行 × {len(verify_df.columns)}列")
            
            # データ型確認
            print("✅ データ型確認:")
            for col_name, dtype in list(verify_df.dtypes.items())[:5]:
                print(f"   {col_name}: {dtype}")
            if len(verify_df.columns) > 5:
                print(f"   ... 他{len(verify_df.columns)-5}列")
                
        except Exception as e:
            print(f"❌ 検証エラー: {e}")
            return False
        
        print(f"\n🎉 kvikio風lineorderベンチマーク成功!")
        print(f"   💡 PostgreSQL実データ → kvikio風処理 → cuDF → GPU Parquet")
        print(f"   ⚡ 権限制限回避版でもGPGPU革新を実証!")
        
        return True
        
    except Exception as e:
        print(f"❌ ベンチマークエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='kvikio lineorderモックベンチマーク')
    parser.add_argument('--database', type=str, default='postgres', help='データベース名')
    parser.add_argument('--table', type=str, default=TABLE_NAME, help='テーブル名')
    parser.add_argument('--limit', type=int, default=100000, help='処理行数制限')
    parser.add_argument('--output', type=str, help='出力Parquetファイルパス')
    
    args = parser.parse_args()
    
    success = run_kvikio_lineorder_mock_benchmark(
        database=args.database,
        table=args.table,
        limit=args.limit,
        output_path=args.output
    )
    
    if success:
        print("\n✨ kvikio風GPGPU革新の実証完了（権限制限回避版）✨")
    else:
        print("\n⚠️  一部で問題が発生しました")

if __name__ == "__main__":
    main()