"""
GPUPGParser Lineorder Table Test
================================

test.mdに基づいたテスト実装
- PostgreSQLのlineorderテーブルからGPU処理で生成したParquetファイルの検証
- 行数の一致確認
- Grid境界スレッドのサンプリング検証
"""

import pytest
import psycopg2
import subprocess
import os
import glob
import cudf
import numpy as np
from typing import Dict, List, Tuple
import json
import tempfile
import shutil


class TestGPUPGParserLineorder:
    """lineorderテーブルのGPU処理テスト"""
    
    @pytest.fixture(scope="class")
    def postgres_connection(self):
        """PostgreSQL接続"""
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres", 
            host="localhost",
            port=5432
        )
        yield conn
        conn.close()
    
    @pytest.fixture(scope="class")
    def expected_row_count(self, postgres_connection):
        """正解値：PostgreSQLから取得した行数"""
        cur = postgres_connection.cursor()
        cur.execute("SELECT count(*) FROM lineorder")
        count = cur.fetchone()[0]
        cur.close()
        return count
    
    @pytest.fixture(scope="class")
    def output_dir(self):
        """テスト用出力ディレクトリ"""
        tmpdir = tempfile.mkdtemp(prefix="gpupgparser_test_")
        yield tmpdir
        # クリーンアップ
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.fixture(scope="class")
    def gpu_processing_result(self, output_dir):
        """GPU処理実行と結果取得"""
        # GPU処理実行
        # 注：実際のテストでは時間がかかるため、チャンク数を減らす
        cmd = [
            "python", "cu_pg_parquet.py",
            "--test",
            "--table", "lineorder",
            "--parallel", "2", 
            "--chunks", "1",
            "--output", output_dir
        ]
        
        # 環境変数設定
        env = os.environ.copy()
        env["RUST_PARALLEL_CONNECTIONS"] = "16"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )
        
        # エラーチェック
        if result.returncode != 0:
            pytest.fail(f"GPU processing failed: {result.stderr}")
        
        # 出力ファイル収集
        parquet_files = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "parquet_files": parquet_files,
            "output_dir": output_dir
        }
    
    def test_row_count_match(self, expected_row_count, gpu_processing_result):
        """テスト1: 行数比較"""
        # GPU版の行数計算
        gpu_row_count = 0
        
        for parquet_file in gpu_processing_result["parquet_files"]:
            df = cudf.read_parquet(parquet_file)
            gpu_row_count += len(df)
            print(f"  {os.path.basename(parquet_file)}: {len(df):,} rows")
        
        print(f"\n正解値の行数: {expected_row_count:,}")
        print(f"GPU版の行数: {gpu_row_count:,}")
        
        # 検証
        assert gpu_row_count == expected_row_count, \
            f"Row count mismatch: expected {expected_row_count}, got {gpu_row_count}"
    
    def test_grid_boundary_threads(self, gpu_processing_result):
        """テスト2: Grid境界スレッドのサンプリング"""
        # テストモードの出力からGrid境界情報を抽出
        stdout = gpu_processing_result["stdout"]
        
        # Grid境界スレッドの情報を探す
        grid_info = self._extract_grid_boundary_info(stdout)
        
        if not grid_info:
            pytest.skip("Grid boundary thread information not found in output")
        
        # 検証項目
        print("\n=== Grid境界スレッドサンプリング ===")
        for thread_id, info in grid_info.items():
            print(f"\nThread {thread_id}:")
            print(f"  - Field offsets: {info.get('field_offsets', 'N/A')}")
            print(f"  - Field lengths: {info.get('field_lengths', 'N/A')}")
            print(f"  - Binary data (±20B): {info.get('binary_context', 'N/A')}")
            print(f"  - validate_and_extract_fields_lite result: {info.get('validate_result', 'N/A')}")
            print(f"  - parse_rows_and_fields_lite result: {info.get('parse_result', 'N/A')}")
    
    def test_sample_row_validation(self, postgres_connection, gpu_processing_result):
        """テスト3: サンプル行の検証（lo_orderkey, lo_linenumber）"""
        # 最初のParquetファイルからサンプル行を取得
        if not gpu_processing_result["parquet_files"]:
            pytest.skip("No parquet files generated")
        
        # サンプル行を読み込み
        df_sample = cudf.read_parquet(gpu_processing_result["parquet_files"][0], nrows=10)
        
        if "lo_orderkey" not in df_sample.columns or "lo_linenumber" not in df_sample.columns:
            pytest.skip("lo_orderkey or lo_linenumber columns not found")
        
        # PostgreSQLから同じ行を取得して比較
        cur = postgres_connection.cursor()
        
        for idx in range(min(5, len(df_sample))):
            orderkey = int(df_sample["lo_orderkey"].iloc[idx])
            linenumber = int(df_sample["lo_linenumber"].iloc[idx])
            
            # PostgreSQLから該当行を取得
            cur.execute(
                "SELECT * FROM lineorder WHERE lo_orderkey = %s AND lo_linenumber = %s",
                (orderkey, linenumber)
            )
            
            pg_row = cur.fetchone()
            if pg_row:
                print(f"\nサンプル行 {idx+1}:")
                print(f"  lo_orderkey: {orderkey}, lo_linenumber: {linenumber}")
                print(f"  PostgreSQL row found: ✓")
            else:
                pytest.fail(f"Row not found in PostgreSQL: orderkey={orderkey}, linenumber={linenumber}")
        
        cur.close()
    
    def _extract_grid_boundary_info(self, stdout: str) -> Dict:
        """標準出力からGrid境界スレッド情報を抽出"""
        # TODO: 実際の出力フォーマットに合わせて実装
        # 現時点では仮実装
        grid_info = {}
        
        # 出力にGrid境界情報が含まれているか確認
        if "Grid boundary thread" in stdout:
            # パース処理を実装
            pass
        
        return grid_info


def test_cu_pg_parquet_exists():
    """cu_pg_parquet.pyが存在することを確認"""
    assert os.path.exists("cu_pg_parquet.py"), "cu_pg_parquet.py not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])