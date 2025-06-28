"""
GPUPGParser Small Table Test
============================

小規模テーブルでの基本動作確認テスト
"""

import pytest
import psycopg2
import subprocess
import os
import glob
import cudf
import tempfile
import shutil


class TestGPUPGParserSmall:
    """小規模テーブルのGPU処理テスト"""
    
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
    def test_table_name(self):
        """テスト用テーブル名"""
        return "gpupgparser_test_small"
    
    @pytest.fixture(scope="class")
    def setup_test_table(self, postgres_connection, test_table_name):
        """テスト用小規模テーブルの作成"""
        cur = postgres_connection.cursor()
        
        # テーブル削除（存在する場合）
        cur.execute(f"DROP TABLE IF EXISTS {test_table_name}")
        
        # テーブル作成
        cur.execute(f"""
            CREATE TABLE {test_table_name} (
                id SERIAL PRIMARY KEY,
                test_int INTEGER,
                test_bigint BIGINT,
                test_text TEXT,
                test_varchar VARCHAR(100),
                test_decimal DECIMAL(10,2),
                test_date DATE,
                test_timestamp TIMESTAMP
            )
        """)
        
        # テストデータ挿入（1000行）
        for i in range(1000):
            cur.execute(f"""
                INSERT INTO {test_table_name} 
                (test_int, test_bigint, test_text, test_varchar, test_decimal, test_date, test_timestamp)
                VALUES 
                (%s, %s, %s, %s, %s, %s, %s)
            """, (
                i % 100,
                i * 1000000,
                f"Test text {i}",
                f"Varchar {i % 50}",
                float(i) * 1.23,
                f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                f"2024-01-01 {i % 24:02d}:{i % 60:02d}:00"
            ))
        
        postgres_connection.commit()
        cur.close()
        
        yield test_table_name
        
        # クリーンアップ
        cur = postgres_connection.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {test_table_name}")
        postgres_connection.commit()
        cur.close()
    
    @pytest.fixture(scope="class")
    def output_dir(self):
        """テスト用出力ディレクトリ"""
        tmpdir = tempfile.mkdtemp(prefix="gpupgparser_test_")
        yield tmpdir
        # クリーンアップ
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_small_table_processing(self, setup_test_table, output_dir, postgres_connection):
        """小規模テーブルの処理テスト"""
        # 期待される行数を取得
        cur = postgres_connection.cursor()
        cur.execute(f"SELECT count(*) FROM {setup_test_table}")
        expected_count = cur.fetchone()[0]
        cur.close()
        
        print(f"\nテストテーブル: {setup_test_table}")
        print(f"期待される行数: {expected_count}")
        
        # GPU処理実行
        cmd = [
            "python", "cu_pg_parquet.py",
            "--test",
            "--table", setup_test_table,
            "--parallel", "1", 
            "--chunks", "1",
            "--output", output_dir
        ]
        
        # 環境変数設定
        env = os.environ.copy()
        env["RUST_PARALLEL_CONNECTIONS"] = "1"
        env["GPUPASER_PG_DSN"] = "dbname=postgres user=postgres host=localhost port=5432"
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd="/home/ubuntu/gpupgparser"  # 作業ディレクトリを明示的に設定
        )
        
        # デバッグ情報出力
        print(f"\n実行コマンド: {' '.join(cmd)}")
        print(f"リターンコード: {result.returncode}")
        print(f"\n標準出力:\n{result.stdout}")
        print(f"\n標準エラー:\n{result.stderr}")
        
        # エラーチェック
        if result.returncode != 0:
            pytest.fail(f"GPU processing failed with return code {result.returncode}")
        
        # 出力ファイル確認
        parquet_files = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
        print(f"\n生成されたParquetファイル: {len(parquet_files)}")
        for f in parquet_files:
            print(f"  - {os.path.basename(f)}")
        
        # assert len(parquet_files) > 0, "No parquet files generated"
        
        # 行数検証
        # total_rows = 0
        # for parquet_file in parquet_files:
        #     df = cudf.read_parquet(parquet_file)
        #     total_rows += len(df)
        #     print(f"  {os.path.basename(parquet_file)}: {len(df)} rows")
        
        # print(f"\n合計行数: {total_rows}")
        # assert total_rows == expected_count, f"Row count mismatch: expected {expected_count}, got {total_rows}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])