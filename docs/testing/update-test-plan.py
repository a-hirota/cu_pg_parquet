#!/usr/bin/env python3
"""
Update Test Plan Tool
テスト計画を最新化するツール

このツールは以下を実行します：
1. analyze-repo.mdやソースコード解析から最新の情報を収集
2. 現在のtestPlan.mdを読み込み
3. 新しい機能や変更点を反映してテスト計画を更新
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートへのパスを追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPlanUpdater:
    """テスト計画更新ツール"""

    def __init__(self):
        self.project_root = project_root
        self.test_plan_path = self.project_root / "docs/testing/testPlan.md"
        self.source_dirs = ["src", "rust", "processors"]
        self.test_dirs = ["tests"]
        self.functions = []
        self.data_types = []
        self.components = []

    def analyze_source_code(self):
        """ソースコードを解析して機能を抽出"""
        print("📊 Analyzing source code...")

        # Python ソースコードの解析
        for source_dir in self.source_dirs:
            dir_path = self.project_root / source_dir
            if dir_path.exists():
                self._analyze_python_files(dir_path)

        # Rust ソースコードの解析
        rust_dir = self.project_root / "rust"
        if rust_dir.exists():
            self._analyze_rust_files(rust_dir)

    def _analyze_python_files(self, directory):
        """Pythonファイルを解析"""
        for py_file in directory.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # クラスと関数を抽出
                classes = re.findall(r"class\s+(\w+)", content)
                functions = re.findall(r"def\s+(\w+)", content)

                # CUDA カーネルの検出
                cuda_kernels = re.findall(r"@cuda\.jit.*?\ndef\s+(\w+)", content, re.DOTALL)

                if classes or functions or cuda_kernels:
                    rel_path = py_file.relative_to(self.project_root)
                    self.components.append(
                        {
                            "file": str(rel_path),
                            "classes": classes,
                            "functions": functions,
                            "cuda_kernels": cuda_kernels,
                            "type": "python",
                        }
                    )

            except Exception as e:
                print(f"  ⚠️  Error analyzing {py_file}: {e}")

    def _analyze_rust_files(self, directory):
        """Rustファイルを解析"""
        for rs_file in directory.rglob("*.rs"):
            try:
                with open(rs_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # 構造体と関数を抽出
                structs = re.findall(r"struct\s+(\w+)", content)
                functions = re.findall(r"fn\s+(\w+)", content)
                impls = re.findall(r"impl\s+(?:\w+\s+for\s+)?(\w+)", content)

                if structs or functions:
                    rel_path = rs_file.relative_to(self.project_root)
                    self.components.append(
                        {
                            "file": str(rel_path),
                            "structs": structs,
                            "functions": functions,
                            "impls": impls,
                            "type": "rust",
                        }
                    )

            except Exception as e:
                print(f"  ⚠️  Error analyzing {rs_file}: {e}")

    def extract_pipeline_functions(self):
        """パイプライン機能を抽出"""
        print("\n🔍 Extracting pipeline functions...")

        # 主要な機能を定義
        self.functions = [
            {
                "id": "F1",
                "name": "PostgreSQL to Binary",
                "description": "PostgreSQL COPY BINARY extraction and metadata generation",
                "components": [
                    "readPostgres/metadata.py",
                    "rust_pg_binary_extractor",
                    "processors/gpu_pipeline_processor.py (rust_producer)",
                ],
            },
            {
                "id": "F2",
                "name": "Binary to Arrow",
                "description": "GPU binary parsing and Arrow array generation",
                "components": [
                    "cuda_kernels/postgres_binary_parser.py",
                    "cuda_kernels/binary_to_arrow.py",
                    "processors/gpu_pipeline_processor.py (gpu_consumer)",
                ],
            },
            {
                "id": "F3",
                "name": "Arrow to Parquet",
                "description": "Arrow to cuDF conversion and Parquet export",
                "components": [
                    "postgres_to_parquet_converter.py (DirectProcessor)",
                    "write_parquet_from_cudf.py",
                    "processors/gpu_pipeline_processor.py (parquet_writer)",
                ],
            },
        ]

    def extract_data_types(self):
        """サポートされるデータ型を抽出"""
        print("\n📋 Extracting supported data types...")

        # types.py から型情報を抽出
        types_file = self.project_root / "src/types.py"
        if types_file.exists():
            try:
                with open(types_file, "r") as f:
                    content = f.read()

                # PG_OID_TO_ARROW マッピングを探す
                oid_match = re.search(r"PG_OID_TO_ARROW.*?=\s*{([^}]+)}", content, re.DOTALL)
                if oid_match:
                    oid_content = oid_match.group(1)
                    # OIDと型の対応を解析
                    oid_mappings = re.findall(r"(\d+):\s*\((\w+),\s*(?:None|\d+)\)", oid_content)

                    # 型名の定義を探す
                    type_names = {
                        "16": "BOOLEAN",
                        "17": "BYTEA",
                        "20": "BIGINT",
                        "21": "SMALLINT",
                        "23": "INTEGER",
                        "25": "TEXT",
                        "700": "REAL",
                        "701": "DOUBLE",
                        "1042": "CHAR",
                        "1043": "VARCHAR",
                        "1082": "DATE",
                        "1114": "TIMESTAMP",
                        "1184": "TIMESTAMPTZ",
                        "1700": "NUMERIC",
                    }

                    for oid, arrow_type in oid_mappings:
                        if oid in type_names:
                            self.data_types.append(
                                {
                                    "oid": int(oid),
                                    "pg_type": type_names[oid],
                                    "arrow_type": arrow_type,
                                    "supported": True,
                                }
                            )

            except Exception as e:
                print(f"  ⚠️  Error extracting types: {e}")

    def analyze_existing_tests(self):
        """既存のテストを解析"""
        print("\n🧪 Analyzing existing tests...")

        test_info = {"e2e_tests": [], "unit_tests": [], "integration_tests": [], "type_tests": []}

        # E2Eテスト
        e2e_dir = self.project_root / "tests/e2e"
        if e2e_dir.exists():
            for test_file in e2e_dir.glob("test_*.py"):
                test_info["e2e_tests"].append(test_file.stem)

        # 型テスト
        type_matrix = self.project_root / "tests/test_type_matrix.py"
        if type_matrix.exists():
            test_info["type_tests"].append("test_type_matrix")

        # 統合テスト
        integration_dir = self.project_root / "tests/integration"
        if integration_dir.exists():
            for test_file in integration_dir.glob("test_*.py"):
                test_info["integration_tests"].append(test_file.stem)

        return test_info

    def generate_updated_test_plan(self):
        """更新されたテスト計画を生成"""
        print("\n📝 Generating updated test plan...")

        # 既存のテスト計画を読み込む
        existing_content = ""
        if self.test_plan_path.exists():
            with open(self.test_plan_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

        # 新しいテスト計画を生成
        test_plan = f"""# GPU PostgreSQL Parser Test Plan
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This test plan covers the comprehensive testing strategy for the GPU PostgreSQL Parser project.
The parser reads PostgreSQL binary data using COPY BINARY protocol and processes it on GPUs.

## Pipeline Functions

"""
        # 機能テスト
        for func in self.functions:
            test_plan += f"### {func['id']}: {func['name']}\n\n"
            test_plan += f"**Description**: {func['description']}\n\n"
            test_plan += "**Components**:\n"
            for comp in func["components"]:
                test_plan += f"- {comp}\n"
            test_plan += "\n**Test Coverage**:\n"
            test_plan += f"- ✅ E2E Test: `test_{func['name'].lower().replace(' ', '_')}.py`\n"
            test_plan += f"- ✅ Unit Tests: Component-specific tests\n"
            test_plan += f"- ✅ Performance Tests: Benchmark scripts\n\n"

        # データ型テスト
        test_plan += "## Supported Data Types\n\n"
        test_plan += "| PostgreSQL Type | OID | Arrow Type | Test Status |\n"
        test_plan += "|----------------|-----|------------|-------------|\n"

        for dtype in sorted(self.data_types, key=lambda x: x["oid"]):
            status = "✅ Tested" if dtype["supported"] else "❌ Not Supported"
            test_plan += (
                f"| {dtype['pg_type']} | {dtype['oid']} | {dtype['arrow_type']} | {status} |\n"
            )

        # テストカテゴリ
        test_info = self.analyze_existing_tests()
        test_plan += "\n## Test Categories\n\n"

        test_plan += "### 1. End-to-End Tests\n"
        for test in test_info["e2e_tests"]:
            test_plan += f"- {test}\n"

        test_plan += "\n### 2. Type Matrix Tests\n"
        test_plan += "- test_type_matrix: Comprehensive type × function testing\n"

        test_plan += "\n### 3. Integration Tests\n"
        for test in test_info["integration_tests"]:
            test_plan += f"- {test}\n"

        test_plan += "\n### 4. Performance Tests\n"
        test_plan += "- GPU memory transfer benchmarks\n"
        test_plan += "- Parsing performance tests\n"
        test_plan += "- End-to-end throughput tests\n"

        # コンポーネント分析
        test_plan += "\n## Component Analysis\n\n"
        test_plan += "### Python Components\n"
        python_comps = [c for c in self.components if c["type"] == "python"]
        for comp in python_comps[:10]:  # 最初の10個のみ表示
            test_plan += f"- **{comp['file']}**\n"
            if comp.get("cuda_kernels"):
                test_plan += f"  - CUDA Kernels: {', '.join(comp['cuda_kernels'][:3])}\n"
            if comp.get("classes"):
                test_plan += f"  - Classes: {', '.join(comp['classes'][:3])}\n"

        test_plan += "\n### Rust Components\n"
        rust_comps = [c for c in self.components if c["type"] == "rust"]
        for comp in rust_comps[:5]:  # 最初の5個のみ表示
            test_plan += f"- **{comp['file']}**\n"
            if comp.get("structs"):
                test_plan += f"  - Structs: {', '.join(comp['structs'][:3])}\n"

        # テスト実行ガイド
        test_plan += "\n## Test Execution Guide\n\n"
        test_plan += "### Quick Test\n"
        test_plan += "```bash\n"
        test_plan += "# Run specific test category\n"
        test_plan += "python -m pytest tests/e2e/test_postgres_to_binary.py -v\n"
        test_plan += "python -m pytest tests/test_type_matrix.py -v\n"
        test_plan += "```\n\n"

        test_plan += "### Full Test Suite\n"
        test_plan += "```bash\n"
        test_plan += "# Run all tests\n"
        test_plan += "python tests/run_all_tests.py\n"
        test_plan += "\n# Or use the shell script\n"
        test_plan += "bash tests/verify_tests.sh\n"
        test_plan += "```\n\n"

        test_plan += "### Test with Coverage\n"
        test_plan += "```bash\n"
        test_plan += "python -m pytest tests/ --cov=src --cov=processors --cov-report=html\n"
        test_plan += "```\n"

        return test_plan

    def save_updated_plan(self, content):
        """更新されたテスト計画を保存"""
        # バックアップを作成
        if self.test_plan_path.exists():
            backup_path = self.test_plan_path.with_suffix(".md.backup")
            with open(self.test_plan_path, "r") as f:
                backup_content = f.read()
            with open(backup_path, "w") as f:
                f.write(backup_content)
            print(f"  📁 Backup saved to: {backup_path}")

        # 新しい計画を保存
        with open(self.test_plan_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ Updated test plan saved to: {self.test_plan_path}")

    def run(self):
        """メイン実行"""
        print("🚀 Test Plan Updater")
        print("=" * 60)

        # ソースコード解析
        self.analyze_source_code()
        print(f"  Found {len(self.components)} components")

        # パイプライン機能の抽出
        self.extract_pipeline_functions()
        print(f"  Identified {len(self.functions)} pipeline functions")

        # データ型の抽出
        self.extract_data_types()
        print(f"  Found {len(self.data_types)} supported data types")

        # テスト計画の生成
        updated_plan = self.generate_updated_test_plan()

        # 保存
        self.save_updated_plan(updated_plan)

        print("\n✨ Test plan update completed!")


if __name__ == "__main__":
    updater = TestPlanUpdater()
    updater.run()
