#!/usr/bin/env python3
"""
Check Test Implementation Tool
テスト実装チェックツール

このツールは以下を実行します：
1. testPlan.mdの内容を解析
2. 実際のテストファイルと比較
3. 稼働プログラムを正しくテストしているかレビュー
4. テスト実装状況のレポートを生成
"""

import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートへのパスを追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestImplementationChecker:
    """テスト実装チェックツール"""

    def __init__(self):
        self.project_root = project_root
        self.test_plan_path = self.project_root / "docs/testing/testPlan.md"
        self.test_dirs = ["tests"]
        self.planned_tests = {}
        self.implemented_tests = {}
        self.test_quality = {}

    def parse_test_plan(self):
        """testPlan.mdを解析"""
        print("📖 Parsing test plan...")

        if not self.test_plan_path.exists():
            print(f"  ❌ Test plan not found: {self.test_plan_path}")
            return

        with open(self.test_plan_path, "r", encoding="utf-8") as f:
            content = f.read()

        # パイプライン機能のテストを抽出
        functions = re.findall(r"### (F\d+): ([^\n]+)", content)
        for func_id, func_name in functions:
            self.planned_tests[func_id] = {
                "name": func_name,
                "e2e_test": None,
                "unit_tests": [],
                "integration_tests": [],
            }

        # データ型テストを抽出
        type_section = re.search(r"## Supported Data Types.*?(?=##|\Z)", content, re.DOTALL)
        if type_section:
            type_rows = re.findall(
                r"\| (\w+)\s*\| (\d+)\s*\| (\w+)\s*\| ([^|]+)\|", type_section.group()
            )
            self.planned_tests["data_types"] = [
                {"pg_type": row[0], "oid": row[1], "arrow_type": row[2], "status": row[3].strip()}
                for row in type_rows
                if row[0] != "PostgreSQL Type"
            ]

    def analyze_test_files(self):
        """実際のテストファイルを解析"""
        print("\n🔍 Analyzing test implementation...")

        # E2Eテスト
        e2e_dir = self.project_root / "tests/e2e"
        if e2e_dir.exists():
            for test_file in e2e_dir.glob("test_*.py"):
                self._analyze_test_file(test_file, "e2e")

        # 型マトリックステスト
        type_matrix = self.project_root / "tests/test_type_matrix.py"
        if type_matrix.exists():
            self._analyze_test_file(type_matrix, "type_matrix")

        # 統合テスト
        integration_dir = self.project_root / "tests/integration"
        if integration_dir.exists():
            for test_file in integration_dir.glob("test_*.py"):
                self._analyze_test_file(test_file, "integration")

    def _analyze_test_file(self, file_path, test_type):
        """個別のテストファイルを解析"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            rel_path = file_path.relative_to(self.project_root)
            file_info = {
                "path": str(rel_path),
                "type": test_type,
                "imports": self._extract_imports(content),
                "test_classes": [],
                "test_functions": [],
                "uses_production_code": False,
                "uses_mocks": False,
                "coverage": [],
            }

            # テストクラスと関数を抽出
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                        test_methods = [
                            n.name
                            for n in node.body
                            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                        ]
                        file_info["test_classes"].append(
                            {"name": node.name, "methods": test_methods}
                        )
                    elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        file_info["test_functions"].append(node.name)
            except:
                pass

            # プロダクションコードの使用をチェック
            production_imports = [
                "src.",
                "processors.",
                "rust",
                "cuda_kernels",
                "DirectProcessor",
                "parse_postgres_raw_binary",
                "write_parquet_from_cudf",
                "ColumnMeta",
            ]
            for imp in production_imports:
                if imp in content:
                    file_info["uses_production_code"] = True
                    break

            # モックの使用をチェック
            mock_indicators = ["mock", "Mock", "MagicMock", "patch", "@patch"]
            for mock in mock_indicators:
                if mock in content:
                    file_info["uses_mocks"] = True
                    break

            # テスト内容の品質チェック
            quality_checks = {
                "has_assertions": bool(re.search(r"assert\s+", content)),
                "tests_edge_cases": bool(
                    re.search(r"(None|null|empty|boundary|edge|max|min)", content, re.I)
                ),
                "tests_errors": bool(re.search(r"(raises|except|error|fail)", content, re.I)),
                "has_setup_teardown": bool(
                    re.search(r"(setUp|tearDown|setup_|teardown_|fixture)", content)
                ),
                "uses_gpu": bool(re.search(r"(cuda|gpu|cudf|cupy)", content, re.I)),
                "tests_performance": bool(
                    re.search(r"(benchmark|performance|throughput|speed)", content, re.I)
                ),
            }
            file_info["quality_checks"] = quality_checks

            self.implemented_tests[str(rel_path)] = file_info

        except Exception as e:
            print(f"  ⚠️  Error analyzing {file_path}: {e}")

    def _extract_imports(self, content):
        """インポートを抽出"""
        imports = []
        import_lines = re.findall(r"^(from .+ import .+|import .+)$", content, re.MULTILINE)
        for line in import_lines[:10]:  # 最初の10個のみ
            imports.append(line.strip())
        return imports

    def check_test_coverage(self):
        """テストカバレッジをチェック"""
        print("\n📊 Checking test coverage...")

        # 機能ごとのテストカバレッジ
        self.test_quality["function_coverage"] = {
            "F1": self._check_function_coverage("postgres_to_binary"),
            "F2": self._check_function_coverage("binary_to_arrow"),
            "F3": self._check_function_coverage("arrow_to_parquet"),
        }

        # データ型のテストカバレッジ
        type_coverage = self._check_type_coverage()
        self.test_quality["type_coverage"] = type_coverage

        # 全体的な品質メトリクス
        all_tests = list(self.implemented_tests.values())
        self.test_quality["overall"] = {
            "total_test_files": len(all_tests),
            "using_production_code": sum(1 for t in all_tests if t["uses_production_code"]),
            "using_mocks": sum(1 for t in all_tests if t["uses_mocks"]),
            "with_assertions": sum(
                1 for t in all_tests if t.get("quality_checks", {}).get("has_assertions")
            ),
            "testing_edge_cases": sum(
                1 for t in all_tests if t.get("quality_checks", {}).get("tests_edge_cases")
            ),
            "testing_errors": sum(
                1 for t in all_tests if t.get("quality_checks", {}).get("tests_errors")
            ),
            "using_gpu": sum(1 for t in all_tests if t.get("quality_checks", {}).get("uses_gpu")),
        }

    def _check_function_coverage(self, function_name):
        """特定の機能のカバレッジをチェック"""
        coverage = {
            "e2e_test": False,
            "unit_tests": False,
            "integration_tests": False,
            "uses_real_code": False,
            "test_files": [],
        }

        for path, info in self.implemented_tests.items():
            if function_name.lower() in path.lower():
                coverage["test_files"].append(path)
                if info["type"] == "e2e":
                    coverage["e2e_test"] = True
                if info["uses_production_code"]:
                    coverage["uses_real_code"] = True

        return coverage

    def _check_type_coverage(self):
        """データ型のカバレッジをチェック"""
        type_matrix_file = "tests/test_type_matrix.py"
        coverage = {"has_type_matrix": False, "tested_types": [], "coverage_percentage": 0}

        if type_matrix_file in self.implemented_tests:
            coverage["has_type_matrix"] = True
            test_info = self.implemented_tests[type_matrix_file]

            # test_type_matrix.pyの内容を確認
            matrix_path = self.project_root / type_matrix_file
            if matrix_path.exists():
                with open(matrix_path, "r") as f:
                    content = f.read()

                # POSTGRES_TYPESから型を抽出
                types_match = re.search(r"POSTGRES_TYPES\s*=\s*{([^}]+)}", content, re.DOTALL)
                if types_match:
                    type_definitions = re.findall(r"(\d+):\s*\(\"(\w+)\"", types_match.group(1))
                    coverage["tested_types"] = [
                        {"oid": oid, "type": typename} for oid, typename in type_definitions
                    ]

                if self.planned_tests.get("data_types"):
                    planned_count = len(self.planned_tests["data_types"])
                    tested_count = len(coverage["tested_types"])
                    coverage["coverage_percentage"] = (
                        (tested_count / planned_count * 100) if planned_count > 0 else 0
                    )

        return coverage

    def generate_report(self):
        """テスト実装レポートを生成"""
        print("\n📝 Generating implementation report...")

        report = f"""# Test Implementation Check Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report analyzes the test implementation against the test plan and verifies that production code is being properly tested.

## Overall Statistics

- **Total Test Files**: {self.test_quality['overall']['total_test_files']}
- **Tests Using Production Code**: {self.test_quality['overall']['using_production_code']} ({self.test_quality['overall']['using_production_code'] / self.test_quality['overall']['total_test_files'] * 100:.1f}%)
- **Tests Using Mocks**: {self.test_quality['overall']['using_mocks']} ({self.test_quality['overall']['using_mocks'] / self.test_quality['overall']['total_test_files'] * 100:.1f}%)
- **Tests with Assertions**: {self.test_quality['overall']['with_assertions']}
- **Tests Checking Edge Cases**: {self.test_quality['overall']['testing_edge_cases']}
- **Tests Using GPU**: {self.test_quality['overall']['using_gpu']}

## Function Coverage Analysis

"""
        # 機能別カバレッジ
        for func_id, coverage in self.test_quality["function_coverage"].items():
            func_name = self.planned_tests.get(func_id, {}).get("name", func_id)
            report += f"### {func_id}: {func_name}\n\n"

            status_e2e = "✅" if coverage["e2e_test"] else "❌"
            status_real = "✅" if coverage["uses_real_code"] else "❌"

            report += f"- **E2E Test**: {status_e2e}\n"
            report += f"- **Uses Real Code**: {status_real}\n"
            report += f"- **Test Files**: {len(coverage['test_files'])}\n"

            if coverage["test_files"]:
                report += "  - " + "\n  - ".join(coverage["test_files"]) + "\n"
            report += "\n"

        # データ型カバレッジ
        type_cov = self.test_quality["type_coverage"]
        report += "## Data Type Coverage\n\n"
        report += f"- **Type Matrix Test**: {'✅ Implemented' if type_cov['has_type_matrix'] else '❌ Not Found'}\n"
        report += f"- **Tested Types**: {len(type_cov['tested_types'])}\n"
        report += f"- **Coverage**: {type_cov['coverage_percentage']:.1f}%\n\n"

        if type_cov["tested_types"]:
            report += "### Tested Data Types\n\n"
            report += "| OID | Type Name |\n"
            report += "|-----|----------|\n"
            for t in sorted(type_cov["tested_types"], key=lambda x: int(x["oid"])):
                report += f"| {t['oid']} | {t['type']} |\n"
            report += "\n"

        # 詳細な実装状況
        report += "## Detailed Test Implementation\n\n"

        # 実プロダクトコードを使用しているテスト
        report += "### Tests Using Production Code ✅\n\n"
        real_tests = [
            (path, info)
            for path, info in self.implemented_tests.items()
            if info["uses_production_code"]
        ]
        for path, info in sorted(real_tests):
            report += f"#### {path}\n"
            report += f"- Type: {info['type']}\n"
            if info["test_classes"]:
                report += f"- Classes: {len(info['test_classes'])}\n"
                for cls in info["test_classes"][:2]:  # 最初の2クラスのみ
                    report += f"  - {cls['name']} ({len(cls['methods'])} methods)\n"
            quality = info.get("quality_checks", {})
            if quality.get("tests_edge_cases"):
                report += "- ✅ Tests edge cases\n"
            if quality.get("tests_errors"):
                report += "- ✅ Tests error handling\n"
            if quality.get("uses_gpu"):
                report += "- ✅ GPU testing\n"
            report += "\n"

        # モックを使用しているテスト
        report += "### Tests Using Mocks ⚠️\n\n"
        mock_tests = [
            (path, info)
            for path, info in self.implemented_tests.items()
            if info["uses_mocks"] and not info["uses_production_code"]
        ]
        if mock_tests:
            report += "These tests may not be testing real production behavior:\n\n"
            for path, info in sorted(mock_tests):
                report += f"- {path}\n"
        else:
            report += "No tests found that only use mocks without production code.\n"
        report += "\n"

        # 推奨事項
        report += "## Recommendations\n\n"

        recommendations = []

        # E2Eテストの推奨
        for func_id, coverage in self.test_quality["function_coverage"].items():
            if not coverage["e2e_test"]:
                func_name = self.planned_tests.get(func_id, {}).get("name", func_id)
                recommendations.append(f"- Add E2E test for {func_id}: {func_name}")

        # 実コード使用の推奨
        mock_only_count = len(
            [
                t
                for t in self.implemented_tests.values()
                if t["uses_mocks"] and not t["uses_production_code"]
            ]
        )
        if mock_only_count > 0:
            recommendations.append(
                f"- Convert {mock_only_count} mock-only tests to use production code"
            )

        # GPU テストの推奨
        gpu_test_count = self.test_quality["overall"]["using_gpu"]
        if gpu_test_count < 5:
            recommendations.append("- Add more GPU-specific tests for CUDA kernel validation")

        # データ型カバレッジの推奨
        if type_cov["coverage_percentage"] < 80:
            recommendations.append(
                f"- Increase data type test coverage (currently {type_cov['coverage_percentage']:.1f}%)"
            )

        if recommendations:
            for rec in recommendations:
                report += rec + "\n"
        else:
            report += "✅ Test implementation looks comprehensive!\n"

        return report

    def save_report(self, report):
        """レポートを保存"""
        report_path = self.project_root / "docs/testing/test-implementation-report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  ✅ Report saved to: {report_path}")

        # JSON形式でも保存
        json_path = self.project_root / "docs/testing/test-implementation-data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "quality": self.test_quality,
                    "implemented_tests": self.implemented_tests,
                },
                f,
                indent=2,
            )
        print(f"  📊 Data saved to: {json_path}")

    def run(self):
        """メイン実行"""
        print("🔍 Test Implementation Checker")
        print("=" * 60)

        # テスト計画を解析
        self.parse_test_plan()
        print(f"  Found {len(self.planned_tests)} planned test categories")

        # 実装を解析
        self.analyze_test_files()
        print(f"  Analyzed {len(self.implemented_tests)} test files")

        # カバレッジをチェック
        self.check_test_coverage()

        # レポートを生成
        report = self.generate_report()

        # 保存
        self.save_report(report)

        print("\n✨ Test implementation check completed!")

        # サマリーを表示
        print("\n📊 Quick Summary:")
        print(
            f"  - Production code usage: {self.test_quality['overall']['using_production_code']}/{self.test_quality['overall']['total_test_files']} tests"
        )
        print(
            f"  - Type coverage: {self.test_quality['type_coverage']['coverage_percentage']:.1f}%"
        )
        print(f"  - GPU tests: {self.test_quality['overall']['using_gpu']} files")


if __name__ == "__main__":
    checker = TestImplementationChecker()
    checker.run()
