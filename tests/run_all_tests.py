#!/usr/bin/env python3
"""
Test Result Verification Tool

This script runs all tests and provides a comprehensive summary of results.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class TestRunner:
    """Run and verify all test results."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_categories": {},
            "summary": {"total_tests": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        }

    def run_test_category(self, category, test_path):
        """Run tests for a specific category."""
        print(f"\n{'='*60}")
        print(f"Running {category} tests...")
        print(f"{'='*60}")

        cmd = [
            "python",
            "-m",
            "pytest",
            test_path,
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--json-report",
            f"--json-report-file=/tmp/pytest_{category}.json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse results
        category_results = {
            "status": "passed" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "tests": [],
        }

        # Try to parse JSON report
        json_report_path = Path(f"/tmp/pytest_{category}.json")
        if json_report_path.exists():
            try:
                with open(json_report_path) as f:
                    report = json.load(f)

                category_results["duration"] = report.get("duration", 0)
                category_results["tests"] = report.get("tests", [])

                # Update summary
                summary = report.get("summary", {})
                self.results["summary"]["total_tests"] += summary.get("total", 0)
                self.results["summary"]["passed"] += summary.get("passed", 0)
                self.results["summary"]["failed"] += summary.get("failed", 0)
                self.results["summary"]["skipped"] += summary.get("skipped", 0)

                # Clean up JSON file
                json_report_path.unlink()
            except Exception as e:
                print(f"Warning: Could not parse JSON report: {e}")

        # Parse output for basic info if JSON not available
        if not category_results["tests"]:
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "passed" in line or "failed" in line or "skipped" in line:
                    # Basic parsing of pytest output
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        try:
                            if "passed" in line:
                                self.results["summary"]["passed"] += int(parts[0])
                                self.results["summary"]["total_tests"] += int(parts[0])
                            elif "failed" in line:
                                self.results["summary"]["failed"] += int(parts[0])
                                self.results["summary"]["total_tests"] += int(parts[0])
                            elif "skipped" in line:
                                self.results["summary"]["skipped"] += int(parts[0])
                                self.results["summary"]["total_tests"] += int(parts[0])
                        except ValueError:
                            pass

        self.results["test_categories"][category] = category_results

        # Print immediate results
        if result.returncode == 0:
            print(f"✅ {category} tests PASSED")
        else:
            print(f"❌ {category} tests FAILED")
            if self.verbose and result.stderr:
                print("STDERR:", result.stderr)

        return result.returncode == 0

    def run_all_tests(self):
        """Run all test categories."""
        test_categories = [
            ("Setup", "tests/setup"),
            ("E2E Function 0", "tests/e2e/test_function0.py"),
            ("E2E Function 1", "tests/e2e/test_function1.py"),
            ("E2E Function 2", "tests/e2e/test_function2.py"),
            ("E2E Function 3", "tests/e2e/test_function3.py"),
            ("E2E Function 4", "tests/e2e/test_function4.py"),
            ("Numeric Types", "tests/datatypes/test_numeric_types.py"),
            ("String Types", "tests/datatypes/test_string_types.py"),
            ("DateTime Types", "tests/datatypes/test_datetime_types.py"),
            ("Other Types", "tests/datatypes/test_other_types.py"),
            ("Integration", "tests/integration"),
        ]

        all_passed = True

        for category, path in test_categories:
            test_path = Path(path)
            if test_path.exists():
                passed = self.run_test_category(category, str(test_path))
                all_passed = all_passed and passed
            else:
                print(f"⚠️  Skipping {category} - path not found: {path}")

        return all_passed

    def print_summary(self):
        """Print test result summary."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        summary = self.results["summary"]
        total = summary["total_tests"]

        if total > 0:
            print(f"Total Tests: {total}")
            print(f"✅ Passed: {summary['passed']} ({summary['passed']/total*100:.1f}%)")
            print(f"❌ Failed: {summary['failed']} ({summary['failed']/total*100:.1f}%)")
            print(f"⏭️  Skipped: {summary['skipped']} ({summary['skipped']/total*100:.1f}%)")

            if summary["errors"] > 0:
                print(f"💥 Errors: {summary['errors']}")
        else:
            print("No tests were run!")

        print("\nCategory Results:")
        for category, results in self.results["test_categories"].items():
            status_icon = "✅" if results["status"] == "passed" else "❌"
            duration = results.get("duration", 0)
            print(f"  {status_icon} {category}: {results['status']} ({duration:.2f}s)")

        # Overall result
        print("\n" + "=" * 60)
        if summary["failed"] == 0 and summary["errors"] == 0 and total > 0:
            print("🎉 ALL TESTS PASSED! 🎉")
            return True
        else:
            print("❌ TESTS FAILED")
            return False

    def save_results(self, output_file="test_results.json"):
        """Save detailed results to JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run all tests and verify results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-o", "--output", default="test_results.json", help="Output file for detailed results"
    )
    parser.add_argument("--category", help="Run only specific test category")

    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    runner = TestRunner(verbose=args.verbose)

    if args.category:
        # Run specific category
        test_path = {
            "setup": "tests/setup",
            "function0": "tests/e2e/test_function0.py",
            "function1": "tests/e2e/test_function1.py",
            "function2": "tests/e2e/test_function2.py",
            "function3": "tests/e2e/test_function3.py",
            "function4": "tests/e2e/test_function4.py",
            "numeric": "tests/datatypes/test_numeric_types.py",
            "string": "tests/datatypes/test_string_types.py",
            "datetime": "tests/datatypes/test_datetime_types.py",
            "other": "tests/datatypes/test_other_types.py",
            "integration": "tests/integration",
        }.get(args.category.lower())

        if test_path:
            runner.run_test_category(args.category, test_path)
        else:
            print(f"Unknown category: {args.category}")
            sys.exit(1)
    else:
        # Run all tests
        all_passed = runner.run_all_tests()

    # Print summary
    summary_passed = runner.print_summary()

    # Save results
    runner.save_results(args.output)

    # Exit with appropriate code
    sys.exit(0 if summary_passed else 1)


if __name__ == "__main__":
    main()
