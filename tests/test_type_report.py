"""
Simple script to generate type test report
"""
import subprocess
import sys


def run_type_tests():
    """Run all type tests and generate report"""
    print("Running type tests...")

    # Run the type tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/e2e/test_all_types.py", "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    # Parse output for test results
    lines = result.stdout.split("\n")

    # Type test results
    type_results = {}

    print("\n" + "=" * 100)
    print("TYPE TEST MATRIX - gpupgparser")
    print("=" * 100)
    print(f"{'Type':<15} {'Status':<20} {'Details':<60}")
    print("-" * 100)

    # Define all types
    all_types = [
        ("SMALLINT", "smallint"),
        ("INTEGER", "integer"),
        ("BIGINT", "bigint"),
        ("REAL", "real"),
        ("DOUBLE", "double"),
        ("NUMERIC", "numeric"),
        ("BOOLEAN", "boolean"),
        ("TEXT", "text"),
        ("VARCHAR", "varchar"),
        ("CHAR", "char"),
        ("BYTEA", "bytea"),
        ("DATE", "date"),
        ("TIMESTAMP", "timestamp"),
        ("TIMESTAMPTZ", "timestamptz"),
        ("TIME", "time"),
        ("UUID", "uuid"),
        ("JSON", "json"),
    ]

    # Parse test output
    for line in lines:
        if "test_type_full_pipeline" in line:
            for type_display, type_key in all_types:
                if f"[{type_key}-" in line:
                    if "PASSED" in line:
                        type_results[type_key] = ("✅ PASSED", "All tests passed")
                    elif "FAILED" in line:
                        type_results[type_key] = ("❌ FAILED", "Test failed")
                    elif "SKIPPED" in line:
                        type_results[type_key] = ("⚫ SKIPPED", "Test skipped")

    # Also check stdout for our print messages
    for line in result.stdout.split("\n"):
        if "✅" in line and ":" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                type_name = parts[0].strip().replace("✅", "").strip()
                for type_display, type_key in all_types:
                    if type_key == type_name:
                        if type_key in type_results and type_results[type_key][0] == "✅ PASSED":
                            type_results[type_key] = ("✅ PASSED", parts[1].strip())
        elif "⚫" in line and ":" in line:
            parts = line.split(":")
            if len(parts) >= 2:
                type_name = parts[0].strip().replace("⚫", "").strip()
                for type_display, type_key in all_types:
                    if type_key == type_name:
                        type_results[type_key] = ("⚫ NOT IMPL", parts[1].strip())

    # Print results
    implemented_count = 0
    for type_display, type_key in all_types:
        if type_key in type_results:
            status, details = type_results[type_key]
            if "PASSED" in status:
                implemented_count += 1
        else:
            status, details = "⚫ UNKNOWN", "Not tested"

        print(f"{type_display:<15} {status:<20} {details:<60}")

    print("\n" + "=" * 100)
    print(
        f"Summary: {implemented_count}/{len(all_types)} types fully supported ({implemented_count/len(all_types)*100:.1f}%)"
    )
    print("=" * 100)

    # Show failed test details
    if "FAILED" in result.stdout:
        print("\nFAILED TEST DETAILS:")
        print("-" * 100)
        in_failure = False
        for line in lines:
            if "FAILURES" in line:
                in_failure = True
            elif "short test summary" in line:
                in_failure = False
            elif in_failure and line.strip():
                print(line)


if __name__ == "__main__":
    run_type_tests()
