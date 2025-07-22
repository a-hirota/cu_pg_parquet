#!/bin/bash
# Test verification script
# Run all tests and show summary

echo "=================================="
echo "GPU PostgreSQL Parser Test Suite"
echo "=================================="
echo "Date: $(date)"
echo

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

# Function to run test category
run_test_category() {
    local category=$1
    local test_path=$2

    echo "----------------------------------------"
    echo "Running: $category"
    echo "----------------------------------------"

    # Run pytest and capture output
    output=$(python -m pytest "$test_path" -v 2>&1)
    exit_code=$?

    # Parse results
    passed=$(echo "$output" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo "0")
    failed=$(echo "$output" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo "0")
    skipped=$(echo "$output" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo "0")

    # Update totals
    TOTAL_PASSED=$((TOTAL_PASSED + passed))
    TOTAL_FAILED=$((TOTAL_FAILED + failed))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + skipped))

    # Show results
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $category: PASSED${NC} (${passed} tests)"
    else
        echo -e "${RED}✗ $category: FAILED${NC} (${failed} failures)"
    fi

    if [ $skipped -gt 0 ]; then
        echo -e "  ${YELLOW}⚠ Skipped: ${skipped} tests${NC}"
    fi

    echo
}

# Test categories
echo "Running all test categories..."
echo

# E2E Tests
run_test_category "Function 0 - PostgreSQL to Arrow Schema" "tests/e2e/test_function0.py"
run_test_category "Function 1 - PostgreSQL to Queue" "tests/e2e/test_function1.py"
run_test_category "Function 2 - Queue to GPU" "tests/e2e/test_function2.py"
run_test_category "Function 3 - GPU Binary Parsing" "tests/e2e/test_function3.py"
run_test_category "Function 4 - Arrow to Parquet" "tests/e2e/test_function4.py"

# Data Type Tests
run_test_category "Numeric Types" "tests/datatypes/test_numeric_types.py"
run_test_category "String Types" "tests/datatypes/test_string_types.py"
run_test_category "DateTime Types" "tests/datatypes/test_datetime_types.py"
run_test_category "Other Types" "tests/datatypes/test_other_types.py"

# Integration Tests
run_test_category "Pipeline Components" "tests/integration/test_pipeline_components.py"
run_test_category "Full Pipeline" "tests/integration/test_full_pipeline.py"

# Summary
echo "=================================="
echo "TEST SUMMARY"
echo "=================================="
TOTAL_TESTS=$((TOTAL_PASSED + TOTAL_FAILED + TOTAL_SKIPPED))
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TOTAL_PASSED${NC}"
echo -e "${RED}Failed: $TOTAL_FAILED${NC}"
echo -e "${YELLOW}Skipped: $TOTAL_SKIPPED${NC}"
echo

# Overall result
if [ $TOTAL_FAILED -eq 0 ] && [ $TOTAL_TESTS -gt 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! 🎉${NC}"
    exit 0
else
    echo -e "${RED}❌ TESTS FAILED${NC}"
    exit 1
fi
