# Testing Tools Documentation

This directory contains tools for managing and analyzing the test suite for the GPU PostgreSQL Parser project.

## Available Tools

### 1. update-test-plan.py

Updates the test plan based on current source code analysis.

**Purpose:**
- Analyzes source code to find new components and functions
- Extracts supported data types from the codebase
- Updates testPlan.md with current implementation status
- Creates a backup of the previous test plan

**Usage:**
```bash
python docs/testing/update-test-plan.py
```

**Output:**
- Updates `testPlan.md` with current information
- Creates `testPlan.md.backup` as a backup

### 2. check-test-implement.py

Checks test implementation against the test plan and verifies production code usage.

**Purpose:**
- Parses testPlan.md to understand planned tests
- Analyzes actual test files for implementation quality
- Checks if tests use production code vs mocks
- Generates a comprehensive report on test coverage

**Usage:**
```bash
python docs/testing/check-test-implement.py
```

**Output:**
- `test-implementation-report.md`: Human-readable report
- `test-implementation-data.json`: Machine-readable data

## Reports

### Test Implementation Report

The implementation report includes:

1. **Overall Statistics**
   - Total test files
   - Tests using production code vs mocks
   - Quality metrics (assertions, edge cases, GPU usage)

2. **Function Coverage**
   - Coverage for each pipeline function (F1, F2, F3)
   - E2E test status
   - Real code usage verification

3. **Data Type Coverage**
   - Type matrix test implementation
   - Coverage percentage
   - List of tested PostgreSQL types

4. **Detailed Analysis**
   - Individual test file analysis
   - Quality checks per test
   - Recommendations for improvement

## Example Workflow

1. **Update Test Plan:**
   ```bash
   # Analyze current codebase and update test plan
   python docs/testing/update-test-plan.py
   ```

2. **Check Implementation:**
   ```bash
   # Verify tests are properly implemented
   python docs/testing/check-test-implement.py
   ```

3. **Review Reports:**
   - Check `test-implementation-report.md` for issues
   - Address any recommendations
   - Ensure high production code usage percentage

## Key Metrics to Monitor

- **Production Code Usage**: Should be > 90%
- **Type Coverage**: Should cover all supported PostgreSQL types
- **GPU Test Coverage**: Essential for CUDA kernel validation
- **Edge Case Testing**: Critical for data integrity

## Integration with CI/CD

These tools can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Check Test Implementation
  run: |
    python docs/testing/update-test-plan.py
    python docs/testing/check-test-implement.py

    # Check if production code usage is sufficient
    python -c "
    import json
    with open('docs/testing/test-implementation-data.json') as f:
        data = json.load(f)
    usage = data['quality']['overall']['using_production_code']
    total = data['quality']['overall']['total_test_files']
    percentage = (usage / total * 100) if total > 0 else 0
    if percentage < 90:
        print(f'ERROR: Only {percentage:.1f}% of tests use production code')
        exit(1)
    "
```
