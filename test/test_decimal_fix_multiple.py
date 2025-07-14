#!/usr/bin/env python3
"""
Test script to run multiple iterations and verify decimal field validation fix is consistent
"""

import subprocess
import os
import sys
import time

def run_single_test(iteration):
    """Run a single test iteration"""
    # Clean up previous output
    subprocess.run(['rm', '-rf', 'output/*.parquet'], shell=True, capture_output=True)
    subprocess.run(['rm', '-rf', 'test_binaries'], shell=True, capture_output=True)
    
    # Set test environment
    env = os.environ.copy()
    env['GPUPGPARSER_TEST_MODE'] = '1'
    
    # Run the parser
    cmd = [
        'python', 'cu_pg_parquet.py',
        '--test',
        '--table', 'customer',
        '--parallel', '8',
        '--chunks', '2',
        '--yes'
    ]
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Iteration {iteration}: ✗ Parser failed")
        return False
    
    # Check for duplicates
    check_cmd = ['python', 'tools/check_duplicate_custkey.py']
    check_result = subprocess.run(check_cmd, capture_output=True, text=True)
    
    has_dup = "c_custkey=0" in check_result.stdout
    
    if has_dup:
        # Count duplicates
        count = 0
        for line in check_result.stdout.split('\n'):
            if "c_custkey=0:" in line and "回出現" in line:
                try:
                    count = int(line.split(':')[1].split('回')[0].strip())
                except:
                    count = -1
        print(f"  Iteration {iteration}: ✗ Found c_custkey=0 duplicates (count={count})")
        return False
    else:
        print(f"  Iteration {iteration}: ✓ No c_custkey=0 duplicates")
        return True

def main():
    """Run multiple test iterations"""
    print("=== Testing Decimal Field Validation Fix (Multiple Iterations) ===")
    print()
    
    num_iterations = 10
    success_count = 0
    
    print(f"Running {num_iterations} iterations...")
    print()
    
    for i in range(1, num_iterations + 1):
        if run_single_test(i):
            success_count += 1
        time.sleep(0.5)  # Brief pause between tests
    
    print()
    print(f"Results: {success_count}/{num_iterations} successful")
    
    if success_count == num_iterations:
        print("\n✓ ALL TESTS PASSED! The decimal validation fix is working consistently.")
        return True
    else:
        print(f"\n✗ INCONSISTENT RESULTS: {num_iterations - success_count} failures detected")
        print("  The issue may not be fully resolved.")
        return False

if __name__ == "__main__":
    os.chdir("/home/ubuntu/gpupgparser")
    success = main()
    sys.exit(0 if success else 1)