#!/usr/bin/env python3
"""
Test script to verify decimal field validation fix
"""

import subprocess
import os
import sys
import time

def run_test():
    """Run the parser with test mode to check for c_custkey=0 duplicates"""
    print("=== Testing Decimal Field Validation Fix ===")
    print()
    
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
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # Check output for decimal field detection
    if "Decimal型フィールド:" in result.stdout:
        print("✓ Decimal field detection is working")
        for line in result.stdout.split('\n'):
            if "Decimal型フィールド:" in line:
                print(f"  {line.strip()}")
    else:
        print("✗ Decimal field detection not found in output")
    
    # Check for any errors
    if result.returncode != 0:
        print(f"\n✗ Command failed with return code {result.returncode}")
        print("\nSTDERR:")
        print(result.stderr)
        return False
    
    print("\n✓ Parser completed successfully")
    
    # Now check for duplicates
    print("\nChecking for c_custkey=0 duplicates...")
    check_cmd = ['python', 'tools/check_duplicate_custkey.py']
    check_result = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if "c_custkey=0" in check_result.stdout:
        print("\n✗ FOUND c_custkey=0 duplicates (issue not fixed)")
        # Show relevant lines
        for line in check_result.stdout.split('\n'):
            if "c_custkey=0" in line or "重複" in line:
                print(f"  {line}")
    else:
        print("\n✓ No c_custkey=0 duplicates found (issue appears to be fixed!)")
    
    return True

if __name__ == "__main__":
    os.chdir("/home/ubuntu/gpupgparser")
    success = run_test()
    sys.exit(0 if success else 1)