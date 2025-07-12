#!/usr/bin/env python3
"""
Test the new --sort feature in show_parquet_sample.py
"""

import subprocess
import sys

def test_sort_feature():
    """Test the sort and gap detection feature"""
    print("=== Testing Sort Feature ===\n")
    
    # Test 1: Sort by c_custkey
    print("1. Testing sort by c_custkey:")
    cmd = ['python', 'tools/show_parquet_sample.py', '--sort', 'c_custkey', '--dir', 'output']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Check if sort analysis output is present
        if "欠落値分析" in result.stdout:
            print("✓ Sort analysis found in output")
            # Extract key information
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['最小値:', '最大値:', '実際の行数:', '期待される行数:', '欠落数:', '欠落なし']):
                    print(f"  {line.strip()}")
        else:
            print("✗ Sort analysis not found in output")
    else:
        print(f"✗ Command failed: {result.stderr}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Sort with filter
    print("2. Testing sort with filter (c_nationkey=10):")
    cmd = ['python', 'tools/show_parquet_sample.py', '--sort', 'c_custkey', '--filter', 'c_nationkey=10', '--dir', 'output']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        if "欠落値分析" in result.stdout:
            print("✓ Sort analysis with filter successful")
            # Show filtered results count
            for line in result.stdout.split('\n'):
                if "c_nationkey = 10" in line or "実際の行数:" in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ Sort analysis not found")
    else:
        print(f"✗ Command failed: {result.stderr}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Invalid column type
    print("3. Testing sort with non-integer column (c_name):")
    cmd = ['python', 'tools/show_parquet_sample.py', '--sort', 'c_name', '--dir', 'output']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "ソートは整数型カラムのみサポート" in result.stdout:
        print("✓ Correctly rejected non-integer column")
    else:
        print("✗ Should have rejected non-integer column")
    
    print("\n" + "="*50 + "\n")
    
    # Test 4: Non-existent column
    print("4. Testing sort with non-existent column:")
    cmd = ['python', 'tools/show_parquet_sample.py', '--sort', 'invalid_column', '--dir', 'output']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "ソートカラム 'invalid_column' が存在しません" in result.stdout:
        print("✓ Correctly handled non-existent column")
    else:
        print("✗ Should have reported non-existent column")

if __name__ == "__main__":
    test_sort_feature()