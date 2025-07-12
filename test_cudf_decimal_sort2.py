#!/usr/bin/env python3
"""
Test cuDF decimal columns from parquet
"""

import cudf
import os

# Read a parquet file that has decimal columns
print("=== Testing cuDF Decimal from Parquet ===\n")

if os.path.exists("output/customer_chunk_0_queue.parquet"):
    # Read just a small sample
    df = cudf.read_parquet("output/customer_chunk_0_queue.parquet", nrows=10)
    
    print("DataFrame columns and types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    print("\nFirst few rows of c_custkey:")
    print(df['c_custkey'].head())
    
    # Check if we can sort
    print("\n\nTrying to sort by c_custkey (decimal)...")
    try:
        df_sorted = df.sort_values('c_custkey')
        print("✓ Sort successful!")
        print("\nSorted c_custkey values:")
        print(df_sorted['c_custkey'])
    except Exception as e:
        print(f"✗ Sort failed: {type(e).__name__}: {e}")
        
        # Try workaround
        print("\nTrying workaround - convert to pandas, sort, convert back...")
        try:
            df_pandas = df.to_pandas()
            df_pandas_sorted = df_pandas.sort_values('c_custkey')
            df_sorted = cudf.from_pandas(df_pandas_sorted)
            print("✓ Workaround successful!")
            print(df_sorted['c_custkey'])
        except Exception as e2:
            print(f"✗ Workaround also failed: {e2}")
    
    # Check specific decimal operations
    print("\n\nChecking decimal operations:")
    
    # Min/max
    try:
        print(f"Min: {df['c_custkey'].min()}")
        print(f"Max: {df['c_custkey'].max()}")
        print("✓ Min/max work")
    except Exception as e:
        print(f"✗ Min/max failed: {e}")
    
    # Comparison
    try:
        mask = df['c_custkey'] > 5277333
        print(f"Values > 5277333: {mask.sum()} rows")
        print("✓ Comparisons work")
    except Exception as e:
        print(f"✗ Comparisons failed: {e}")
        
else:
    print("No parquet file found at output/customer_chunk_0_queue.parquet")