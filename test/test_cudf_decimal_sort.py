#!/usr/bin/env python3
"""
Test if cuDF supports sorting decimal columns
"""

import cudf
import pyarrow as pa
import pandas as pd

# Create test data
print("=== Testing cuDF Decimal Sort ===\n")

# Create a simple decimal column
df_pandas = pd.DataFrame({
    'id': [3, 1, 4, 1, 5, 9, 2, 6],
    'value': [30.5, 10.5, 40.5, 10.5, 50.5, 90.5, 20.5, 60.5]
})

# Convert to cuDF with decimal type
df_cudf = cudf.from_pandas(df_pandas)

# Convert 'id' column to decimal128
df_cudf['id_decimal'] = df_cudf['id'].astype('decimal128')

print("Original DataFrame:")
print(df_cudf)
print(f"\nColumn types:")
for col, dtype in df_cudf.dtypes.items():
    print(f"  {col}: {dtype}")

# Try to sort by decimal column
print("\n\nTrying to sort by decimal column...")
try:
    df_sorted = df_cudf.sort_values('id_decimal')
    print("✓ Sort successful!")
    print(df_sorted)
except Exception as e:
    print(f"✗ Sort failed: {type(e).__name__}: {e}")

# Try with compute if needed
print("\n\nTrying different approaches...")

# Approach 1: Convert to int before sorting
try:
    print("\n1. Converting decimal to int then sorting:")
    df_cudf['id_as_int'] = df_cudf['id_decimal'].astype('int64')
    df_sorted = df_cudf.sort_values('id_as_int')
    print("✓ Success!")
    print(df_sorted[['id_decimal', 'id_as_int']])
except Exception as e:
    print(f"✗ Failed: {e}")

# Approach 2: Direct decimal comparison
try:
    print("\n2. Testing decimal comparisons:")
    mask = df_cudf['id_decimal'] > 3
    print(f"Values > 3: {mask.sum()} rows")
    print("✓ Comparisons work!")
except Exception as e:
    print(f"✗ Comparisons failed: {e}")

# Check cuDF version
print(f"\n\ncuDF version: {cudf.__version__}")