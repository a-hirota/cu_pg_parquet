#!/usr/bin/env python3
import cudf

# Create a simple test DataFrame
df = cudf.DataFrame({
    'c_custkey': [1, 2, 3, 4, 5],
    'c_name': ['A', 'B', 'C', 'D', 'E']
})

print("Original DataFrame:")
print(df)

# Test filtering
filter_column = 'c_custkey'
filter_value = 3

print(f"\nTrying to filter {filter_column} == {filter_value}")

# Method 1: Direct filtering
try:
    filtered_df = df[df[filter_column] == filter_value]
    print("Method 1 (Direct filtering) SUCCESS:")
    print(filtered_df)
except Exception as e:
    print(f"Method 1 FAILED: {e}")

# Method 2: Using loc
try:
    mask = df[filter_column] == filter_value
    filtered_df = df.loc[mask]
    print("\nMethod 2 (Using loc) SUCCESS:")
    print(filtered_df)
except Exception as e:
    print(f"Method 2 FAILED: {e}")

# Method 3: Using query
try:
    filtered_df = df.query(f'{filter_column} == {filter_value}')
    print("\nMethod 3 (Using query) SUCCESS:")
    print(filtered_df)
except Exception as e:
    print(f"Method 3 FAILED: {e}")