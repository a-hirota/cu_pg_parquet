#!/usr/bin/env python3
import cudf

# Load actual parquet file
df = cudf.read_parquet('output/customer_chunk_0_queue.parquet')

print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"c_custkey dtype: {df['c_custkey'].dtype}")

# Check some values
print(f"\nFirst 5 c_custkey values:")
print(df['c_custkey'].head())

# Try filtering
filter_column = 'c_custkey'
filter_value = 9100005

print(f"\nTrying to filter {filter_column} == {filter_value}")
print(f"Filter value type: {type(filter_value)}")

# Check if the value exists
print(f"\nChecking if value exists:")
print(f"Min c_custkey: {df['c_custkey'].min()}")
print(f"Max c_custkey: {df['c_custkey'].max()}")

# Try the comparison
try:
    mask = df[filter_column] == filter_value
    print(f"\nMask type: {type(mask)}")
    print(f"Mask shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
    print(f"Mask dtype: {mask.dtype if hasattr(mask, 'dtype') else 'No dtype'}")
    
    # Try filtering
    filtered_df = df[mask]
    print(f"\nFiltered DataFrame shape: {filtered_df.shape}")
except Exception as e:
    print(f"\nError during filtering: {e}")
    import traceback
    traceback.print_exc()