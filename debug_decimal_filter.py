#!/usr/bin/env python3
import cudf
from decimal import Decimal

# Load actual parquet file
df = cudf.read_parquet('output/customer_chunk_0_queue.parquet')

print(f"DataFrame shape: {df.shape}")
print(f"c_custkey dtype: {df['c_custkey'].dtype}")

# Try different filter values
filter_column = 'c_custkey'
test_values = [
    9100005,  # int
    str(9100005),  # string
    Decimal('9100005'),  # decimal
]

for filter_value in test_values:
    print(f"\n{'='*50}")
    print(f"Testing filter_value: {filter_value} (type: {type(filter_value)})")
    
    try:
        # Try the comparison
        comparison_result = df[filter_column] == filter_value
        print(f"Comparison result type: {type(comparison_result)}")
        
        # Check if it's a Series
        if isinstance(comparison_result, cudf.Series):
            print(f"Comparison is a Series with shape: {comparison_result.shape}")
            print(f"Comparison dtype: {comparison_result.dtype}")
            
            # Try filtering
            filtered_df = df[comparison_result]
            print(f"Filtered DataFrame shape: {filtered_df.shape}")
            
            if len(filtered_df) > 0:
                print("SUCCESS: Found matching records!")
                print(filtered_df.head())
        else:
            print(f"Unexpected comparison result: {comparison_result}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()