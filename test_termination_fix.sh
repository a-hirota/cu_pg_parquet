#!/bin/bash

# Test script to verify the 0xFFFF termination fix for customer table
set -e

# Set the table name
export TABLE_NAME="customer"
export GPUPGPARSER_TEST_MODE="1"

# Set parallel workers and chunks for customer table
export RUST_PARALLEL_CONNECTIONS="8"
export TOTAL_CHUNKS="2"

echo "=== Testing customer table with 0xFFFF prefix fix ==="
echo "Table: $TABLE_NAME"
echo "Workers: $RUST_PARALLEL_CONNECTIONS"
echo "Chunks: $TOTAL_CHUNKS"
echo

# Clean up old test files
rm -rf test_binaries/*

# Run the single chunk processor
echo "=== Running single chunk processor for chunk 0 ==="
export CHUNK_ID="0"
/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk

echo
echo "=== Running single chunk processor for chunk 1 ==="
export CHUNK_ID="1"
/home/ubuntu/gpupgparser/rust_bench_optimized/target/release/pg_fast_copy_single_chunk

# Copy binaries for testing
echo
echo "=== Copying binaries for testing ==="
timestamp=$(date +%Y%m%d_%H%M%S)
test_dir="test_binaries/$timestamp"
mkdir -p "$test_dir"
cp /dev/shm/${TABLE_NAME}_chunk_*.bin "$test_dir/"

echo "Binaries copied to: $test_dir"

# Search for PGCOPY headers and check for 0xFFFF
echo
echo "=== Searching for PGCOPY headers without 0xFFFF ==="
python3 /home/ubuntu/gpupgparser/tools/show_bin_rust.py --dir "$test_dir" --search "50 47 43 4F 50 59 0A FF 0D 0A"

echo
echo "=== Done ==="