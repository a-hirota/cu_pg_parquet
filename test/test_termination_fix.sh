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

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Use environment variable or relative path
RUST_BINARY="${GPUPGPARSER_RUST_BINARY:-$PROJECT_ROOT/rust_pg_binary_extractor/target/release/pg_chunk_extractor}"

# Run the single chunk processor
echo "=== Running single chunk processor for chunk 0 ==="
export CHUNK_ID="0"
"$RUST_BINARY"

echo
echo "=== Running single chunk processor for chunk 1 ==="
export CHUNK_ID="1"
"$RUST_BINARY"

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
python3 "$PROJECT_ROOT/tools/show_bin_rust.py" --dir "$test_dir" --search "50 47 43 4F 50 59 0A FF 0D 0A"

echo
echo "=== Done ==="