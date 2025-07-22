# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

gpupgparser (GPU PostgreSQL Parser) is a high-performance tool that reads PostgreSQL binary data directly using the COPY BINARY protocol and processes it on GPUs. It bypasses CPU bottlenecks by transferring data directly to GPU memory and parsing it using CUDA kernels.

## Key Technologies

- **Python**: Main implementation with Numba CUDA kernels
- **Rust**: High-performance binary extraction and GPU memory transfer
- **CUDA**: GPU kernel implementations via Numba and CuPy
- **cuDF**: GPU DataFrame library for data processing
- **Apache Arrow/Parquet**: Data serialization formats

## Common Development Commands

### Python Development

```bash
# Run the main parser
python cu_pg_parquet.py

# Run with specific table
python -m gpupaser.main --table your_table --limit 10000 --parquet output.parquet

# Run tests
python test_postgres_binary_cuda_parquet.py --table large_table --output output.parquet
python test_decimal_fix.py
python test_sort_feature.py

# Run benchmarks
python benchmarks/compression_experiment.py
python benchmarks/gpu_parallel_benchmark.py
python benchmarks/gpu_pipeline_benchmark.py
```

### Rust Development

```bash
# Setup environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cudf_dev

# Build Rust Python extension
cd rust
maturin develop              # Development build
maturin develop --release    # Release build

# Build Rust binary extractors
cd rust_pg_binary_extractor
cargo build --release

# Run Rust tests
cargo test
cargo test --release
```

### Environment Variables

```bash
# PostgreSQL connection
export GPUPASER_PG_DSN="dbname=postgres user=postgres host=localhost port=5432"

# Rust configuration
export RUST_LOG=info
export RUST_PARALLEL_CONNECTIONS=16

# Testing/debugging
export GPUPGPARSER_TEST_MODE=1
export GPUPGPARSER_DEBUG=1
```

## Architecture Overview

The project consists of several key components:

1. **PostgreSQL Binary Reader**: Connects to PostgreSQL and retrieves data using COPY BINARY protocol
2. **GPU Memory Transfer**: Direct transfer of binary data to GPU memory (via Rust or Python)
3. **CUDA Kernels** (`src/cuda_kernels/`): Parse PostgreSQL binary format on GPU
4. **Data Processing**: Convert parsed data to cuDF DataFrames or Parquet files
5. **Multi-GPU Support**: Parallel processing across multiple GPUs

### Data Flow
```
PostgreSQL → COPY BINARY → Binary Data → GPU Memory → CUDA Parsing → cuDF/Parquet
```

### Key Source Directories

- `src/`: Main Python implementation
  - `cuda_kernels/`: CUDA kernel implementations for parsing
  - `readPostgres/`: PostgreSQL reading utilities
- `rust/`: Rust implementation for high-performance extraction
- `rust_pg_binary_extractor/`: Standalone Rust binary extractors
- `benchmarks/`: Performance testing scripts
- `test/`: Test files and test data

## Working with the Codebase

### Processing Large Tables

The system processes data in chunks (default 65,535 rows) to manage GPU memory efficiently. For tables larger than this, data is automatically chunked and processed sequentially or in parallel.

### GPU Memory Management

The project automatically detects available GPU memory and adjusts chunk sizes accordingly. When working with very large datasets, consider using Parquet output to avoid memory limitations.

### Testing Approach

- Unit tests are scattered throughout `/test` and `/archive/test`
- Test binary data is stored in `/test_binaries`
- Use environment variables to control test behavior

### Performance Optimization

- The Rust implementation provides faster binary extraction than Python
- Multi-GPU processing is supported for parallel chunk processing
- Parquet compression uses zstd by default for optimal performance

## Important Notes

- Requires CUDA-capable GPU and CUDA toolkit installation
- PostgreSQL database must support COPY BINARY protocol
- The project contains both experimental (`archive/`) and production code
- When modifying CUDA kernels, ensure compatibility with both Numba and CuPy
