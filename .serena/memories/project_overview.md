# GPU PostgreSQL Parser (gpupgparser) - Project Overview

## Purpose
GPU PostgreSQL Parser (gpupgparser) is a high-performance tool designed to read PostgreSQL binary data directly using the COPY BINARY protocol and process it on GPUs. It bypasses CPU bottlenecks by transferring data directly to GPU memory and parsing it using CUDA kernels.

## Tech Stack
- **Python**: Main implementation language with Numba CUDA kernels
- **Rust**: High-performance binary extraction and GPU memory transfer
- **CUDA**: GPU kernel implementations via Numba and CuPy
- **cuDF**: GPU DataFrame library for data processing
- **Apache Arrow/Parquet**: Data serialization formats
- **PostgreSQL**: Source database using COPY BINARY protocol
- **Testing**: pytest with various plugins
- **Linting/Formatting**: Black, isort, flake8, pre-commit

## Project Structure
```
gpupgparser/
├── src/                        # Main Python implementation
│   ├── cuda_kernels/          # CUDA kernel implementations
│   ├── postgres_reader/       # PostgreSQL reading utilities
│   └── utils/                 # Utility functions
├── rust/                      # Rust Python extension
├── rust_pg_binary_extractor/  # Standalone Rust binary extractors
├── tests/                     # Test suite
│   ├── e2e/                  # End-to-end tests
│   ├── integration/          # Integration tests
│   └── utils/                # Test utilities
├── docs/                      # Documentation
├── benchmarks/               # Performance benchmarks
└── processors/              # Data processors
```

## Key Features
- Direct PostgreSQL COPY BINARY protocol support
- GPU-accelerated binary data parsing
- Multi-GPU parallel processing support
- Zero-copy Arrow to cuDF conversion
- Efficient Parquet output with compression
- Support for most PostgreSQL data types

## Processing Flow
1. PostgreSQL → COPY BINARY → Binary Data
2. Binary Data → GPU Memory Transfer (Rust/Python)
3. GPU Memory → CUDA Kernel Parsing
4. Parsed Data → cuDF DataFrame/Arrow Arrays
5. DataFrame → Parquet File Output
