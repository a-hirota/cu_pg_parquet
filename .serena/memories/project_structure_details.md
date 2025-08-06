# Project Structure Details

## Core Source Directories

### `/src` - Main Python Implementation
- `postgres_to_parquet_converter.py` - Main converter with DirectProcessor class
- `postgres_to_cudf.py` - PostgreSQL to cuDF conversion logic
- `write_parquet_from_cudf.py` - Parquet writing utilities
- `types.py` - Type definitions and PostgreSQL to Arrow mappings
- **`/cuda_kernels`** - GPU kernel implementations
  - Binary parsing kernels
  - GPU configuration utilities
- **`/postgres_reader`** - PostgreSQL connection and reading
- **`/utils`** - Utility functions

### `/rust` - Rust Python Extension
- High-performance binary extraction
- Direct GPU memory transfer
- Python bindings via PyO3
- Built with maturin

### `/rust_pg_binary_extractor` - Standalone Rust Tools
- Binary extraction utilities
- Performance-critical components
- Can be used independently

### `/tests` - Test Suite
- **`/e2e`** - End-to-end tests for complete workflows
  - `test_rust_extraction.py` - Rust extractor tests
  - `test_gpu_processing.py` - GPU processing tests
  - `test_arrow_to_parquet.py` - Conversion tests
  - `test_all_types.py` - Data type coverage tests
- **`/integration`** - Integration tests
  - `test_full_pipeline.py` - Full pipeline tests
- **`/utils`** - Test utilities and fixtures
- `conftest.py` - pytest configuration and fixtures
- `test_config.py` - Test configuration
- `run_all_tests.py` - Test runner script

### `/docs` - Documentation
- **`/testing`** - Test documentation
  - `testPlan.md` - Comprehensive test plan
- **`/benchmarks`** - Performance documentation
- **`/implementation`** - Implementation details
- `usage_guide.md` - User guide

### `/benchmarks` - Performance Testing
- `compression_experiment.py` - Compression benchmarks
- `gpu_parallel_benchmark.py` - Parallel processing tests
- `gpu_pipeline_benchmark.py` - Pipeline performance

### `/processors` - Data Processing Components
- Additional processing utilities
- Custom processors for specific use cases

## Key Entry Points

### Main Scripts
- `cu_pg_parquet.py` - Primary entry point
- `simple_test.py` - Quick testing script
- `tests/run_all_tests.py` - Complete test runner

### Configuration Files
- `pytest.ini` - pytest configuration
- `.pre-commit-config.yaml` - Code quality hooks
- `CLAUDE.md` - AI assistant instructions

## Data Flow Architecture

```
PostgreSQL Database
    ↓ (COPY BINARY)
Binary Data Stream
    ↓ (Rust/Python)
GPU Memory Transfer
    ↓ (CUDA Kernels)
Parsed Data in GPU
    ↓ (Arrow Arrays)
cuDF DataFrame
    ↓ (Compression)
Parquet File
```

## Module Dependencies

```
types.py (Type Definitions)
    ↓
postgres_reader/ (Data Acquisition)
    ↓
cuda_kernels/ (GPU Processing)
    ↓
postgres_to_cudf.py (Conversion)
    ↓
postgres_to_parquet_converter.py (Main Logic)
    ↓
write_parquet_from_cudf.py (Output)
```

## Testing Structure

```
tests/
├── Unit Tests (scattered in test_*.py files)
├── Integration Tests (/integration)
├── E2E Tests (/e2e)
│   ├── Function 1: Binary Extraction
│   ├── Function 2: GPU Processing
│   └── Function 3: Parquet Conversion
└── Performance Tests (/benchmarks)
```

## Important Patterns

### GPU Memory Management
- Uses RMM (Rapids Memory Manager)
- Chunk-based processing (default 65,535 rows)
- Automatic memory scaling

### Error Handling
- Graceful degradation for unsupported types
- Detailed error messages with context
- Test mode for debugging

### Parallel Processing
- Multi-GPU support
- Concurrent chunk processing
- Thread-safe operations
